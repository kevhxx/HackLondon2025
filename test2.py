import polars as pl
import pandas as pd
from tsfresh.feature_selection.selection import select_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LassoCV
from imblearn.pipeline import Pipeline
import multiprocessing
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

# --- Helper Functions ---

def cyclical_time_encoding_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Encode time cyclically using sine and cosine transformations."""
    return (
        df.with_columns(
            pl.col("ACC_TIME").str.split(":").list.eval(
                pl.element().cast(pl.UInt32).fill_null(0)
            ).alias("time_parts"),
        )
        .with_columns(
            (pl.col("time_parts").list.get(0)*3600 
             + pl.col("time_parts").list.get(1)*60 
             + pl.col("time_parts").list.get(2)
            ).alias("total_seconds")
        )
        .with_columns(
            (2 * np.pi * pl.col("total_seconds") / 86400).alias("radians")
        )
        .with_columns(
            pl.col("radians").sin().alias("ACC_TIME_SIN"),
            pl.col("radians").cos().alias("ACC_TIME_COS")
        )
        .drop(["time_parts", "total_seconds", "radians", "ACC_TIME"])
    )

def align_polars_data(X: pl.DataFrame, y: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align Polars DataFrames by common IDs."""
    X = X.with_columns(pl.col("ID").cast(pl.Utf8))
    y = y.with_columns(pl.col("ID").cast(pl.Utf8))
    common_ids = (
        X.select("ID")
        .join(y.select("ID"), on="ID", how="semi")
        .unique()
        .sort("ID")
    )
    return (
        X.join(common_ids, on="ID").sort("ID"),
        y.join(common_ids, on="ID").sort("ID")
    )

def safe_align_index(X, y):
    """Safely align indices of X and y DataFrames."""
    if X.index.equals(y.index):
        return X, y
    if 'ID' in X.columns and 'ID' in y.columns:
        common_ids = np.intersect1d(X['ID'], y['ID'])
        X = X[X['ID'].isin(common_ids)].set_index('ID')
        y = y[y['ID'].isin(common_ids)].set_index('ID')
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print("Warning: Unable to align indices, resetting indices.")
    return X, y

# --- Custom Feature Selector ---

class EnhancedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_levels=0.3, min_features=5, alpha=0.01, max_iter=20000):
        self.fdr_levels = fdr_levels
        self.min_features = max(min_features, 3)
        self.alpha = alpha
        self.max_iter = max_iter
        self.primary_cols_ = []
        self.primary_idx_ = []
        self.secondary_mask_ = []
        self.scaler_ = None

    def fit(self, X, y):
        self._primary_selection(X, y)
        X_primary = self._get_primary_features(X)
        self.scaler_ = StandardScaler().fit(X_primary)
        X_scaled = self.scaler_.transform(X_primary)
        self._secondary_selection(X_scaled, y)
        return self

    def _primary_selection(self, X, y):
        self.primary_cols_ = []
        if hasattr(X, 'columns'):
            df_X = X
        else:
            df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        fdr = self.fdr_levels
        try:
            X_selected = select_features(df_X, y, fdr_level=fdr)
            if len(X_selected.columns) >= self.min_features:
                self.primary_cols_ = X_selected.columns.tolist()
        except Exception as e:
            print(f"FDR {fdr} failed: {str(e)}")
        if not self.primary_cols_:
            print("Falling back to variance-based selection.")
            variances = np.var(df_X, axis=0)
            selected_idx = np.argsort(variances)[-self.min_features:]
            self.primary_cols_ = df_X.columns[selected_idx].tolist()
        self.primary_idx_ = [df_X.columns.get_loc(col) for col in self.primary_cols_]

    def _secondary_selection(self, X, y):
        try:
            en = LassoCV(alphas=[self.alpha], max_iter=self.max_iter, cv=3, random_state=42)
            en.fit(X, y)
            self.secondary_mask_ = en.coef_ != 0
            if np.sum(self.secondary_mask_) < self.min_features:
                print(f"Secondary selection yielded too few features ({np.sum(self.secondary_mask_)}), using top importance.")
                top_idx = np.argsort(np.abs(en.coef_))[::-1][:self.min_features]
                self.secondary_mask_ = np.zeros_like(en.coef_, dtype=bool)
                self.secondary_mask_[top_idx] = True
        except Exception as e:
            print(f"Model selection failed: {str(e)}, using all primary features.")
            self.secondary_mask_ = np.ones(X.shape[1], dtype=bool)

    def transform(self, X):
        X_primary = self._get_primary_features(X)
        if self.scaler_ is None:
            raise NotFittedError("This selector must be fitted first.")
        X_scaled = self.scaler_.transform(X_primary)
        X_final = X_scaled[:, self.secondary_mask_]
        if X_final.shape[1] == 0:
            raise ValueError("No features selected, check selection parameters.")
        assert X_final.shape[1] == len(self.get_feature_names()), "Feature dimension mismatch."
        return X_final

    def _get_primary_features(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.primary_cols_].values
        else:
            return X[:, self.primary_idx_]

    def get_feature_names(self):
        return [col for col, mask in zip(self.primary_cols_, self.secondary_mask_) if mask]

# --- Pipeline and Prediction Functions ---

def create_SMOTE_pipeline():
    """Create a pipeline with feature selection, SMOTE, and a classifier."""
    return Pipeline([
        ('feature_selector', EnhancedFeatureSelector()),
        ('smote', SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=42)),
        ('classifier', HistGradientBoostingClassifier())
    ])

def predict_new_sample(model, user_input_df, train_median, feature_columns):
    """Predict probabilities for a new sample."""
    aligned_df = pd.DataFrame(columns=feature_columns)
    for col in user_input_df.columns:
        if col in feature_columns:
            aligned_df[col] = user_input_df[col]
    filled_df = aligned_df.fillna(train_median)
    processed_data = filled_df.astype(np.float64)
    try:
        proba = model.predict_proba(processed_data)[0]
        return proba
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None

# --- Main Execution ---

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Ensure multiprocessing compatibility

    # Load data
    features = pl.read_csv("data/features_processed1.csv")
    patient_info = pl.read_csv("data/patient_info_processed.csv", ignore_errors=True)

    # Data preprocessing
    features_ids = features["ID"].unique().to_list()
    patient_info_ids = patient_info["ID"].unique().to_list()
    print("IDs only in features:", set(features_ids) - set(patient_info_ids))
    print("IDs only in patient_info:", set(patient_info_ids) - set(features_ids))

    dataX = features.filter(pl.col("ID").is_in(patient_info["ID"]))
    dataY = patient_info.filter(pl.col("ID").is_in(features["ID"]))
    merged = features.join(patient_info, on="ID", how="inner")
    data_hrv0 = (
        merged.filter(pl.col("HRV") == 0)
        .with_columns(pl.col("ID").cast(pl.Utf8))
        .unique(subset=["ID"], keep="first")
        .sort("ID")
    )
    dataX_hrv0 = data_hrv0.select(pl.exclude(["ADHD", "HRV_TIME", "HRV"]))
    dataY_hrv0 = data_hrv0.select(["ID", "ADHD"])
    dataX_hrv0 = cyclical_time_encoding_polars(dataX_hrv0)
    dataX_hrv0, dataY_hrv0 = align_polars_data(dataX_hrv0, dataY_hrv0)
    dataY_hrv0 = dataY_hrv0.with_columns(pl.col("ADHD").cast(pl.Float32).fill_null(-1))

    # Convert to pandas
    raw_dataX = dataX_hrv0.to_pandas().copy()
    raw_dataY = dataY_hrv0.to_pandas()["ADHD"].copy()
    dataX_pd, dataY_pd = safe_align_index(raw_dataX, raw_dataY)
    dataX_pd = dataX_pd.apply(pd.to_numeric, errors='coerce')
    dataX_pd.columns = dataX_pd.columns.str.strip().str.lower()
    dataX_pd = dataX_pd.loc[:, ~dataX_pd.columns.duplicated(keep='first')]

    # Handle missing values and split data
    missing_threshold = 0.3
    missing_cols = dataX_pd.columns[dataX_pd.isna().mean() > missing_threshold]
    if len(missing_cols) > 0:
        print(f"Dropping columns with high missing rates: {missing_cols.tolist()}")
        dataX_pd = dataX_pd.drop(columns=missing_cols)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        dataX_pd, dataY_pd,
        test_size=0.2,
        stratify=dataY_pd,
        random_state=6
    )
    train_median = X_train_raw.median()
    X_train_raw = X_train_raw.fillna(train_median)
    X_test_raw = X_test_raw.fillna(train_median)

    # Train the model
    model = create_SMOTE_pipeline()
    model.fit(X_train_raw, y_train)

    # Save the model
    joblib.dump({
        'model': model,
        'train_median': X_train_raw.median(),
        'feature_columns': X_train_raw.columns
    }, 'trained_model.pkl')

    # Load the model
    saved_data = joblib.load('trained_model.pkl')
    model = saved_data['model']
    train_median = saved_data['train_median']
    feature_columns = saved_data['feature_columns']

    # Prediction
    columns = [
        'ID', 'SEX', 'AGE', 'ACC', 'ACC_DAYS',
        'HRV_HOURS', 'CPT_II', 'ADD', 'BIPOLAR', 'UNIPOLAR',
        'ANXIETY', 'SUBSTANCE', 'OTHER', 'CT', 'MDQ_POS', 'WURS',
        'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MED', 'MED_Antidepr',
        'MED_Moodstab', 'MED_Antipsych', 'MED_Anxiety_Benzo', 'MED_Sleep',
        'MED_Analgesics_Opioids', 'MED_Stimulants', 'filter_$'
    ]
    df = pd.read_csv('patient_info_processed.csv', usecols=columns)
    row = df[df['ID'] == 9]
    
    proba = predict_new_sample(model, row, train_median, feature_columns)

    print(f"""
    Prediction Results:
    - Negative Probability: {proba[0]:.1%}
    - Positive Probability: {proba[1]:.1%}
    """)