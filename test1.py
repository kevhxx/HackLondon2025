import polars as pl
import pandas as pd
from tsfresh.feature_selection.selection import select_features
from sklearn.preprocessing import StandardScaler
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFdr, f_classif
import multiprocessing

# Load data with proper decimal handling (for European-style CSV)
features = pl.read_csv("data/features_processed1.csv")
patient_info = pl.read_csv("data/patient_info_processed.csv", ignore_errors=True)

# 打印两个文件的 ID 差异
features_ids = features["ID"].unique().to_list()
patient_info_ids = patient_info["ID"].unique().to_list()
print("仅存在于 features 的 ID:", set(features_ids) - set(patient_info_ids))
print("仅存在于 patient_info 的 ID:", set(patient_info_ids) - set(features_ids))

# 直接筛选两个 DataFrame 的交集 ID
dataX = features.filter(pl.col("ID").is_in(patient_info["ID"]))
dataY = patient_info.filter(pl.col("ID").is_in(features["ID"]))

# 通过 Join 对齐 ID
merged = features.join(patient_info, on="ID", how="inner")

# 筛选 HRV=0 的数据并确保 ID 唯一性（统一 ID 类型为字符串）
data_hrv0 = (
    merged.filter(pl.col("HRV") == 0)
    .with_columns(pl.col("ID").cast(pl.Utf8))  # 关键修复：统一类型
    .unique(subset=["ID"], keep="first")
    .sort("ID")
)

# 分割特征和目标变量（保留 ACC_TIME）
dataX_hrv0 = data_hrv0.select(pl.exclude(["ADHD", "HRV_TIME", "HRV"]))
dataY_hrv0 = data_hrv0.select(["ID", "ADHD"])


# 时间编码函数（保持不变）
def cyclical_time_encoding_polars(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("ACC_TIME").str.split(":").list.eval(
                pl.element().cast(pl.UInt32).fill_null(0)
            ).alias("time_parts"),
        )
        .with_columns(
            (pl.col("time_parts").list.get(0) * 3600
             + pl.col("time_parts").list.get(1) * 60
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


dataX_hrv0 = cyclical_time_encoding_polars(dataX_hrv0)


# 索引对齐函数（确保类型一致性）
def align_polars_data(X: pl.DataFrame, y: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """强制 ID 类型一致后对齐"""
    # 统一类型
    X = X.with_columns(pl.col("ID").cast(pl.Utf8))
    y = y.with_columns(pl.col("ID").cast(pl.Utf8))

    # 获取共同 ID
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


dataX_hrv0, dataY_hrv0 = align_polars_data(dataX_hrv0, dataY_hrv0)

# 类型转换（确保 ADHD 为 Float32）
dataY_hrv0 = dataY_hrv0.with_columns(
    pl.col("ADHD").cast(pl.Float32).fill_null(-1)
)

# 最终验证
assert dataX_hrv0["ID"].equals(dataY_hrv0["ID"]), f"""
索引未对齐详情：
- X 类型: {dataX_hrv0['ID'].dtype}, Y 类型: {dataY_hrv0['ID'].dtype}
- 前5个ID对比：
  X: {dataX_hrv0["ID"].head(5).to_list()}
  Y: {dataY_hrv0["ID"].head(5).to_list()}
"""

# ===================== 基础依赖导入 =====================
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
import shap


# ===================== 修复特征选择器 =====================
class EnhancedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fdr_levels=0.3, min_features=5, alpha=0.01, max_iter=20000):
        self.fdr_levels = fdr_levels
        self.min_features = max(min_features, 3)
        self.alpha = alpha
        self.max_iter = max_iter
        self.primary_cols_ = []  # 首次选择的列名
        self.primary_idx_ = []  # 首次选择的列索引
        self.secondary_mask_ = []  # 二次选择的布尔掩码
        self.scaler_ = None  # 需要初始化scaler_

    def fit(self, X, y):
        # 初始特征选择
        self._primary_selection(X, y)  # 修正方法名

        # 准备二次选择数据
        X_primary = self._get_primary_features(X)

        # 标准化处理
        self.scaler_ = StandardScaler().fit(X_primary)
        X_scaled = self.scaler_.transform(X_primary)

        # 二次特征选择
        self._secondary_selection(X_scaled, y)
        return self

    def _primary_selection(self, X, y):
        """初次特征选择（FDR/方差）"""
        # 统一使用正确的属性名
        self.primary_cols_ = []

        if hasattr(X, 'columns'):
            df_X = X
        else:
            df_X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])

        # FDR选择逻辑
        fdr = self.fdr_levels
        try:
            X_selected = select_features(df_X, y, fdr_level=fdr)
            print(X_selected)
            if len(X_selected.columns) >= self.min_features:
                self.primary_cols_ = X_selected.columns.tolist()
        except Exception as e:
            print(f"FDR {fdr} 失败: {str(e)}")

        # 保底策略
        if not self.primary_cols_:
            print("启用方差保底选择")
            variances = np.var(df_X, axis=0)
            selected_idx = np.argsort(variances)[-self.min_features:]
            self.primary_cols_ = df_X.columns[selected_idx].tolist()

        # 记录列索引
        self.primary_idx_ = [df_X.columns.get_loc(col) for col in self.primary_cols_]

    def _secondary_selection(self, X, y):
        """二次特征选择（模型筛选）"""
        try:
            en = LassoCV(
                alphas=[self.alpha],
                max_iter=self.max_iter,
                cv=3,
                random_state=42
            )
            en.fit(X, y)
            self.secondary_mask_ = en.coef_ != 0

            # 保底机制
            if np.sum(self.secondary_mask_) < self.min_features:
                print(f"二次选择特征不足({np.sum(self.secondary_mask_)}个)，启用重要性排序")
                top_idx = np.argsort(np.abs(en.coef_))[::-1][:self.min_features]
                self.secondary_mask_ = np.zeros_like(en.coef_, dtype=bool)
                self.secondary_mask_[top_idx] = True

        except Exception as e:
            print(f"模型选择失败: {str(e)}, 使用全部初选特征")
            self.secondary_mask_ = np.ones(X.shape[1], dtype=bool)

    def transform(self, X):
        # 获取首次选择特征
        X_primary = self._get_primary_features(X)

        # 标准化
        if self.scaler_ is None:
            raise NotFittedError("需要先调用fit方法")
        X_scaled = self.scaler_.transform(X_primary)

        # 应用二次选择
        X_final = X_scaled[:, self.secondary_mask_]

        # 最终维度验证
        if X_final.shape[1] == 0:
            raise ValueError("最终特征数量为0，请检查选择参数")

        assert X_final.shape[1] == len(self.get_feature_names()), "特征维度不匹配"

        return X_final

    def _get_primary_features(self, X):
        """统一获取首次选择特征"""
        if isinstance(X, pd.DataFrame):
            return X[self.primary_cols_].values
        else:
            return X[:, self.primary_idx_]

    def get_params(self, deep=True):
        return {'fdr_levels': self.fdr_levels,
                'min_features': self.min_features,
                'alpha': self.alpha}

    def set_params(self, **params):
        if 'fdr_levels' in params:
            params['fdr_levels'] = tuple(params['fdr_levels']) if isinstance(params['fdr_levels'], list) else params[
                'fdr_levels']
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_names(self):
        """获取最终选择的特征名称"""
        return [col for col, mask in zip(self.primary_cols_, self.secondary_mask_) if mask]


# ===================== 数据准备 =====================
# 转换时保留原始数据副本
raw_dataX = dataX_hrv0.to_pandas().copy()
raw_dataY = dataY_hrv0.to_pandas()["ADHD"].copy()


# 索引对齐增强版
def safe_align_index(X, y):
    """安全对齐索引的三重校验"""
    # 第一层校验：索引完全匹配
    if X.index.equals(y.index):
        return X, y

    # 第二层校验：ID列匹配
    if 'ID' in X.columns and 'ID' in y.columns:
        common_ids = np.intersect1d(X['ID'], y['ID'])
        X = X[X['ID'].isin(common_ids)].set_index('ID')
        y = y[y['ID'].isin(common_ids)].set_index('ID')
    else:
        # 第三层校验：强制重置索引
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        print("警告：无法对齐索引，已重置索引")

    return X, y


dataX_pd, dataY_pd = safe_align_index(raw_dataX, raw_dataY)
dataX_pd = dataX_pd.apply(pd.to_numeric, errors='coerce')

# 增强型特征去重（处理大小写和空格）
dataX_pd.columns = dataX_pd.columns.str.strip().str.lower()
dataX_pd = dataX_pd.loc[:, ~dataX_pd.columns.duplicated(keep='first')]

# 动态缺失值处理（保留原始数据）
missing_threshold = 0.3
missing_cols = dataX_pd.columns[dataX_pd.isna().mean() > missing_threshold]
if len(missing_cols) > 0:
    print(f"删除高缺失率列: {missing_cols.tolist()}")
    dataX_pd = dataX_pd.drop(columns=missing_cols)

# 数据分割（先分割再填充）
X_temp, X_test_raw, y_temp, y_test = train_test_split(
    dataX_pd, dataY_pd,
    test_size=0.2,
    stratify=dataY_pd if dataY_pd.nunique() > 1 else None,
    random_state=6
)

# 安全填充（用训练集中位数填充）
train_median = X_temp.median()
X_train_raw = X_temp.fillna(train_median)
X_test_raw = X_test_raw.fillna(train_median)  # 使用训练集统计量

# 最终数据校验
print(f"[数据报告] 总样本: {len(dataX_pd)} | 特征数: {X_train_raw.shape[1]}")
print(f"训练集正样本比例: {y_temp.mean():.1%} | 测试集正样本比例: {y_test.mean():.1%}")

# 特征类型转换（保留列名）
X_train_raw = X_train_raw.astype(np.float64)
X_test_raw = X_test_raw.astype(np.float64)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    dataX_pd, dataY_pd,
    test_size=0.2,
    stratify=dataY_pd,
    random_state=6
)

# ===================== 动态参数配置 =====================
n_positive = sum(y_train == 1)
safe_n_neighbors = max(1, min(3, (n_positive - 1) // 2))  # 修正1：避免除零问题

# 交叉验证策略优化
if n_positive < 5:
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True)
elif 5 <= n_positive < 20:
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True)
else:
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True)

# 模型参数配置修正
base_params = {
    'random_state': 42,
    'categorical_features': None,
    'monotonic_cst': None,
    'scoring': 'balanced_accuracy'  # 修正2：使用原生支持的正则化方式
}

if n_positive < 5:
    scoring = make_scorer(roc_auc_score, needs_proba=True)  # 修正3：需要概率预测
    model_config = {
        **base_params,
        'max_depth': 2,  # 限制树深
        'min_samples_leaf': 20  # 防止过拟合
    }
else:
    scoring = make_scorer(balanced_accuracy_score)
    model_config = {
        **base_params,
        'learning_rate': 0.05,
        'max_depth': 3,
        'class_weight': None,  # 修正4：HistGradientBoosting无此参数
        'l2_regularization': 0.1  # 改用正确的正则化参数
    }

# 安全邻居数最终校验（修正5）
safe_n_neighbors = min(safe_n_neighbors, n_positive - 1) if n_positive > 1 else 0


# ===================== 管道构建 =====================
def create_SMOTE_pipeline():
    return Pipeline([
        ('feature_selector', EnhancedFeatureSelector()),
        ('smote', SMOTE(
            sampling_strategy=0.5,  # 将少数类扩至多数类的50%
            k_neighbors=3,  # 降低k值适应小样本
            random_state=42
        )),
        ('classifier', HistGradientBoostingClassifier())
    ])


model = create_SMOTE_pipeline()

# 全量训练
model.fit(X_train_raw, y_train)

# 测试评估
test_proba = model.predict_proba(X_test_raw)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)

# 结果展示
print(f"""
    [模型报告]
    测试集AUC: {test_auc:.3f}
    SMOTE参数: k_neighbors={model.named_steps['smote'].k_neighbors}
    """)

import joblib

# 保存训练参数
joblib.dump({
    'model': model,
    'train_median': X_train_raw.median(),
    'feature_columns': X_train_raw.columns
}, 'src/utils/trained_model.pkl')

# 加载保存的模型
saved_data = joblib.load('src/utils/trained_model.pkl')
model = saved_data['model']
train_median = saved_data['train_median']
feature_columns = saved_data['feature_columns']

# 用户输入示例（需包含部分特征）

# If you want to specifically select only these columns
columns = [
    'ID', 'SEX', 'AGE', 'ACC', 'ACC_DAYS',
    'HRV_HOURS', 'CPT_II', 'ADD', 'BIPOLAR', 'UNIPOLAR',
    'ANXIETY', 'SUBSTANCE', 'OTHER', 'CT', 'MDQ_POS', 'WURS',
    'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MED', 'MED_Antidepr',
    'MED_Moodstab', 'MED_Antipsych', 'MED_Anxiety_Benzo', 'MED_Sleep',
    'MED_Analgesics_Opioids', 'MED_Stimulants', 'filter_$'
]

multiprocessing.freeze_support()

# Read CSV with specific columns
df = pd.read_csv('src/utils/data/patient_info_processed.csv', usecols=columns)
row = df[df['ID'] == 7]

print(row)


def predict_new_sample(model, user_input_df, train_median, feature_columns):
    """
    输入参数：
    - model: 训练好的Pipeline模型
    - user_input_df: 用户输入的原始DataFrame
    - train_median: 训练集各列中位数（用于填充缺失值）
    - feature_columns: 训练数据原始列名列表
    
    返回：
    - 预测概率（格式：[阴性概率, 阳性概率]）
    - 特征重要性（SHAP值）
    """
    # 1. 数据对齐
    # 确保输入包含所有训练时的特征列
    aligned_df = pd.DataFrame(columns=feature_columns)
    for col in user_input_df.columns:
        if col in feature_columns:
            aligned_df[col] = user_input_df[col]

    # 2. 缺失值填充
    filled_df = aligned_df.fillna(train_median)

    # 3. 数据类型转换
    processed_data = filled_df.astype(np.float64)

    # 4. 执行预测
    try:
        proba = model.predict_proba(processed_data)[0]
        return proba
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None, None


# 执行预测
proba = predict_new_sample(model, row, train_median, feature_columns)

print(f"""
预测结果：
- 阴性概率：{proba[0]:.1%}
- 阳性概率：{proba[1]:.1%}
""")
