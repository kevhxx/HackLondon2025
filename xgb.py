import polars as pl
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

patient_info = pl.read_csv(
    "src/utils/data/patient_info_processed.csv", 
    ignore_errors=True,
    # Add decimal handling if needed (e.g., European commas)
    # decimal=",",  # Uncomment if CSV uses commas for decimals <button class="citation-flag" data-index="1"><button class="citation-flag" data-index="5"><button class="citation-flag" data-index="7">
)

# Process data (simplified)
data = (
    patient_info
    .with_columns(pl.col("ID").cast(pl.Utf8))  # Ensure ID is string <button class="citation-flag" data-index="3">
    .unique(subset=["ID"], keep="first")       # Deduplicate by ID
    .sort("ID")
)

# 分割特征和目标变量（保留 ACC_TIME）
dataX = data.select(pl.exclude(["ADHD", "ACC_TIME", "HRV_TIME"]))
print(dataX)
dataY = data.select(["ID", "ADHD"])

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

dataX, dataY = align_polars_data(dataX, dataY)
# 最终验证
assert dataX["ID"].equals(dataY["ID"]), f"""
索引未对齐详情：
- X 类型: {dataX['ID'].dtype}, Y 类型: {dataY['ID'].dtype}
- 前5个ID对比：
  X: {dataX["ID"].head(5).to_list()}
  Y: {dataY["ID"].head(5).to_list()}
"""
dataX = dataX.drop(dataX.columns[0])
dataY = dataY.drop(dataY.columns[0])
# 类型转换（确保 ADHD 为 Float32）
dataY = dataY.with_columns(
    pl.col("ADHD").cast(pl.Float32).fill_null(-1)
)

##Train-Test-Split

X_train, X_test, y_train, y_test = train_test_split(
        dataX, dataY,
        test_size=0.2,
        random_state=6
    )

# Example for classification
model = XGBClassifier(
    objective='binary:logistic',  # Use 'reg:squarederror' for regression
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=6  # Matches the random_state in train_test_split <button class="citation-flag" data-index="2"><button class="citation-flag" data-index="5">
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")  # <button class="citation-flag" data-index="8">

# Save model
joblib.dump(model, "xgboost_model.pkl")  # <button class="citation-flag" data-index="9">

# Load model
model = joblib.load("xgboost_model.pkl")  # <button class="citation-flag" data-index="5"><button class="citation-flag" data-index="6">

# Predict on new data (convert to Pandas first)
data = {"SEX":"1","AGE":"1","ACC":"1", "ACC_DAYS" :"3", "HRV":"1","HRV_HOURS":"10","CPT_II":"2","ADD":"1","BIPOLAR":"1","UNIPOLAR":"9","ANXIETY":"10","SUBSTANCE":"9","OTHER":"0","CT":"9","MDQ_POS":"0","WURS":"98","ASRS":"5","MADRS":"10000","HADS_A":"10","HADS_D":"10","MED":"1","MED_Antidepr":"1","MED_Moodstab":"1","MED_Antipsych":"1","MED_Anxiety_Benzo":"1","MED_Sleep":"1","MED_Analgesics_Opioids":"1","MED_Stimulants":"1"}
df = pl.from_dict(data)

missing_features = set(X_train.columns) - set(df.columns)
for feat in missing_features:
    df = df.with_columns(pl.lit(0).alias(feat))  # Use Polars' `with_columns` <button class="citation-flag" data-index="5"><button class="citation-flag" data-index="6">

probabilities = model.predict_proba(df)  # <button class="citation-flag" data-index="8">

print(probabilities)