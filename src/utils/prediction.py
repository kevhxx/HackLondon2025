import pandas as pd
import numpy as np
import multiprocessing
import os
import cloudpickle
import joblib
import polars as pl

from src.utils.xgb import X_train

# 加载保存的模型
print("Loading model..." + "    " + os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl'))
saved_data = cloudpickle.load(open(os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl'), 'rb'))
# model = saved_data['model']
# train_median = saved_data['train_median']
# feature_columns = saved_data['feature_columns']

# 用户输入示例（需包含部分特征）
# If you want to specifically select only these columns
columns = [
    'SEX', 'AGE', 'ACC', 'ACC_DAYS',
    'HRV_HOURS', 'CPT_II', 'ADD', 'BIPOLAR', 'UNIPOLAR',
    'ANXIETY', 'SUBSTANCE', 'OTHER', 'CT', 'MDQ_POS', 'WURS',
    'ASRS', 'MADRS', 'HADS_A', 'HADS_D', 'MED', 'MED_Antidepr',
    'MED_Moodstab', 'MED_Antipsych', 'MED_Anxiety_Benzo', 'MED_Sleep',
    'MED_Analgesics_Opioids', 'MED_Stimulants', 'HRV'
]

multiprocessing.freeze_support()


# Read CSV with specific columns
# os.getcwd() 的上一层目录的 data/patient_info_processed.csv
# if __name__ == '__main__':
#     df = pd.read_csv("/src/utils/data/patient_info_processed.csv", usecols=columns)
# else:
#     df = pd.read_csv(os.path.join(os.getcwd(), 'src/utils', 'data', 'patient_info_processed.csv'), usecols=columns)
# row = df[df['ID'] == 7]


# def predict_new_sample(model, user_input_df, train_median, feature_columns):
#     """
#     输入参数：
#     - model: 训练好的Pipeline模型
#     - user_input_df: 用户输入的原始DataFrame
#     - train_median: 训练集各列中位数（用于填充缺失值）
#     - feature_columns: 训练数据原始列名列表
#
#     返回：
#     - 预测概率（格式：[阴性概率, 阳性概率]）
#     - 特征重要性（SHAP值）
#     """
#     # 1. 数据对齐
#     # 确保输入包含所有训练时的特征列
#     aligned_df = pd.DataFrame(columns=feature_columns)
#     for col in user_input_df.columns:
#         if col in feature_columns:
#             aligned_df[col] = user_input_df[col]
#
#     # 2. 缺失值填充
#     filled_df = aligned_df.fillna(train_median)
#
#     # 3. 数据类型转换
#     processed_data = filled_df.astype(np.float64)
#
#     # 4. 执行预测
#     try:
#         proba = model.predict_proba(processed_data)[0]
#         return proba
#     except Exception as e:
#         print(f"预测失败: {str(e)}")
#         return None, None


def predict_new_sample(model, user_input_df, train_median, feature_columns):
    """Predict probabilities for a new sample."""
    # 统一列名格式
    user_input_df.columns = user_input_df.columns.str.strip().str.lower()
    feature_columns = [col.lower() for col in feature_columns]

    # 对齐列
    aligned_df = user_input_df.reindex(columns=feature_columns)

    # 填充缺失值
    filled_df = aligned_df.fillna(train_median)

    # 数据类型转换
    processed_data = filled_df.astype(np.float64)

    # 检查是否有有效数据
    if processed_data.shape[0] == 0:
        print("No valid data for prediction after alignment.")
        return None

    # 执行预测
    try:
        proba = model.predict_proba(processed_data)[0]
        return proba
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None


#
# # 执行预测
# proba = predict_new_sample(model, row, train_median, feature_columns)

# print(f"""
# 预测结果：
# - 阴性概率：{proba[0]:.1%}
# - 阳性概率：{proba[1]:.1%}
# """)
#
# async def do_prediction(df: pd.DataFrame) -> list[float]:
#     proba = predict_new_sample(model, df, train_median, feature_columns)
#     # print(f"""
#     # 预测结果：
#     # - 阴性概率：{proba[0]:.1%}
#     # - 阳性概率：{proba[1]:.1%}
#     # """)
#     return proba

model = joblib.load(os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl'))


async def do_prediction(df: pd.DataFrame) -> list[float]:
    df = pl.from_dict(data)
    missing_features = set(X_train.columns) - set(df.columns)
    for feat in missing_features:
        df = df.with_columns(pl.lit(0).alias(
            feat))  # Use Polars' `with_columns` <button class="citation-flag" data-index="5"><button class="citation-flag" data-index="6">

    probabilities = model.predict_proba(df)  # <button class="citation-flag" data-index="8">
    return probabilities


if __name__ == '__main__':
    # 执行预测
    data = {"SEX": "0", "AGE": "1", "ACC": "0", "HRV": "1", "HRV_TIME": "11:11", "HRV_HOURS": "2", "CPT_II": "999",
            "ADHD": "1", "ADD": "1", "BIPOLAR": "9", "UNIPOLAR": "9", "ANXIETY": "1", "SUBSTANCE": "9", "OTHER": "0",
            "CT": "9", "MDQ_POS": "0", "WURS": "98", "ASRS": "5", "MADRS": "11", "HADS_A": "10", "HADS_D": "10",
            "MED": "1"}
    row = pd.DataFrame([data])
    import asyncio
    proba = asyncio.run(do_prediction(row))
    print(proba)