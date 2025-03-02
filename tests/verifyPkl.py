import numpy as np
import inspect
import joblib

# 1. 加载模型 上一个目录的 trained_model.pkl
with open("/src/utils/trained_model.pkl", "rb") as file:
    print(f"Loading model from: {file.name}")
    model = joblib.load(file)

# 2. 查看模型的类型
print(f"Model type: {type(model)}")

# 3. 查看模型的方法和属性
print(dir(model))

# 4. 查看输入特征数量
if hasattr(model, 'n_features_in_'):
    n_features = model.n_features_in_
    print(f"Model expects {n_features} features.")
else:
    print("Cannot determine expected number of features.")

# 5. 查看 predict 方法的签名和文档
try:
    print(inspect.signature(model.predict))
    print(model.predict.__doc__)
except Exception as e:
    print(e)

# 6. 创建示例输入数据并进行预测
X_new = np.random.rand(1, n_features)
predictions = model.predict(X_new)
print(f"Predictions: {predictions}")

# 7. 查看输出格式
print(f"Predictions shape: {predictions.shape}")
