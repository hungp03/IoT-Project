import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics

# Tạo dữ liệu giả
X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Chuyển đổi dữ liệu thành DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Đặt tham số cho mô hình, sử dụng GPU
params = {
    'tree_method': 'hist',  
    'device': 'cuda',
    'max_depth': 5,
    'learning_rate': 0.1,
    'objective': 'binary:logistic', # Cho bài toán phân loại nhị phân
    'eval_metric': 'logloss',      # Đánh giá theo logloss             
}

# Huấn luyện mô hình
model = xgb.train(params, dtrain, num_boost_round=100)

# Dự đoán với mô hình đã huấn luyện
y_pred_gpu = model.predict(dtest)

# Chuyển đổi xác suất thành nhãn (0 hoặc 1) sử dụng ngưỡng 0.5
y_pred_gpu_binary = (y_pred_gpu > 0.5).astype(int)

f1_gpu = sklearn.metrics.f1_score(y_test, y_pred_gpu_binary, average="macro")

print("Prediction done")
print(f"F1 Score on GPU: {f1_gpu}")
