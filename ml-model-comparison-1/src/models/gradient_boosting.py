import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")
    
    data = pd.read_csv(filepath)
    if 'Spent' not in data.columns:
        raise ValueError("Không tìm thấy cột 'Spent' làm mục tiêu!")
    
    X = pd.get_dummies(data.drop('Spent', axis=1), drop_first=True)
    y = data['Spent'].values.reshape(-1, 1)
    return X, y

def plot_learning_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss (MSE)', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss (MSE)', marker='s')
    plt.xlabel('Estimators')
    plt.ylabel('Loss (MSE)')
    plt.title('GradientBoostingRegressor Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def train_and_evaluate(
    filepath,
    test_size=0.2,
    random_state=None,
    n_estimators=50,
    **gbr_params
):
    # 1. Load dữ liệu
    X, y = load_data(filepath)
    
    # 2. Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    
    # 3. Scale X
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)
    
    # 4. Scale y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    y_test_scaled  = scaler_y.transform(y_test).ravel()
    
    # 5. Khởi tạo và train GBR
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **gbr_params
    )
    model.fit(X_train_scaled, y_train_scaled)
    
    # 6. Tính loss từng epoch
    train_losses = []
    val_losses   = []
    for i, (y_tr_pred, y_val_pred) in enumerate(zip(
            model.staged_predict(X_train_scaled),
            model.staged_predict(X_test_scaled)
        ), start=1):
        mse_tr = mean_squared_error(y_train_scaled, y_tr_pred)
        mse_va = mean_squared_error(y_test_scaled,  y_val_pred)
        train_losses.append(mse_tr)
        val_losses.append(mse_va)
        
        if i % (n_estimators // 5) == 0 or i == n_estimators:
            print(f"Epoch {i}/{n_estimators}, "
                  f"Train Loss: {mse_tr:.6f}, "
                  f"Val Loss: {mse_va:.6f}")
    
    # 7. Vẽ biểu đồ
    plot_learning_curve(train_losses, val_losses)
    
    best_val = min(val_losses)
    print("\nHuấn luyện hoàn tất. Validation loss tốt nhất: "
          f"{best_val:.6f}\n")
    
    y_test_pred = model.predict(X_test_scaled)
    
    def compute_metrics(y_true, y_pred):
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2
    
    mse_te, rmse_te, mae_te, r2_te = compute_metrics(y_test_scaled, y_test_pred)
    
    print("\nKết quả đánh giá model trên tập kiểm tra (scaled):")
    print(f" Mean Squared Error (MSE):  {mse_te:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse_te:.4f}")
    print(f" Mean Absolute Error (MAE):  {mae_te:.4f}")
    print(f" R² Score: {r2_te:.4f}")
    
    return model, scaler_X, scaler_y

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    filepath = os.path.join(base_dir, 'data', 'Marketing_end.csv')
    
    model, scaler_X, scaler_y = train_and_evaluate(
        filepath,
        test_size=0.2,
        random_state=None,
        n_estimators=50,          
        learning_rate=0.1,
        max_depth=3
    )
