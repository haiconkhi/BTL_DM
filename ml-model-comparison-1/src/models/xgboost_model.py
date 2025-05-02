import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")
    
    data = pd.read_csv(filepath)
    
    if 'Spent' not in data.columns:
        raise ValueError("Không tìm thấy cột 'Spent' làm mục tiêu!")
    
    X = pd.get_dummies(data.drop('Spent', axis=1), drop_first=True)
    y = data['Spent']
    return X, y

def train_xgboost_model(X_train, y_train, X_val, y_val,
                        n_estimators=50, learning_rate=0.1):
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=10
    )
    
    return model

def train_and_evaluate_xgboost(filepath, test_size=0.2):
    X, y = load_data(filepath)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True
    )

    model = train_xgboost_model(X_train, y_train, X_val, y_val,
                                n_estimators=50, learning_rate=0.1)

    ev = model.evals_result()
    train_rmse = ev['validation_0']['rmse']
    val_rmse   = ev['validation_1']['rmse']
    
    for epoch in range(9, 50, 10):
        tr_mse  = train_rmse[epoch]**2
        val_mse = val_rmse[epoch]**2
        print(f"Epoch {epoch+1}/50, Train Loss: {tr_mse:.6f}, Val Loss: {val_mse:.6f}")
    
    best_epoch = min(range(len(val_rmse)), key=lambda i: val_rmse[i])
    best_val_mse = val_rmse[best_epoch]**2
    print(f"\nHuấn luyện hoàn tất. Validation loss tốt nhất: {best_val_mse:.6f}\n")
    
    preds = model.predict(X_test)
    mse  = mean_squared_error(y_test, preds)
    rmse = mse**0.5
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    
    print("Kết quả đánh giá mô hình XGBoost:")
    print(f" Mean Squared Error (MSE): {mse:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" R² Score: {r2:.4f}")
    
    return model

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    filepath = os.path.join(base_dir, 'data', 'Marketing_end.csv')
    
    model = train_and_evaluate_xgboost(filepath, test_size=0.2)