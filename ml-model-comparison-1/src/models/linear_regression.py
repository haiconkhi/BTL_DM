import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")
    
    data = pd.read_csv(filepath)
    if 'Spent' not in data.columns:
        raise ValueError("Không tìm thấy cột 'Spent' làm mục tiêu!")

    X = pd.get_dummies(data.drop('Spent', axis=1), drop_first=True)
    y = data['Spent'].values.reshape(-1, 1)
    return X, y

def train_and_evaluate_nn(
    filepath,
    test_size=0.2,
    random_state=None,
    epochs=50,
    batch_size=32
):
    # 1. Load dữ liệu
    X, y = load_data(filepath)

    # 2. Chia tập train/test
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 3. Scale dữ liệu
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    # 4. Tạo model Keras
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)  # output layer cho regression
    ])

    model.compile(optimizer='adam', loss='mse')

    # 5. Huấn luyện
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # 6. Dự đoán
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_val_scaled)

    # 7. Đánh giá
    mse = mean_squared_error(y_val_scaled, y_pred_scaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_scaled, y_pred_scaled)
    r2 = r2_score(y_val_scaled, y_pred_scaled)

    print("\nKết quả đánh giá mô hình:")
    print(f" Mean Squared Error (MSE): {mse:.4f}")
    print(f" Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" Mean Absolute Error (MAE): {mae:.4f}")
    print(f" R² Score: {r2:.4f}")

    min_val_loss = np.min(history.history['val_loss'])
    print(f"\nHuấn luyện hoàn tất. Validation loss tốt nhất: {min_val_loss:.6f}")

    return model, scaler_X, scaler_y, history

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    filepath = os.path.join(base_dir, 'data', 'Marketing_end.csv')

    model, scaler_X, scaler_y, history = train_and_evaluate_nn(
        filepath,
        test_size=0.2,
        random_state=42, 
        epochs=50
    )
