import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_ann_model(input_dim, architecture="default"):  
    if architecture == "default":
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
    elif architecture == "alternative":
        model = Sequential([
            Dense(256, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
    else:
        raise ValueError("Architecture không được hỗ trợ!")
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def train_ann_model(X_train, y_train, epochs=100, batch_size=16, architecture="default"):  
    model = build_ann_model(X_train.shape[1], architecture=architecture)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,  
        callbacks=[early_stop],
        verbose=1
    )
    
    best_val_loss = min(history.history['val_loss'])
    print(f"Huấn luyện hoàn tất. Validation loss tốt nhất: {best_val_loss:.6f}")
    return model, history


def load_and_preprocess_data(filepath):  
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {filepath}")
    
    data = pd.read_csv(filepath)
    if 'Spent' not in data.columns:
        raise ValueError("Không tìm thấy cột 'Spent' làm mục tiêu!")
    
    y = data['Spent'].values
    X = pd.get_dummies(data.drop('Spent', axis=1), drop_first=True)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    
    return X_scaled, y_scaled, scaler_y


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    filepath = os.path.join(base_dir, 'data', 'Marketing_end.csv')
    
    X, y, scaler_y = load_and_preprocess_data(filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2 
    )
    
    # Train model and get history
    model, history = train_ann_model(
        X_train, y_train,
        epochs=50,
        batch_size=10,
        architecture="default"
    )

    # Plot training and validation loss curves
    plt.figure(figsize=(14, 5))

    # Linear scale subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Log scale subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves (log scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Evaluate on test set
    y_pred_test = model.predict(X_test).flatten()
    mse_test  = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    mae_test  = mean_absolute_error(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)
    
    print("\nKết quả đánh giá model trên tập kiểm tra:")
    print(f"   Mean Squared Error (MSE):       {mse_test:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse_test:.4f}")
    print(f"   Mean Absolute Error (MAE):      {mae_test:.4f}")
    print(f"   R² Score:                        {r2_test:.4f}")
