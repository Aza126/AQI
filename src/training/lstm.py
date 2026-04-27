# src/training/lstm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.common.utils import load_config, logger
from src.common.schema import MODEL_INPUT_COLUMNS, TARGET_COLUMN

def train_lstm():
    config = load_config()
    logger.info("Loading training data for LSTM...")
    
    df = pd.read_parquet(config["artifacts"]["training_data_path"])
    
    X = df[MODEL_INPUT_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Reshape về 3D: (số mẫu, 1 bước thời gian, số đặc trưng)
    X_lstm = np.expand_dims(X, axis=1)
    
    # Xây dựng cấu trúc mô hình
    model = Sequential([
        LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1) # Dự báo giá trị AQI
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    logger.info("Training LSTM model...")
    model.fit(X_lstm, y, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    
    # Lưu mô hình .h5
    model_path = config["model"]["lstm"]["path"]
    # Đảm bảo thư mục tồn tại
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model.save(model_path)
    logger.info(f"LSTM model saved to {model_path}")

if __name__ == "__main__":
    train_lstm()
