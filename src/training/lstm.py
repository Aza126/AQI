# src/training/lstm.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from src.common.utils import load_config, logger
from src.common.schema import MODEL_INPUT_COLUMNS, TARGET_COLUMN, META_COLUMN, TIME_COLUMN

def create_sequences(df, n_steps):
    X, y = [], []
    # Đảm bảo dữ liệu được sắp xếp theo thời gian và nhóm theo thành phố trước khi tạo chuỗi
    df = df.sort_values(by=[META_COLUMN, TIME_COLUMN])
    for city in df[META_COLUMN].unique():
        city_df = df[df[META_COLUMN] == city].copy()
        values = city_df[MODEL_INPUT_COLUMNS].values
        targets = city_df[TARGET_COLUMN].values
        
        if len(values) < n_steps: continue
        
        for i in range(len(values) - n_steps):
            X.append(values[i : (i + n_steps)])
            y.append(targets[i + n_steps])
            
    return np.array(X), np.array(y)

def train_lstm():
    config = load_config()
    logger.info("🧠 Đang huấn luyện LSTM...")
    
    df = pd.read_parquet(config["artifacts"]["training_data_path"])
    X, y = create_sequences(df, n_steps=24)
    
    # Dùng Input layer để model tường minh hơn trong file .keras
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(64, activation='tanh', return_sequences=False), # tanh ổn định hơn relu cho LSTM
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    
    path = config["model"]["lstm"]["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info(f"✅ LSTM saved tại {config['model']['lstm']['path']}")

if __name__ == "__main__":
    train_lstm()
