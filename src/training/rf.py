# src/training/rf.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.common.utils import load_config, save_pickle, logger
from src.common.schema import MODEL_INPUT_COLUMNS, TARGET_COLUMN

def train_rf():
    config = load_config()
    logger.info("🌲 Đang huấn luyện Random Forest...")
    
    # 1. Load dữ liệu đã chuẩn bị
    df = pd.read_parquet(config["artifacts"]["training_data_path"])
    
    X = df[MODEL_INPUT_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # 2. Split dữ liệu (Train/Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Huấn luyện
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Đánh giá
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logger.info(f"✅ RF Train xong. MAE: {mae:.2f}, R2: {r2_score(y_test, preds):.2f}")
    
    # 5. Lưu model
    path = config["model"]["rf"]["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True) # Tạo thư mục nếu chưa có
    save_pickle(model, path)

if __name__ == "__main__":
    train_rf()
