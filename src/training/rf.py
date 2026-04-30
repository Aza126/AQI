# src/training/rf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.common.utils import load_config, save_pickle, logger
from src.common.schema import MODEL_INPUT_COLUMNS, TARGET_COLUMN

def train_rf():
    config = load_config()
    logger.info("Loading training data for Random Forest...")
    
    # Nạp dữ liệu từ Parquet
    df = pd.read_parquet(config["artifacts"]["training_data_path"])
    
    # Chia dữ liệu X (features) và y (target)
    X = df[MODEL_INPUT_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # Huấn luyện mô hình
    logger.info("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    
    # Đánh giá nhanh
    preds = rf_model.predict(X)
    logger.info(f"RF Training - MAE: {mean_absolute_error(y, preds):.4f}, R2: {r2_score(y, preds):.4f}")
    
    # Lưu mô hình vào path trong config
    model_path = config["model"]["rf"]["path"]
    save_pickle(rf_model, model_path)
    logger.info(f"Random Forest model saved to {model_path}")

if __name__ == "__main__":
    train_rf()
