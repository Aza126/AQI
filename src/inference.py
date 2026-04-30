# src/inference.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.common.utils import load_config, load_pickle, logger
from src.common.database import get_collection
from src.common.schema import TIME_COLUMN, META_COLUMN, MODEL_INPUT_COLUMNS
from src.preprocess import run_preprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_all_models(config):
    models = {}
    run_models = config["inference"]["run_models"]

    for m_type in run_models:
        path = config["model"][m_type]["path"]
        if not os.path.exists(path):
            logger.error(f"❌ Không tìm thấy file model tại {path}. Hãy chạy train_{m_type}.py trước!")
            continue

    if "rf" in run_models:
        models["rf"] = load_pickle(config["model"]["rf"]["path"])
        logger.info("🌲 Loaded Random Forest model")

    if "lstm" in run_models:
        models["lstm"] = load_model(config["model"]["lstm"]["path"])
        logger.info("🧠 Loaded LSTM model")

    return models

def run_inference():
    logger.info("🔮 Bắt đầu quy trình dự báo AQI...")
    config = load_config()
    processed_col = get_collection(config, "processed_collection")

    # Tìm mốc dự báo mới nhất của từng model trong DB để tránh trùng
    # Thay vì gọi run_preprocess(), ta lấy trực tiếp dữ liệu đã xử lý từ DB
    # Lấy dữ liệu của 24 giờ gần nhất cho tất cả các thành phố
    data = list(processed_col.find().sort(TIME_COLUMN, -1).limit(500)) 
    
    if not data:
        logger.error("❌ Không có dữ liệu trong db_processed để dự báo.")
        return

    df_processed = pd.DataFrame(data)
    # Vì ta lấy sort -1 (mới nhất lên đầu), nên cần sort lại cho đúng trình tự thời gian
    df_processed = df_processed.sort_values([META_COLUMN, TIME_COLUMN])

    models = load_all_models(config)
    results = []

    for city in df_processed[META_COLUMN].unique():
        city_df = df_processed[df_processed[META_COLUMN] == city].sort_values(TIME_COLUMN)
        
        if len(city_df) < 24:
            logger.warning(f"⚠️ {city} thiếu dữ liệu chuỗi. Bỏ qua.")
            continue

        latest_row = city_df.tail(1)
        # Sửa lỗi lấy timestamp
        dt_now = latest_row[TIME_COLUMN].iloc[0]
        if hasattr(dt_now, 'to_pydatetime'):
            dt_now = dt_now.to_pydatetime()

        # --- Random Forest ---
        if "rf" in models:
            X_rf = latest_row[MODEL_INPUT_COLUMNS].values
            rf_pred = models["rf"].predict(X_rf)[0]
            
            results.append({
                TIME_COLUMN: dt_now,
                META_COLUMN: city,
                "model_type": "rf",
                "predicted_aqi": round(float(rf_pred), 2),
                "created_at": pd.Timestamp.now().to_pydatetime()
            })

        # --- LSTM ---
        if "lstm" in models:
            X_lstm_seq = city_df.tail(24)[MODEL_INPUT_COLUMNS].values # .astype(np.float32)
            X_lstm_seq = np.expand_dims(X_lstm_seq, axis=0)           # Kết quả phải là shape (1, 24, n_features)
            
            lstm_pred = models["lstm"].predict(X_lstm_seq, verbose=0)[0][0]
            
            results.append({
                TIME_COLUMN: dt_now,
                META_COLUMN: city,
                "model_type": "lstm",
                "predicted_aqi": round(float(lstm_pred), 2),
                "created_at": pd.Timestamp.now().to_pydatetime()
            })

    if results:
        collection = get_collection(config, "prediction_collection")
        collection.insert_many(results)
        logger.info(f"✅ Đã lưu {len(results)} kết quả dự báo.")

if __name__ == "__main__":
    run_inference()
