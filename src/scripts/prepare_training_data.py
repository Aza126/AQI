# src/scripts/prepare_training_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.common.utils import load_config, save_pickle, logger
from src.common.database import get_collection
from src.common.schema import (
    MODEL_INPUT_COLUMNS, TIME_COLUMN, RAW_COLUMNS, 
    META_COLUMN, TARGET_COLUMN, LAG_FEATURES
)

def calculate_aqi_pm25(pm25):
    """
    Tính AQI cho PM2.5 theo chuẩn EPA 2024.
    Sử dụng phương pháp nội suy tuyến tính.
    """
    # Danh sách điểm dừng: (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500)
    ]
    
    # Xử lý các trường hợp biên
    if pm25 < 0: return 0
    if pm25 > 325.4: return 501
    
    for low_c, high_c, low_i, high_i in breakpoints:
        if low_c <= pm25 <= high_c:
            aqi = ((high_i - low_i) / (high_c - low_c)) * (pm25 - low_c) + low_i
            return int(round(aqi)) # Làm tròn về số nguyên theo quy định EPA
            
    return 0

def add_advanced_features(df: pd.DataFrame):
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    # 1. Cyclical Time Features (Vòng tròn đơn vị)
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # 2. Lag Features (cho Random Forest)
    # Lưu ý: Cần sort theo thời gian trước khi shift
    df = df.sort_values(by=[META_COLUMN, TIME_COLUMN])
    
    # Tạo lag 1h, 2h, 3h theo từng thành phố
    for i in range(1, 4):
        col_name = f"pm25_lag_{i}h"
        df[col_name] = df.groupby(META_COLUMN)["pm2_5"].shift(i)
    
    return df

def run_prepare_training():
    logger.info("🚀 Bắt đầu quy trình xử lý dữ liệu huấn luyện...")
    config = load_config()
    raw_col = get_collection(config, "raw_collection")
    
    df = pd.DataFrame(list(raw_col.find({}, {"_id": 0}))) # Loại bỏ _id ngay khi query
    if df.empty: return logger.error("❌ MongoDB trống!")

    # --- TIỀN XỬ LÝ ---
    # Nội suy (Interpolation) như báo cáo đã nêu
    df[RAW_COLUMNS] = df.groupby(META_COLUMN)[RAW_COLUMNS].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # Tính Target AQI
    df[TARGET_COLUMN] = df["pm2_5"].apply(calculate_aqi_pm25)
    
    # Feature Engineering: Thêm đặc trưng (Sin/Cos + 3 Lags)
    df = add_advanced_features(df)
    
    # --- CHUẨN HÓA (STANDARDIZATION) ---
    # Loại bỏ các dòng NaN (do shift tạo ra ở 3 dòng đầu mỗi thành phố)
    df = df.dropna(subset=MODEL_INPUT_COLUMNS + [TARGET_COLUMN])

    # Chuẩn hóa
    scaler = StandardScaler()
    # Chỉ scale các đặc trưng đầu vào, KHÔNG scale cột 'aqi' (Target) và 'timestamp'
    X_scaled = scaler.fit_transform(df[MODEL_INPUT_COLUMNS])
    
    # Lưu scaler vào artifacts/
    save_pickle(scaler, config["artifacts"]["scaler_path"])
    logger.info(f"✅ Đã lưu bộ chuẩn hóa tại {config['artifacts']['scaler_path']}. Bộ này sẽ được dùng để đồng nhất dữ liệu khi dự báo (Inference).")

    # --- LƯU TRỮ ---
    df_final = pd.DataFrame(X_scaled, columns=MODEL_INPUT_COLUMNS)
    """
    df_final[TARGET_COLUMN] = df[TARGET_COLUMN].values
    df_final[TIME_COLUMN] = df[TIME_COLUMN].values
    df_final[META_COLUMN] = df[META_COLUMN].values
    """
    # Để tránh lỗi khi concat, reset index trước khi gán thêm cột
    # viết gọn lại để tránh ghi đè nhiều lần
    df_final = pd.concat([
    df_final, 
    df[[TARGET_COLUMN, TIME_COLUMN, META_COLUMN]].reset_index(drop=True)
], axis=1)

    df_final.to_parquet(config["artifacts"]["training_data_path"], index=False)
    logger.info(f"Done! Saved {len(df_final)} records to {config['artifacts']['training_data_path']}")

if __name__ == "__main__":
    run_prepare_training()
