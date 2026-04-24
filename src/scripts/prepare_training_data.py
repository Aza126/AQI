# src/scripts/prepare_training_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.common import load_config, save_pickle, logger, get_collection, MODEL_INPUT_COLUMNS, TIME_COLUMN, RAW_COLUMNS, META_COLUMN

def add_time_features(df: pd.DataFrame):
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def calculate_aqi_pm25(pm25):
    """Công thức đơn giản tính AQI từ PM2.5"""
    if pm25 <= 12: return (50/12) * pm25
    if pm25 <= 35.4: return ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
    if pm25 <= 55.4: return ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
    if pm25 <= 150.4: return ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
    return 201 # Mức nguy hại

def run_prepare_training():
    logger.info("Preparing training data...")
    config = load_config()
    raw_col = get_collection(config, "raw_collection")

    data = list(raw_col.find())
    if not data:
        logger.error("No data found in MongoDB for training")
        return

    df = pd.DataFrame(data)

    # 1. TÍNH TOÁN CỘT AQI (TARGET) TRƯỚC KHI SCALE
    if "pm2_5" in df.columns:
        df["aqi"] = df["pm2_5"].apply(calculate_aqi_pm25)
    else:
        logger.error("Không tìm thấy cột pm2_5 để tính AQI")
        return

    # 2. Thực hiện các bước tiếp theo như cũ
    # Xử lý lỗi: Xóa cột _id của MongoDB trước khi lưu Parquet
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    df = add_time_features(df)
    
    # Nội suy dữ liệu thiếu
    df[RAW_COLUMNS] = df[RAW_COLUMNS].interpolate(method='linear', limit_direction='both')
    
    # Kiểm tra xem có đủ cột không
    df = df.dropna(subset=MODEL_INPUT_COLUMNS)
    
    if df.empty:
        logger.error("Dataframe is empty after dropping NaNs")
        return

    X = df[MODEL_INPUT_COLUMNS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Lưu scaler
    save_pickle(scaler, config["artifacts"]["scaler_path"])

    # 3. LƯU DỮ LIỆU TỔNG HỢP, khi lưu parquet: Nhớ giữ lại cột 'aqi' không bị scale để làm Target
    df_final = pd.DataFrame(X_scaled, columns=MODEL_INPUT_COLUMNS)
    
    # Gán các cột cần thiết vào df_final
    df_final["aqi"] = df["aqi"].values 
    df_final[TIME_COLUMN] = df[TIME_COLUMN].values
    df_final[META_COLUMN] = df[META_COLUMN].values # Thêm thành phố nếu cần truy vết

    # Lưu file duy nhất
    df_final.to_parquet(config["artifacts"]["training_data_path"])
    
    logger.info(f"Prepared {len(df_final)} records with 'aqi' column. Scaler and Data saved.")


if __name__ == "__main__":
    run_prepare_training()
