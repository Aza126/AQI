# src/preprocess.py
import pandas as pd
import numpy as np
from datetime import timedelta
from src.common.utils import load_config, load_pickle, logger
from src.common.database import get_collection
from src.common.schema import (
    MODEL_INPUT_COLUMNS,
    validate_columns,
    TIME_COLUMN,
    META_COLUMN,
    RAW_COLUMNS,
    LAG_FEATURES
)

def add_time_features(df: pd.DataFrame):
    """Tạo đặc trưng tuần hoàn thời gian."""
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    hour = df[TIME_COLUMN].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def add_lag_features(df: pd.DataFrame):
    """Tạo 3 cột trễ (1h, 2h, 3h) theo từng thành phố."""
    df = df.sort_values(by=[META_COLUMN, TIME_COLUMN])
    for i in range(1, 4):
        col_name = f"pm25_lag_{i}h"
        # Shift dữ liệu trong nhóm thành phố để tránh tràn dữ liệu giữa các tỉnh
        df[col_name] = df.groupby(META_COLUMN)["pm2_5"].shift(i)
    return df

def preprocess_df(df: pd.DataFrame, scaler):
    """Quy trình tiền xử lý chuẩn hóa."""
    df = df.copy()
    
    # 1. Nội suy dữ liệu thiếu (Xử lý các ô trống cảm biến)
    # Thay vì: df[RAW_COLUMNS] = df[RAW_COLUMNS].interpolate(...)
    # Áp dụng nội suy theo từng thành phố để tránh ảnh hưởng chéo giữa các tỉnh:
    df[RAW_COLUMNS] = df.groupby(META_COLUMN)[RAW_COLUMNS].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    # 2. Tạo đặc trưng mới (Time + Lag)
    df = add_time_features(df)
    df = add_lag_features(df)
    
    # 3. Loại bỏ NaN (do lag tạo ra) và lọc cột
    df = df.dropna(subset=MODEL_INPUT_COLUMNS)
    
    X = df[MODEL_INPUT_COLUMNS]
    # Kiểm tra xem có khớp với MODEL_INPUT_COLUMNS trong schema không
    validate_columns(X)
    
    # 4. Chuẩn hóa bằng Scaler đã load
    X_scaled = scaler.transform(X)
    
    return X_scaled, df

def run_preprocess():
    """Hàm chạy để đẩy dữ liệu đã xử lý lên MongoDB (Processed Collection)."""
    logger.info("⚡Đang thực hiện tiền xử lý dữ liệu...")
    config = load_config()
    raw_col = get_collection(config, "raw_collection")
    processed_col = get_collection(config, "processed_collection")
    scaler = load_pickle(config["artifacts"]["scaler_path"])

    # Tìm mốc thời gian cuối cùng đã xử lý
    latest_record = processed_col.find_one(sort=[(TIME_COLUMN, -1)])
    query = {}

    if latest_record:
        # FIX LOGIC: Thay vì lấy lớn hơn, ta lấy LÙI LẠI 3 TIẾNG để làm "ngữ cảnh" (context) cho hàm shift()
        overlap_time = latest_record[TIME_COLUMN] - timedelta(hours=3)
        query = {TIME_COLUMN: {"$gt": overlap_time}}
    
    data = list(raw_col.find(query, {"_id": 0}))
    if not data:
        logger.info("ℹ️ Không có dữ liệu thô mới để xử lý.")
        return None, None

    df = pd.DataFrame(data)
    X_scaled, df_processed = preprocess_df(df, scaler)

    # Sau khi shift xong, ta chỉ lọc giữ lại những dòng thực sự mới để lưu vào DB (tránh lưu trùng)
    if latest_record:
        # Lọc df_processed để chỉ lấy các dòng có thời gian lớn hơn mốc đã xử lý cuối cùng
        # (X_scaled là ma trận numpy nên ta phải dùng mask của pandas để lọc tương ứng)
        mask = df_processed[TIME_COLUMN] > latest_record[TIME_COLUMN]
        df_processed = df_processed[mask]
        X_scaled = X_scaled[mask.values]

    if df_processed.empty:
        logger.info("Không có bản ghi mới nào được tạo ra sau khi làm sạch.")
        return None, None

    # Chuyển đổi để lưu lên Atlas
    processed_records = []
    for i in range(len(df_processed)):
        row = df_processed.iloc[i]
        record = {
            TIME_COLUMN: row[TIME_COLUMN].to_pydatetime(),
            META_COLUMN: row[META_COLUMN]
        }
        # Lưu các giá trị đã scale
        for j, col in enumerate(MODEL_INPUT_COLUMNS):
            record[col] = float(X_scaled[i][j])
        processed_records.append(record)

    if processed_records:
        processed_col.insert_many(processed_records)
        logger.info(f"✅ Đã xử lý và lưu {len(processed_records)} bản ghi.")

    return X_scaled, df_processed

if __name__ == "__main__":
    run_preprocess()
