import pandas as pd
import numpy as np

from src.common.utils import load_config, load_pickle, logger
from src.common.database import get_collection
from src.common.schema import (
    MODEL_INPUT_COLUMNS,
    validate_columns,
    TIME_COLUMN,
    META_COLUMN
)

def add_time_features(df: pd.DataFrame):
    # Đảm bảo cột timestamp là datetime
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def preprocess_df(df: pd.DataFrame, scaler):
    df = df.copy()
    df = add_time_features(df)
    # Thay vì chỉ: df = df.dropna(subset=MODEL_INPUT_COLUMNS)
    # Hãy thêm dòng nội suy vào trước đó:
    df[RAW_COLUMNS] = df[RAW_COLUMNS].interpolate(method='linear', limit_direction='both')
    df = df.dropna(subset=MODEL_INPUT_COLUMNS) # Chỉ drop những dòng ở rìa không thể nội suy

    X = df[MODEL_INPUT_COLUMNS]
    validate_columns(X)
    X_scaled = scaler.transform(X)
    
    return X_scaled, df

def run_preprocess():
    logger.info("Starting preprocessing...")
    config = load_config()

    raw_col = get_collection(config, "raw_collection")
    processed_col = get_collection(config, "processed_collection")
    scaler = load_pickle(config["artifacts"]["scaler_path"])

    # Truy vấn dữ liệu thô (Có thể thêm query thời gian ở đây để không query lại toàn bộ DB)
    data = list(raw_col.find())

    if not data:
        logger.warning("No raw data found")
        return None, None

    df = pd.DataFrame(data)
    X_scaled, df_processed = preprocess_df(df, scaler)

    # Đẩy dữ liệu processed lên Atlas
    processed_records = []
    for i in range(len(df_processed)):
        row = df_processed.iloc[i]
        record = {
            TIME_COLUMN: row[TIME_COLUMN].to_pydatetime(),
            META_COLUMN: row[META_COLUMN]
        }
        # Thêm các cột feature đã được scale
        for j, col in enumerate(MODEL_INPUT_COLUMNS):
            record[col] = float(X_scaled[i][j])
            
        processed_records.append(record)

    if processed_records:
        processed_col.insert_many(processed_records)
        logger.info(f"Inserted {len(processed_records)} processed records.")

    return X_scaled, df_processed

if __name__ == "__main__":
    run_preprocess()
