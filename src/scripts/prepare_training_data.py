import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.common.utils import load_config, save_pickle, logger
from src.common.database import get_collection
from src.common.schema import MODEL_INPUT_COLUMNS, TIME_COLUMN

def add_time_features(df: pd.DataFrame):
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df["hour"] = df[TIME_COLUMN].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def run_prepare_training():
    logger.info("Preparing training data...")
    config = load_config()
    raw_col = get_collection(config, "raw_collection")

    data = list(raw_col.find())
    if not data:
        logger.error("No data for training")
        return

    df = pd.DataFrame(data)
    df = add_time_features(df)
    # Thay vì chỉ: df = df.dropna(subset=MODEL_INPUT_COLUMNS)
    # Hãy thêm dòng nội suy vào trước đó:
    df[RAW_COLUMNS] = df[RAW_COLUMNS].interpolate(method='linear', limit_direction='both')
    df = df.dropna(subset=MODEL_INPUT_COLUMNS) # Chỉ drop những dòng ở rìa không thể nội suy
    df = df.dropna(subset=MODEL_INPUT_COLUMNS)

    X = df[MODEL_INPUT_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    save_pickle(scaler, config["artifacts"]["scaler_path"])

    df_scaled = pd.DataFrame(X_scaled, columns=MODEL_INPUT_COLUMNS)
    df_scaled.to_parquet(config["artifacts"]["training_data_path"])

    logger.info("Training data prepared and artifacts saved.")

if __name__ == "__main__":
    run_prepare_training()
