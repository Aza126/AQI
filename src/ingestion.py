# src/ingestion.py
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from src.common import load_config, logger, get_collection, TIME_COLUMN, META_COLUMN, RAW_COLUMNS

def fetch_air_quality(lat: float, lon: float, base_url: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(RAW_COLUMNS),
        "timezone": "UTC", # Quan trọng: Luôn lấy giờ chuẩn UTC
        "past_days": 7, # Lấy dữ liệu của 7 ngày trước đó
        "forecast_days": 1 # Chỉ lấy thêm 1 ngày dự báo (hoặc set = 0 nếu chỉ muốn lấy quá khứ)
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

def transform_raw(data: dict, city_name: str):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    records = []

    for i, t in enumerate(times):
        # Lưu vào MongoDB dưới dạng object datetime (UTC)
        dt_obj = datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("UTC"))
        
        record = {
            TIME_COLUMN: dt_obj,
            META_COLUMN: city_name
        }

        for col in RAW_COLUMNS:
            values = hourly.get(col, [])
            # Chuyển về float để tránh lỗi định dạng khi training
            val = values[i] if i < len(values) else None
            record[col] = float(val) if val is not None else None

        records.append(record)
    return records

def run_ingestion():
    logger.info("Starting ingestion...")
    config = load_config()
    base_url = config["api"]["base_url"]
    collection = get_collection(config, "raw_collection")
    locations = config["locations"]

    all_records = []
    for loc in locations:
        try:
            logger.info(f"Fetching data for {loc['name']}")
            data = fetch_air_quality(loc["lat"], loc["lon"], base_url)
            records = transform_raw(data, loc["name"])
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed to fetch {loc['name']}: {e}")

    if all_records:
        # Xóa dữ liệu cũ (tùy chọn) hoặc chỉ insert mới
        collection.delete_many({}) 
        collection.insert_many(all_records)
        logger.info(f"Inserted {len(all_records)} raw records.")

if __name__ == "__main__":
    run_ingestion()
