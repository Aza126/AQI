# src/ingestion.py
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from src.common import load_config, logger, get_collection, TIME_COLUMN, META_COLUMN, RAW_COLUMNS
from pymongo import UpdateOne

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

    # Tạo danh sách các tác vụ Upsert
    operations = []
    for record in all_records:
        # Cặp nhận diện duy nhất: Thời gian + Tên thành phố
        filter_query = {
            TIME_COLUMN: record[TIME_COLUMN],
            META_COLUMN: record[META_COLUMN]
        }
        # Cập nhật toàn bộ dữ liệu mới vào bản ghi
        update_query = {"$set": record}
            
        operations.append(UpdateOne(filter_query, update_query, upsert=True))

    # Thực thi đồng loạt (Bulk Write) để tối ưu hiệu năng
    if operations:
        result = collection.bulk_write(operations)
        logger.info(f"Ingestion done: {result.upserted_count} new, {result.modified_count} updated.")
if __name__ == "__main__":
    run_ingestion()
