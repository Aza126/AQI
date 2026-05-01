# src/ingestion.py
import requests
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import time

from src.common.utils import load_config, logger
from src.common.database import get_collection
from src.common.schema import TIME_COLUMN, META_COLUMN, RAW_COLUMNS

def fetch_air_quality(lat: float, lon: float, config):
    base_url = config["api"]["base_url"]
    weather_url = config["api"]["weather_url"]
    common_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC", # Luôn lấy giờ chuẩn UTC
        "past_days": 7, # Lấy dữ liệu của 7 ngày trước đó
        "forecast_days": 1 # Chỉ lấy thêm 1 ngày dự báo (hoặc set = 0 nếu chỉ muốn lấy quá khứ)
    }

    # 1. Gọi API Lấy chất lượng không khí (từ base_url trong config.yaml)
    aqi_params = common_params.copy()
    aqi_params["hourly"] = "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide"
    
    aqi_response = requests.get(base_url, params=aqi_params)
    aqi_response.raise_for_status()
    aqi_data = aqi_response.json()

    # 2. Gọi API Lấy thời tiết (từ weather_url)
    weather_params = common_params.copy()
    weather_params["hourly"] = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    
    weather_response = requests.get(weather_url, params=weather_params)
    weather_response.raise_for_status()
    weather_data = weather_response.json()

    # 3. Gộp chung dữ liệu (Merge dictionary)
    combined_hourly = aqi_data.get("hourly", {})
    combined_hourly.update(weather_data.get("hourly", {})) # Nhét thêm data thời tiết vào

    aqi_data["hourly"] = combined_hourly

    return aqi_data # Trả về data đã gộp chung hoàn chỉnh

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
    collection = get_collection(config, "raw_collection")
    locations = config["locations"]

    all_records = []
    for loc in locations:
        try:
            logger.info(f"Fetching data for {loc['name']}")
            data = fetch_air_quality(loc["lat"], loc["lon"], config)
            records = transform_raw(data, loc["name"])
            all_records.extend(records)
            time.sleep(2) # Thêm delay để tránh bị rate limit
        except Exception as e:
            logger.error(f"Failed to fetch {loc['name']}: {e}")
            time.sleep(5) # Nếu lỗi, đợi lâu hơn trước khi thử tiếp

    if all_records:
        # 1. Tìm mốc thời gian mới nhất hiện có trong DB
        # Giả sử trường thời gian là 'timestamp'
        latest_record = collection.find_one(sort=[("timestamp", -1)])
        
        if latest_record:
            latest_time = latest_record["timestamp"]

            # Kiểm tra nếu latest_time từ DB đã có múi giờ (aware) 
            # thì đảm bảo dữ liệu mới cũng phải có múi giờ để so sánh
            new_records = []
            for r in all_records:
                # Ép kiểu timestamp của record mới sang UTC nếu nó chưa có múi giờ
                r_time = r["timestamp"]
                if r_time.tzinfo is None:
                    r_time = r_time.replace(tzinfo=timezone.utc)
                
                # Ép kiểu latest_time sang UTC nếu nó chưa có múi giờ
                compare_latest = latest_time
                if compare_latest.tzinfo is None:
                    compare_latest = compare_latest.replace(tzinfo=timezone.utc)
                # 2. Lọc: Chỉ lấy các bản ghi có timestamp lớn hơn latest_time
                if r_time > compare_latest:
                    new_records.append(r)
        else:
            # Nếu DB trống, lấy toàn bộ
            new_records = all_records

        # 3. Chèn dữ liệu mới
        if new_records:
            try:
                collection.insert_many(new_records, ordered=False) # ordered=False để bỏ qua lỗi trùng lặp nếu có
                logger.info(f"Đã chèn thêm {len(new_records)} bản ghi mới.")
            except Exception as e:
                logger.error(f"Lỗi khi chèn dữ liệu: {e}")
        else:
            logger.info("Dữ liệu đã cũ hoặc trùng lặp. Không có gì để cập nhật.")

if __name__ == "__main__":
    run_ingestion()
