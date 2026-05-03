import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

from src.common.utils import load_config, logger
from src.common.database import get_collection
from src.common.schema import TIME_COLUMN, META_COLUMN, RAW_COLUMNS

# Định nghĩa hàm gọi API có cơ chế Retry
# Thử lại 3 lần, bắt đầu đợi 2s, sau đó 4s, 8s nếu lỗi mạng/timeout
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout)),
    before_sleep=lambda retry_state: logger.warning(f"Đang thử lại lần {retry_state.attempt_number}...")
)
def safe_requests_get(url, params):
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

def fetch_air_quality(lat: float, lon: float, config):
    base_url = config["api"]["base_url"]
    weather_url = config["api"]["weather_url"]
    
    # TÍNH TOÁN THỜI GIAN: Lấy từ 24h trước (giờ này hôm qua) đến hiện tại
    # Lùi lại thêm 1 giờ đệm để đảm bảo không sót dữ liệu do API cập nhật chậm
    now_utc = datetime.now(timezone.utc)
    start_time = (now_utc - timedelta(days=1, hours=1)).strftime('%Y-%m-%d')
    end_time = now_utc.strftime('%Y-%m-%d')

    common_params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "UTC",
        "start_date": start_time,
        "end_date": end_time
    }

    # 1. Gọi API AQI - sử dụng hàm safe_requests_get
    aqi_params = common_params.copy()
    aqi_params["hourly"] = "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide"
    aqi_data = safe_requests_get(base_url, aqi_params)

    # 2. Gọi API thời tiết
    weather_params = common_params.copy()
    weather_params["hourly"] = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    weather_data = safe_requests_get(weather_url, weather_params)

    # 3. Gộp dữ liệu
    combined_hourly = aqi_data.get("hourly", {})
    combined_hourly.update(weather_data.get("hourly", {}))
    aqi_data["hourly"] = combined_hourly

    return aqi_data

def transform_raw(data: dict, city_name: str):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    records = []

    # Lấy giờ hiện tại (làm tròn đầu giờ) để lọc bỏ các bản ghi dự báo quá xa trong tương lai
    now_limit = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    for i, t in enumerate(times):
        dt_obj = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        
        # Chỉ lấy dữ liệu từ quá khứ đến hiện tại (không lấy dự báo tương lai của API vào db_raw)
        if dt_obj > now_limit:
            continue

        record = {
            TIME_COLUMN: dt_obj,
            META_COLUMN: city_name
        }

        for col in RAW_COLUMNS:
            values = hourly.get(col, [])
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
        except Exception as e:
            logger.error(f"Failed to fetch {loc['name']}: {e}")
            time.sleep(5)
        time.sleep(2)

    if all_records:
        # Lọc bản ghi mới dựa trên DB hiện tại
        latest_record = collection.find_one(sort=[(TIME_COLUMN, -1)])
        
        if latest_record:
            latest_time = latest_record[TIME_COLUMN]
            if latest_time.tzinfo is None:
                latest_time = latest_time.replace(tzinfo=timezone.utc)
            
            new_records = [r for r in all_records if r[TIME_COLUMN] > latest_time]
        else:
            new_records = all_records

        if new_records:
            try:
                # Sắp xếp theo thời gian để đảm bảo thứ tự chèn
                new_records.sort(key=lambda x: x[TIME_COLUMN])
                collection.insert_many(new_records, ordered=False)
                logger.info(f"Inserted {len(new_records)} new records.")
            except Exception as e:
                logger.error(f"Error inserting records: {e}")
        else:
            logger.info("No new data to update.")

if __name__ == "__main__":
    run_ingestion()
