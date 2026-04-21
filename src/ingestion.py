import requests
from datetime import datetime
from src.common.utils import load_config, logger
from src.common.database import get_collection
from src.common.schema import TIME_COLUMN, META_COLUMN, RAW_COLUMNS

def fetch_air_quality(lat: float, lon: float, base_url: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(RAW_COLUMNS)
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    return response.json()

def transform_raw(data: dict, city_name: str):
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    records = []

    for i, t in enumerate(times):
        # Convert ISO string to datetime object for MongoDB Time Series
        dt_obj = datetime.fromisoformat(t) if isinstance(t, str) else t
        
        record = {
            TIME_COLUMN: dt_obj,
            META_COLUMN: city_name
        }

        for col in RAW_COLUMNS:
            values = hourly.get(col, [])
            record[col] = values[i] if i < len(values) else None

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
        logger.info(f"Fetching data for {loc['name']}")
        data = fetch_air_quality(loc["lat"], loc["lon"], base_url)
        records = transform_raw(data, loc["name"])
        all_records.extend(records)

    if all_records:
        collection.insert_many(all_records)
        logger.info(f"Inserted {len(all_records)} raw records.")

    logger.info("Ingestion done.")

if __name__ == "__main__":
    run_ingestion()