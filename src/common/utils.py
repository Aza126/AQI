# src/common/utils.py
import os
import yaml
import joblib
from dotenv import load_dotenv
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        logger.error(f"Missing ENV variable: {key}")
        raise Exception(f"Missing ENV variable: {key}")
    return value

def get_config(config, *keys):
    value = config
    for k in keys:
        if k not in value:
            raise KeyError(f"Missing config key: {'.'.join(keys)}")
        value = value[k]
    return value

def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    logger.info(f"Loading config...")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["mongo", "model", "artifacts", "inference"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing config section: {key}")

    valid_models = config["model"].keys()
    models = get_config(config, "inference", "run_models")

    for m in models:
        if m not in valid_models:
            raise ValueError(f"Invalid model: {m}")

    return config

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)

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