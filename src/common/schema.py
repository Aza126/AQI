# src/common/schema.py
import logging
logger = logging.getLogger(__name__)

# ========================
# TIME & META (For MongoDB Time Series)
# ========================
TIME_COLUMN = "timestamp"
META_COLUMN = "city"

# ========================
# TARGET
# ========================
TARGET_COLUMN = "aqi"

# ========================
# RAW DATA
# ========================
RAW_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
    "temperature_2m",       # Nhiệt độ (cách mặt đất 2m)
    "relative_humidity_2m", # Độ ẩm tương đối
    "surface_pressure",     # Áp suất bề mặt
    "wind_speed_10m"        # Tốc độ gió (cách mặt đất 10m)
]

# ========================
# FEATURE ENGINEERING
# ========================
LAG_FEATURES = ["pm25_lag_1h", "pm25_lag_2h", "pm25_lag_3h"]

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide",
    "temperature_2m",        # Thêm vào để mô hình học yếu tố nhiệt độ
    "relative_humidity_2m",   # Thêm vào vì độ ẩm ảnh hưởng đến AQI
    "wind_speed_10m",         # Thêm vào vì gió giúp tán bụi
    "hour_sin", "hour_cos"
] +  LAG_FEATURES
# ========================
# MODEL INPUT
# ========================
MODEL_INPUT_COLUMNS = FEATURE_COLUMNS

# ========================
# VALIDATION
# ========================
def validate_columns(df):
    cols = set(df.columns)

    missing = set(MODEL_INPUT_COLUMNS) - cols
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = cols - set(MODEL_INPUT_COLUMNS)
    if extra:
        logger.warning(f"Extra columns ignored: {extra}")

    return True