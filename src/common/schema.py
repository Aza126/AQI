import logging
logger = logging.getLogger(__name__)

# ========================
# TIME
# ========================
TIME_COLUMN = "timestamp"


# ========================
# TARGET
# ========================
TARGET_COLUMN = "aqi"


# ========================
# RAW DATA (từ API)
# ========================
RAW_COLUMNS = [
    "pm2_5",
    "pm10",
    "temperature",
    "humidity",
    "wind_speed"
]


# ========================
# FEATURE ENGINEERING
# ========================
FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "temperature",
    "humidity",
    "wind_speed",
    "hour_sin",
    "hour_cos"
]


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
