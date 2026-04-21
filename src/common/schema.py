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
    missing = set(MODEL_INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")