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
    "carbon_monoxide"
]

# ========================
# FEATURE ENGINEERING
# ========================
FEATURE_COLUMNS = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "ozone",
    "carbon_monoxide",
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