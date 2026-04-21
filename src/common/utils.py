import os
import yaml
import joblib
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import Dict, Any
import logging
# ========================
# LOGGING
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load .env
load_dotenv()


# ========================
# ENV
# ========================
def get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        logger.error(f"Missing ENV variable: {key}")
        raise Exception(f"Missing ENV variable: {key}")
    return value


# ========================
# CONFIG
# ========================
def get_config(config, *keys):
    value = config
    for k in keys:
        if k not in value:
            raise KeyError(f"Missing config key: {'.'.join(keys)}")
        value = value[k]
    return value


def load_config(path: str = "configs/config.yaml") -> Dict[str, Any]:
    logger.info(f"Loading config...")
    with open(path, "r") as f:
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

# ========================
# MONGODB
# ========================
_client = None

def get_mongo_client():
    global _client
    if _client is None:
        uri = get_env("MONGO_URI")
        logger.info("Connecting to MongoDB...")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def get_database(config: dict):
    client = get_mongo_client()
    db_name = get_config(config, "mongo", "db_name")
    return client[db_name]


def get_collection(config: dict, key: str):
    db = get_database(config)
    name = get_config(config, "mongo", key)
    return db[name]


# ========================
# FILE UTILS
# ========================
def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)

