from pymongo import MongoClient
from .config_utils import get_env, get_config # Import từ file utils

# ========================
# MONGODB CONNECTION
# ========================
_client = None

def get_mongo_client():
    global _client
    if _client is None:
        uri = get_env("MONGO_URI")
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