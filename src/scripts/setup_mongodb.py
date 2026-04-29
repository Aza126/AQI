# src/scripts/setup_mongodb.py
import sys
import os

# Chỉ đường cho Python nhận diện thư mục src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

from src.common.database import get_mongo_client
from src.common.utils import load_config, logger

def setup_timeseries_collections():
    logger.info("Connecting to MongoDB Atlas...")
    config = load_config()
    client = get_mongo_client()
    db = client[config["mongo"]["db_name"]]
    
    # Danh sách các collections cần tạo dưới dạng Time Series
    collections_to_create = [
        config["mongo"]["raw_collection"],
        config["mongo"]["processed_collection"],
        config["mongo"]["prediction_collection"]
    ]
    
    existing_collections = db.list_collection_names()
    
    for coll_name in collections_to_create:
        if coll_name in existing_collections:
            logger.info(f"Collection '{coll_name}' already exists. Skipping...")
            continue
            
        logger.info(f"Creating Time Series collection: {coll_name}...")
        try:
            db.create_collection(
                coll_name,
                timeseries={
                    "timeField": "timestamp",
                    "metaField": "city",
                    "granularity": "hours"
                }
            )
            logger.info(f"Successfully created '{coll_name}'.")
        except Exception as e:
            logger.error(f"Failed to create '{coll_name}': {e}")

    logger.info("MongoDB setup completed.")

if __name__ == "__main__":
    setup_timeseries_collections()