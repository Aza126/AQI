import gdown
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thay thế bằng ID thực tế của bạn trên Google Drive
DRIVE_FILES = {
    "artifacts/scaler.pkl": "FILE_ID_CUA_SCALER",
    "artifacts/training_data_scaled.parquet": "FILE_ID_CUA_PARQUET",
    "models/rf/rf_v1.pkl": "FILE_ID_CUA_RF",
    "models/lstm/lstm_v1.h5": "FILE_ID_CUA_LSTM"
}

def download_drive_artifacts():
    for file_path, drive_id in DRIVE_FILES.items():
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        url = f"https://drive.google.com/uc?id={drive_id}"
        logger.info(f"Downloading {file_path} from Drive...")
        
        # Tải file xuống
        gdown.download(url, file_path, quiet=False)
        
    logger.info("All artifacts and models downloaded successfully.")

if __name__ == "__main__":
    download_drive_artifacts()