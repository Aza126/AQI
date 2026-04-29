# src/scripts/download_models.py
import os
import gdown
import logging
from src.common.utils import load_config, get_env

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_from_drive():
    try:
        config = load_config()
        
        # Ánh xạ từ cấu hình và biến môi trường
        files_to_check = {
            config["artifacts"]["scaler_path"]: get_env("DRIVE_ID_SCALER"),
            config["artifacts"]["training_data_path"]: get_env("DRIVE_ID_TRAINING_DATA"),
            config["model"]["rf"]["path"]: get_env("DRIVE_ID_RF"),
            config["model"]["lstm"]["path"]: get_env("DRIVE_ID_LSTM")
        }
        
        for file_path, drive_id in files_to_check.items():
            if not drive_id:
                logger.warning(f"Bỏ qua {file_path}: DRIVE_ID không tồn tại trong .env")
                continue

            if not os.path.exists(file_path):
                logger.info(f"🚀 Đang tải {file_path} từ Google Drive (ID: {drive_id})...")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Sử dụng tham số id trực tiếp giúp gdown xử lý tốt hơn
                gdown.download(id=drive_id, output=file_path, quiet=False)
            else:
                logger.info(f"✅ File {file_path} đã có sẵn cục bộ. Bỏ qua tải xuống.")
                
        logger.info("✨ Hoàn thành kiểm tra và tải các artifacts/models.")
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình tải file: {e}")

if __name__ == "__main__":
    download_from_drive()
