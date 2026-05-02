import sys
import os

# Thêm thư mục hiện tại vào PYTHONPATH để nhận diện các module trong src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import và chạy trực tiếp hàm main hoặc logic từ file dashboard/app.py
from dashboard.app import main # Giả sử file app.pycó hàm main()

if __name__ == "__main__":
    main()