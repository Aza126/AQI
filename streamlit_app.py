import sys
import os

# Sử dụng insert(0, ...) thay vì append để đưa thư mục gốc lên ưu tiên hàng đầu
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Import hàm main
try:
    from src.dashboard.app import main
except ImportError as e:
    # Nếu vẫn lỗi, đoạn này giúp nhìn thấy lỗi cụ thể trên giao diện Streamlit
    import streamlit as st
    st.error(f"Không tìm thấy module: {e}")
    st.info(f"Đường dẫn gốc: {root_path}")
    st.write("Danh sách Python path hiện tại:", sys.path)
    raise e

if __name__ == "__main__":
    main()
