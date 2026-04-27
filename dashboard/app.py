import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from src.common import load_config, get_collection, TIME_COLUMN, META_COLUMN

st.set_page_config(page_title="AQI Dashboard", layout="wide")

# Sử dụng cache để không nạp lại config nhiều lần
@st.cache_resource
def get_db_collections():
    config = load_config()
    return get_collection(config, "raw_collection"), get_collection(config, "prediction_collection"), config

def load_city_data(selected_city):
    raw_col, pred_col, _ = get_db_collections()
    
    # Tối ưu: Lọc ngay tại MongoDB thay vì load hết về rồi mới lọc bằng Pandas
    query = {META_COLUMN: selected_city}
    
    actual_data = list(raw_col.find(query).sort(TIME_COLUMN, -1).limit(100))
    pred_data = list(pred_col.find(query).sort(TIME_COLUMN, -1).limit(200))
    
    return pd.DataFrame(actual_data), pd.DataFrame(pred_data)

st.title("🌍 AQI Real-time Monitoring & Prediction")

raw_col_init, _, config = get_db_collections()
all_cities = [loc['name'] for loc in config['locations']]

# --- Sidebar ---
st.sidebar.header("Cấu hình bộ lọc")
selected_city = st.sidebar.selectbox("📍 Chọn thành phố", all_cities)
num_hours = st.sidebar.slider("Số giờ hiển thị", 12, 168, 48)

# Load dữ liệu
df_actual, df_pred = load_city_data(selected_city)

if not df_actual.empty:
    df_actual = df_actual.sort_values(TIME_COLUMN)
    latest_aqi = df_actual.iloc[-1].get('aqi', df_actual.iloc[-1].get('pm2_5', 0))
    
    # --- Row 1: Metrics (Các chỉ số nhanh) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chỉ số hiện tại", f"{latest_aqi:.1f}")
    with col2:
        status = "Tốt" if latest_aqi <= 50 else "Trung bình" if latest_aqi <= 100 else "Kém"
        st.metric("Trạng thái không khí", status)
    with col3:
        st.metric("Thành phố", selected_city)

    # --- Row 2: Chart ---
    fig = go.Figure()

    # 1. Đường thực tế
    y_col = "aqi" if "aqi" in df_actual.columns else "pm2_5"
    fig.add_trace(go.Scatter(
        x=df_actual[TIME_COLUMN], y=df_actual[y_col],
        mode='lines+markers', name='Thực tế',
        line=dict(color='#00d1ff', width=3),
        marker=dict(size=4)
    ))

    if not df_pred.empty:
        # 2. Dự báo từ Random Forest
        rf_data = df_pred[df_pred["model"] == "rf"].sort_values(TIME_COLUMN)
        fig.add_trace(go.Scatter(
            x=rf_data[TIME_COLUMN], y=rf_data["prediction"],
            mode='markers', name='Dự báo (RF)',
            marker=dict(color='#ff4b4b', size=8, symbol='diamond')
        ))

        # 3. Dự báo từ LSTM
        lstm_data = df_pred[df_pred["model"] == "lstm"].sort_values(TIME_COLUMN)
        fig.add_trace(go.Scatter(
            x=lstm_data[TIME_COLUMN], y=lstm_data["prediction"],
            mode='lines', name='Dự báo (LSTM)',
            line=dict(color='#00ff00', width=2, dash='dot')
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Thời gian (UTC)",
        yaxis_title="Chỉ số (AQI/PM2.5)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # --- Row 3: Data Table ---
    with st.expander("Xem bảng dữ liệu chi tiết"):
        st.dataframe(df_actual.tail(10), use_container_width=True)
else:
    st.warning(f"Chưa có dữ liệu cho {selected_city}. Vui lòng chạy ingestion.py trước.")

# Heatmap
st.subheader("📍 Mạng lưới quan trắc & Mật độ ô nhiễm")

# 1. Lấy dữ liệu thực tế mới nhất của tất cả thành phố để làm Heatmap
heat_data = []
for loc in config['locations']:
    # Lấy dữ liệu gần nhất của thành phố này từ df_actual (nếu có) 
    # hoặc query nhanh từ DB
    city_df = df_actual[df_actual[META_COLUMN] == loc['name']]
    if not city_df.empty:
        val = city_df.iloc[-1].get('pm2_5', 0)
        # Format: [lat, lon, weight]
        heat_data.append([loc['lat'], loc['lon'], float(val)])

m = folium.Map(location=[16.0, 108.0], zoom_start=5)

# 2. Thêm lớp HeatMap
if heat_data:
    HeatMap(
        heat_data,
        name="Mật độ ô nhiễm",
        min_opacity=0.3,     # Độ mờ tối thiểu
        max_val=max([x[2] for x in heat_data]) if heat_data else 100, # Tỉ lệ màu dựa trên AQI cao nhất
        radius=25,           # Bán kính điểm nhiệt (pixel)
        blur=15,             # Độ nhòe để các điểm hòa vào nhau
        scale_radius=False   # ĐỂ FALSE: điểm nhiệt giữ nguyên size pixel khi zoom (dễ nhìn)
                             # ĐỂ TRUE: điểm nhiệt sẽ to ra/nhỏ lại theo tỉ lệ km trên bản đồ
    ).add_to(m)

# 3. Vẫn vẽ Marker để người dùng bấm vào xem tên
for loc in config['locations']:
    folium.CircleMarker(
        [loc['lat'], loc['lon']],
        radius=5,
        popup=loc['name'],
        color='blue',
        fill=True
    ).add_to(m)

st_folium(m, width=1300, height=450)
