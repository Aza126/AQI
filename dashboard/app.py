# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.common.database import get_collection
from src.common.utils import load_config, calculate_aqi_pm25
from src.common.schema import TIME_COLUMN, META_COLUMN, TARGET_COLUMN

# 1. Cấu hình trang
st.set_page_config(page_title="AQI Monitoring & Forecasting", layout="wide")

@st.cache_data(ttl=300)  # Cache dữ liệu 5 phút một lần
def load_data():
    config = load_config()
    raw_col = get_collection(config, "raw_collection")
    pred_col = get_collection(config, "prediction_collection")

    # Lấy dữ liệu thực tế (72 giờ gần nhất)
    raw_data = list(raw_col.find().sort(TIME_COLUMN, -1).limit(500))
    df_raw = pd.DataFrame(raw_data)
    
    # Lấy dữ liệu dự báo
    pred_data = list(pred_col.find().sort(TIME_COLUMN, -1).limit(200))
    df_pred = pd.DataFrame(pred_data)
    
    return df_raw, df_pred

def main():
    st.title("🌬️ Hệ Thống Theo Dõi & Dự Báo Chất Lượng Không Khí (PM2.5)")
    
    try:
        df_raw, df_pred = load_data()
    except Exception as e:
        st.error(f"Không thể kết nối dữ liệu: {e}")
        return

    if df_raw.empty:
        st.warning("Chưa có dữ liệu trong Database.")
        return

    # --- SIDEBAR: Chọn địa điểm ---
    cities = df_raw[META_COLUMN].unique()
    selected_city = st.sidebar.selectbox("📍 Chọn địa điểm", cities)

    # Lọc dữ liệu theo thành phố
    city_raw = df_raw[df_raw[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)
    city_pred = df_pred[df_pred[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)

    # --- Kpi Metrics ---
    latest_aqi = city_raw.iloc[-1]["pm2_5"] # Giả sử lấy pm2.5 thực tế mới nhất
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("PM2.5 Hiện Tại", f"{latest_aqi:.2f} µg/m³")
    with col2:
        if not city_pred.empty:
            last_rf = city_pred[city_pred["model_type"] == "rf"].iloc[-1]["predicted_aqi"]
            st.metric("Dự báo (RF)", f"{last_rf:.1f} AQI")
    with col3:
        if not city_pred.empty:
            last_lstm = city_pred[city_pred["model_type"] == "lstm"].iloc[-1]["predicted_aqi"]
            st.metric("Dự báo (LSTM)", f"{last_lstm:.1f} AQI")

    # --- BIỂU ĐỒ CHÍNH ---
    st.subheader(f"Biểu đồ chỉ số tại {selected_city}")
    
    fig = go.Figure()

    # Quy đổi PM2.5 thực tế sang AQI trước khi vẽ
    city_raw["aqi_real"] = city_raw["pm2_5"].apply(calculate_aqi_pm25)
    # Đường dữ liệu thực tế
    fig.add_trace(go.Scatter(
        x=city_raw[TIME_COLUMN], 
        y=city_raw["aqi_real"],  # Vẽ AQI thực tế
        name='AQI Thực tế',
        line=dict(color='royalblue', width=3)
    ))

    # Đường dự báo RF
    if not city_pred.empty:
        rf_df = city_pred[city_pred["model_type"] == "rf"]
        fig.add_trace(go.Scatter(
            x=rf_df[TIME_COLUMN], 
            y=rf_df["predicted_aqi"], 
            mode='markers',
            name='Dự báo RF',
            marker=dict(color='orange', size=8, symbol='x')
        ))

        # Đường dự báo LSTM
        lstm_df = city_pred[city_pred["model_type"] == "lstm"]
        fig.add_trace(go.Scatter(
            x=lstm_df[TIME_COLUMN], 
            y=lstm_df["predicted_aqi"], 
            mode='markers',
            name='Dự báo LSTM',
            marker=dict(color='red', size=8, symbol='diamond')
        ))

    fig.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Giá trị",
        legend_title="Chú thích",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- BẢNG DỮ LIỆU ---
    if st.checkbox("Hiển thị bảng dữ liệu chi tiết"):
        st.write("Dữ liệu dự báo gần nhất:")
        st.dataframe(city_pred.tail(10))

    st.subheader("🔥 Bản đồ nhiệt AQI theo thời gian (Tất cả địa điểm)")

    # Chuẩn bị dữ liệu cho Heatmap (Lấy top 100 bản ghi mới nhất)
    df_heatmap = df_raw.copy()
    df_heatmap["aqi"] = df_heatmap["pm2_5"].apply(calculate_aqi_pm25)

    # Tạo bảng pivot: Dòng là địa điểm, Cột là Thời gian, Giá trị là AQI
    pivot_df = df_heatmap.pivot_table(
        index=META_COLUMN, 
        columns=df_heatmap[TIME_COLUMN].dt.strftime('%H:00 %d/%m'), 
        values="aqi"
    )

    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(x="Thời gian", y="Địa điểm", color="Chỉ số AQI"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="RdYlGn_r", # Đỏ (Xấu) -> Vàng -> Xanh (Tốt)
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


if __name__ == "__main__":
    main()
