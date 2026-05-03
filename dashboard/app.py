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

@st.cache_data(ttl=3600)  # Cache dữ liệu 1 tiếng một lần
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

    # CHUYỂN ĐỔI MÚI GIỜ SANG GIỜ VIỆT NAM (UTC+7)
    if not df_raw.empty:
        df_raw[TIME_COLUMN] = pd.to_datetime(df_raw[TIME_COLUMN])
        # Nếu chưa có múi giờ (naive) thì gán UTC, sau đó đổi sang múi giờ VN
        if df_raw[TIME_COLUMN].dt.tz is None:
            df_raw[TIME_COLUMN] = df_raw[TIME_COLUMN].dt.tz_localize('UTC')
        df_raw[TIME_COLUMN] = df_raw[TIME_COLUMN].dt.tz_convert('Asia/Ho_Chi_Minh')

    if not df_pred.empty:
        df_pred[TIME_COLUMN] = pd.to_datetime(df_pred[TIME_COLUMN])
        if df_pred[TIME_COLUMN].dt.tz is None:
            df_pred[TIME_COLUMN] = df_pred[TIME_COLUMN].dt.tz_localize('UTC')
        df_pred[TIME_COLUMN] = df_pred[TIME_COLUMN].dt.tz_convert('Asia/Ho_Chi_Minh')
    
    return df_raw, df_pred

def main():
    st.title("Hệ Thống Theo Dõi & Dự Báo AQI")
    
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
    selected_city = st.sidebar.selectbox("Chọn địa điểm", cities)

    # Lọc dữ liệu theo thành phố
    city_raw = df_raw[df_raw[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)
    city_pred = df_pred[df_pred[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)

    # --- Kpi Metrics ---
    latest_aqi = calculate_aqi_pm25(city_raw.iloc[-1]["pm2_5"]) # Giả sử lấy pm2.5 thực tế mới nhất
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AQI Thực tế", f"{latest_aqi:.1f}")
    with col2:
        if not city_pred.empty:
            last_rf = city_pred[city_pred["model_type"] == "rf"].iloc[-1]["predicted_aqi"]
            st.metric("Dự báo - RF", f"{last_rf:.1f}")
    with col3:
        if not city_pred.empty:
            last_lstm = city_pred[city_pred["model_type"] == "lstm"].iloc[-1]["predicted_aqi"]
            st.metric("Dự báo - LSTM", f"{last_lstm:.1f}")

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

    st.subheader("Bản đồ nhiệt AQI")

    # Chuẩn bị dữ liệu cho Heatmap (Lấy 100 bản ghi mới nhất)
    df_heatmap = df_raw.copy()
    df_heatmap["aqi"] = df_heatmap["pm2_5"].apply(calculate_aqi_pm25)

    # Tạo bảng pivot: Dòng là địa điểm, Cột là Thời gian, Giá trị là AQI
    pivot_df = df_heatmap.pivot_table(
        index=META_COLUMN, 
        columns=TIME_COLUMN,
        values="aqi"
    )

    # NỘI SUY TRÊN BẢNG PIVOT (Lấp đầy ô trống do API mất dòng)
    # axis=1 nghĩa là nội suy theo chiều ngang (theo thời gian)
    pivot_df = pivot_df.interpolate(method='linear', axis=1, limit_direction='both')

    # FORMAT LẠI TEXT CỘT SAU KHI ĐÃ SẮP XẾP ĐÚNG
    pivot_df.columns = pivot_df.columns.strftime('%H:00 %d/%m')

    # Phân bổ tỷ lệ % (từ 0 đến 500)
    aqi_colorscale = [
        [0.0, "#00E400"],     # Tốt (0 - 50)
        [0.1, "#FFFF00"],     # Trung bình (51 - 100)
        [0.2, "#FF7E00"],     # Kém (101 - 150)
        [0.3, "#FF0000"],     # Xấu (151 - 200)
        [0.4, "#8F3F97"],     # Rất xấu (201 - 300)
        [0.6, "#7E0023"],     # Nguy hại (301 - 500)
        [1.0, "#7E0023"]      # Chặn trên
    ]

    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(x="Thời gian", y="Địa điểm", color="Chỉ số AQI"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale=aqi_colorscale,
        range_color=[0, 500], # Ép khoảng dữ liệu từ 0 đến 500 để màu ánh xạ chính xác
        aspect="auto"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)


if __name__ == "__main__":
    main()
