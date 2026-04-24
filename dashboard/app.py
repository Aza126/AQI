import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.common import load_config, get_collection, TIME_COLUMN, META_COLUMN

st.set_page_config(page_title="AQI Dashboard", layout="wide")

def load_data():
    config = load_config()
    raw_col = get_collection(config, "raw_collection")
    pred_col = get_collection(config, "prediction_collection")
    
    # Lấy dữ liệu và chuyển sang DataFrame
    actual_df = pd.DataFrame(list(raw_col.find().sort(TIME_COLUMN, -1).limit(500)))
    pred_df = pd.DataFrame(list(pred_col.find().sort(TIME_COLUMN, -1).limit(500)))
    
    return actual_df, pred_df, config

st.title("📈 Đối chiếu Quỹ đạo Ô nhiễm")

actual, pred, config = load_data()

if not actual.empty:
    # --- Sidebar lọc thành phố ---
    cities = actual[META_COLUMN].unique()
    selected_city = st.sidebar.selectbox("Chọn thành phố", cities)

    # Lọc dữ liệu theo thành phố đã chọn
    city_actual = actual[actual[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)
    city_pred = pred[pred[META_COLUMN] == selected_city].sort_values(TIME_COLUMN)

    fig = go.Figure()

    # Đường line thực tế (Sử dụng pm2_5 hoặc aqi nếu bạn đã tính ở bước trước)
    y_col = "aqi" if "aqi" in city_actual.columns else "pm2_5"
    
    fig.add_trace(go.Scatter(
        x=city_actual[TIME_COLUMN], 
        y=city_actual[y_col],
        mode='lines', # Bỏ markers để đường mượt hơn
        name=f'Thực tế ({selected_city})',
        line=dict(color='#00d1ff', width=3)
    ))

    # Điểm dự báo RF
    rf_pred = city_pred[city_pred["model"] == "rf"]
    fig.add_trace(go.Scatter(
        x=rf_pred[TIME_COLUMN], 
        y=rf_pred["prediction"],
        mode='markers', 
        name='Dự báo (Random Forest)',
        marker=dict(color='#ff4b4b', size=10, symbol='circle')
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Thời gian",
        yaxis_title="Chỉ số AQI",
        hovermode="x unified",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Đang chờ dữ liệu từ database...")
