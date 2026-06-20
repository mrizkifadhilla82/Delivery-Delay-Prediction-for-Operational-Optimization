import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings
import requests
from pathlib import Path
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# TENSORFLOW / KERAS IMPORTS
# Harus di-import di sini agar joblib bisa
# mendeserialisasi model yang menggunakan scikeras
# ─────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier  # noqa: F401

# Fungsi ini HARUS ada di sini karena joblib menyimpan
# referensi ke create_model saat model di-pickle dari notebook
def create_model(optimizer="adam", dropout_rate=0.2, input_shape=None):
    m = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,) if input_shape else (None,)),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return m

# Resolve base directory relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Delivery_Logistics.csv"
MODEL_PATH = BASE_DIR / "models" / "delivery_delay_model.pkl"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        color: white;
    }
    .hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; }
    .hero-banner p { font-size: 1rem; margin: 0; opacity: 0.85; }

    /* KPI Cards */
    .kpi-card {
        background: #1a1d29;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #7c3aed; }
    .kpi-label { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; }

    /* Prediction Card */
    .pred-card-delay {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: white;
    }
    .pred-card-ontime {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: white;
    }
    .pred-label { font-size: 1rem; opacity: 0.9; margin-bottom: 8px; }
    .pred-result { font-size: 2.2rem; font-weight: 700; }

    /* Section title */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #7c3aed;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["delayed"] = df["delayed"].map({"yes": 1, "no": 0})
    return df

import sys
sys.modules['__main__'].create_model = create_model

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <span style='font-size:2.5rem;'>🚚</span>
            <h2 style='color:#e2e8f0; margin:8px 0 4px 0; font-size:1.2rem;'>Delivery Predictor</h2>
            <p style='color:#64748b; font-size:0.78rem;'>Powered by TensorFlow</p>
        </div>
        <hr style='border-color:#2d3748; margin: 8px 0 16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigasi",
        ["📊 Dashboard Analytics", "🤖 Prediksi Keterlambatan"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2d3748;'>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color:#475569; font-size:0.75rem; text-align:center;'>
            Muhammad Rizki Fadhilla<br>DSML40 — THT
        </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard Analytics":

    st.markdown("""
        <div class='hero-banner'>
            <h1>📊 Dashboard Analytics</h1>
            <p>Eksplorasi insight dari data operasional pengiriman logistik secara interaktif.</p>
        </div>
    """, unsafe_allow_html=True)

    # ── KPI ROW ──────────────────────────────────────────────────────────────
    total = len(df)
    delayed = df["delayed"].sum()
    ontime = total - delayed
    delay_rate = delayed / total * 100
    avg_dist = df["distance_km"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value'>{total:,}</div>
            <div class='kpi-label'>Total Pengiriman</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#ef4444;'>{delayed:,}</div>
            <div class='kpi-label'>Pengiriman Terlambat</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#22c55e;'>{ontime:,}</div>
            <div class='kpi-label'>Pengiriman Tepat Waktu</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#f59e0b;'>{delay_rate:.1f}%</div>
            <div class='kpi-label'>Tingkat Keterlambatan</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 1: Distribusi Target + Cuaca ─────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Distribusi Status Pengiriman</div>", unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[ontime, delayed],
            names=["On Time", "Delayed"],
            color_discrete_sequence=["#22c55e", "#ef4444"],
            hole=0.55,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(font=dict(color="#94a3b8")),
            margin=dict(t=10, b=10)
        )
        fig_pie.update_traces(textfont_color="white", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Keterlambatan per Kondisi Cuaca</div>", unsafe_allow_html=True)
        weather_df = df.groupby("weather_condition")["delayed"].mean().reset_index()
        weather_df.columns = ["Cuaca", "Delay Rate"]
        weather_df["Delay Rate (%)"] = weather_df["Delay Rate"] * 100
        fig_weather = px.bar(
            weather_df.sort_values("Delay Rate (%)", ascending=False),
            x="Cuaca", y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale="Reds",
            text="Delay Rate (%)"
        )
        fig_weather.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_weather.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", title="Delay Rate (%)"),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_weather, use_container_width=True)

    # ── ROW 2: Delivery Mode + Region ────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>Keterlambatan per Mode Pengiriman</div>", unsafe_allow_html=True)
        mode_df = df.groupby("delivery_mode")["delayed"].mean().reset_index()
        mode_df["Delay Rate (%)"] = mode_df["delayed"] * 100
        fig_mode = px.bar(
            mode_df.sort_values("Delay Rate (%)", ascending=False),
            x="delivery_mode", y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale="Purples",
            text="Delay Rate (%)"
        )
        fig_mode.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_mode.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis=dict(color="#94a3b8", title="Delivery Mode"),
            yaxis=dict(color="#94a3b8", title="Delay Rate (%)"),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_mode, use_container_width=True)

    with col4:
        st.markdown("<div class='section-title'>Keterlambatan per Region</div>", unsafe_allow_html=True)
        region_df = df.groupby("region")["delayed"].mean().reset_index()
        region_df["Delay Rate (%)"] = region_df["delayed"] * 100
        fig_region = px.bar(
            region_df.sort_values("Delay Rate (%)", ascending=False),
            x="region", y="Delay Rate (%)",
            color="Delay Rate (%)",
            color_continuous_scale="Blues",
            text="Delay Rate (%)"
        )
        fig_region.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_region.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            xaxis=dict(color="#94a3b8", title="Region"),
            yaxis=dict(color="#94a3b8", title="Delay Rate (%)"),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_region, use_container_width=True)

    # ── ROW 3: Distribusi Jarak ───────────────────────────────────────────────
    st.markdown("<div class='section-title'>Distribusi Jarak Pengiriman (km)</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        df, x="distance_km", color="delayed",
        color_discrete_map={0: "#22c55e", 1: "#ef4444"},
        labels={"delayed": "Status", "distance_km": "Jarak (km)"},
        nbins=50, barmode="overlay", opacity=0.75
    )
    fig_hist.for_each_trace(lambda t: t.update(name="Delayed" if t.name == "1" else "On Time"))
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8"),
        legend=dict(font=dict(color="#94a3b8")),
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: PREDIKSI KETERLAMBATAN
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
        <div class='hero-banner'>
            <h1>🤖 Prediksi Keterlambatan Pengiriman</h1>
            <p>Masukkan parameter pengiriman untuk mendapatkan prediksi status dari model TensorFlow.</p>
        </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("<div class='section-title'>📋 Parameter Pengiriman</div>", unsafe_allow_html=True)

        with st.form("prediction_form"):
            delivery_partner = st.selectbox(
                "Mitra Pengiriman",
                sorted(df["delivery_partner"].unique())
            )
            package_type = st.selectbox(
                "Jenis Paket",
                sorted(df["package_type"].unique())
            )
            vehicle_type = st.selectbox(
                "Jenis Kendaraan",
                sorted(df["vehicle_type"].unique())
            )
            delivery_mode = st.selectbox(
                "Mode Pengiriman",
                sorted(df["delivery_mode"].unique())
            )
            region = st.selectbox(
                "Region",
                sorted(df["region"].unique())
            )
            weather_condition = st.selectbox(
                "Kondisi Cuaca",
                sorted(df["weather_condition"].unique())
            )

            st.markdown("<br>", unsafe_allow_html=True)

            distance_km = st.slider(
                "Jarak Pengiriman (km)",
                min_value=float(df["distance_km"].min()),
                max_value=float(df["distance_km"].max()),
                value=float(df["distance_km"].median()),
                step=0.1
            )
            package_weight_kg = st.slider(
                "Berat Paket (kg)",
                min_value=float(df["package_weight_kg"].min()),
                max_value=float(df["package_weight_kg"].max()),
                value=float(df["package_weight_kg"].median()),
                step=0.1
            )
            delivery_cost = st.number_input(
                "Biaya Pengiriman (Rp)",
                min_value=float(df["delivery_cost"].min()),
                max_value=float(df["delivery_cost"].max()),
                value=float(df["delivery_cost"].median()),
                step=10.0
            )

            submitted = st.form_submit_button(
                "🔍 Prediksi Sekarang",
                use_container_width=True,
                type="primary"
            )

    with col_result:
        st.markdown("<div class='section-title'>📈 Hasil Prediksi</div>", unsafe_allow_html=True)

        if submitted:
            payload = {
                "delivery_partner": delivery_partner,
                "package_type": package_type,
                "vehicle_type": vehicle_type,
                "delivery_mode": delivery_mode,
                "region": region,
                "weather_condition": weather_condition,
                "distance_km": float(distance_km),
                "package_weight_kg": float(package_weight_kg),
                "delivery_cost": float(delivery_cost)
            }

            with st.spinner("Menganalisis..."):
                serving_mode = "local"
                try:
                    # Attempt connection to FastAPI model serving endpoint
                    response = requests.post("http://localhost:8000/predict", json=payload, timeout=2.0)
                    if response.status_code == 200:
                        res_data = response.json()
                        prob = float(res_data["probability"])
                        serving_mode = "api"
                    else:
                        raise Exception("API error")
                except Exception:
                    # Fallback to local prediction
                    input_df = pd.DataFrame([payload])
                    prob_raw = model.predict_proba(input_df)
                    if hasattr(prob_raw, "shape") and len(prob_raw.shape) > 1 and prob_raw.shape[1] > 1:
                        prob = float(prob_raw[0, 1])
                    elif hasattr(prob_raw, "shape") and len(prob_raw.shape) > 1:
                        prob = float(prob_raw[0, 0])
                    else:
                        prob = float(prob_raw[0])
                
                is_delayed = prob >= 0.5

            # Result badge
            if is_delayed:
                st.markdown(f"""<div class='pred-card-delay'>
                    <div class='pred-label'>Status Pengiriman</div>
                    <div class='pred-result'>⚠️ Terlambat</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='pred-card-ontime'>
                    <div class='pred-label'>Status Pengiriman</div>
                    <div class='pred-result'>✅ Tepat Waktu</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if serving_mode == "api":
                st.info("🤖 **Serving Mode: REST API (FastAPI Active)**\n\nPrediksi dihitung oleh model server eksternal dan dicatat di `logs/prediction_logs.csv`.")
            else:
                st.warning("💻 **Serving Mode: Local Fallback (FastAPI Offline)**\n\nAPI server tidak mendeteksi koneksi, prediksi dihitung secara lokal di client.")
            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 28}},
                title={"text": "Probabilitas Keterlambatan", "font": {"color": "#94a3b8", "size": 13}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": "#ef4444" if is_delayed else "#22c55e", "thickness": 0.25},
                    "bgcolor": "#1a1d29",
                    "bordercolor": "#2d3748",
                    "steps": [
                        {"range": [0, 50], "color": "#14532d"},
                        {"range": [50, 75], "color": "#713f12"},
                        {"range": [75, 100], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=250,
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Summary table
            st.markdown("<div class='section-title'>📝 Ringkasan Input</div>", unsafe_allow_html=True)
            summary = {
                "Parameter": ["Mitra", "Jenis Paket", "Kendaraan", "Mode", "Region", "Cuaca", "Jarak (km)", "Berat (kg)", "Biaya (Rp)"],
                "Nilai": [delivery_partner, package_type, vehicle_type, delivery_mode, region, weather_condition, f"{distance_km:.1f}", f"{package_weight_kg:.1f}", f"{delivery_cost:,.0f}"]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        else:
            st.markdown("""
                <div style='
                    border: 2px dashed #2d3748;
                    border-radius: 12px;
                    padding: 60px 20px;
                    text-align: center;
                    color: #475569;
                '>
                    <div style='font-size: 3rem;'>🤖</div>
                    <div style='font-size: 1rem; margin-top: 12px;'>Isi form di kiri, lalu klik<br><strong style='color:#7c3aed;'>Prediksi Sekarang</strong></div>
                </div>
            """, unsafe_allow_html=True)
