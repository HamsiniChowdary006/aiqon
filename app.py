"""Streamlit demo: enter a shipment -> congestion probability, confidence, reasons, actions.

Run (from the project root, after `python src/train.py`):
    python -m streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Port Congestion Risk", page_icon="🚢", layout="wide")

if not (ROOT / "models" / "rf_clf_calibrated.joblib").exists():
    st.error("Trained models not found. Run `python src/train.py` first, then reload this page.")
    st.stop()

from data_generation import PORTS          # noqa: E402
from explain import explain                # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLORS = {"HIGH": "#C44E52", "MEDIUM": "#DD8452", "LOW": "#55A868"}

PRESETS = {
    "Custom": None,
    "Calm day": dict(port="Mundra", month=3, dow=1, hour=11, queue=2, incoming=4, wind=8, rain=0.0,
                     size=5000, delay=6.0),
    "Busy + stormy (monsoon)": dict(port="Visakhapatnam", month=8, dow=5, hour=23, queue=6, incoming=7,
                                    wind=38, rain=7.0, size=9000, delay=14.0),
}

# ------------------------------------------------------------------ sidebar inputs
st.sidebar.header("Shipment details")
preset_name = st.sidebar.selectbox("Scenario preset", list(PRESETS))
P = PRESETS[preset_name] or dict(port="Mundra", month=6, dow=2, hour=14, queue=4, incoming=6, wind=15,
                                 rain=1.0, size=6000, delay=8.0)
k = preset_name          # changing the preset changes widget keys, which resets the widgets to the preset

port = st.sidebar.selectbox("Port", list(PORTS), index=list(PORTS).index(P["port"]), key=f"port_{k}")
berths = PORTS[port][0]
st.sidebar.caption(f"{port} has {berths} berths")
month = st.sidebar.selectbox("Month", range(1, 13), index=P["month"] - 1,
                             format_func=lambda m: MONTHS[m - 1], key=f"month_{k}")
dow = st.sidebar.selectbox("Day of week", range(7), index=P["dow"], format_func=lambda d: DAYS[d], key=f"dow_{k}")
hour = st.sidebar.slider("Arrival hour (0-23)", 0, 23, P["hour"], key=f"hour_{k}")
queue = st.sidebar.slider("Vessels already in queue", 0, 25, P["queue"], key=f"queue_{k}")
incoming = st.sidebar.slider("Vessels arriving in next 24h", 0, 25, P["incoming"], key=f"inc_{k}")
wind = st.sidebar.slider("Wind speed (km/h)", 0, 70, P["wind"], key=f"wind_{k}")
rain = st.sidebar.slider("Rain (mm/h)", 0.0, 15.0, float(P["rain"]), 0.5, key=f"rain_{k}")
size = st.sidebar.slider("Vessel size (TEU)", 500, 15000, P["size"], 500, key=f"size_{k}")
unknown = st.sidebar.checkbox("Port's recent average delay is unknown", key=f"unk_{k}")
delay_hist = st.sidebar.slider("Port's recent average delay (h)", 0.0, 30.0, float(P["delay"]), 0.5,
                               disabled=unknown, key=f"delay_{k}")

# ------------------------------------------------------------------ prediction
out = explain(port=port, hour=hour, day_of_week=dow, month=month, berth_capacity=berths,
              queue_length=queue, incoming_24h=incoming, wind_speed_kmh=wind, rain_mm_hr=rain,
              vessel_size_teu=size, recent_avg_delay_hrs=np.nan if unknown else delay_hist)
band, p = out["band"], out["probability"]

st.title("🚢 Port Congestion Risk Predictor")
st.caption("Congestion = waiting time before berthing > 12 hours. Trained on a synthetic dataset, "
           "so this demonstrates the method and is not validated on real port data.")

c1, c2, c3 = st.columns([1.2, 1, 1])
c1.markdown(
    f"<div style='background:{COLORS[band]};padding:18px;border-radius:10px;color:white;'>"
    f"<div style='font-size:15px;'>{band} RISK</div>"
    f"<div style='font-size:44px;font-weight:700;'>{p:.0%}</div></div>", unsafe_allow_html=True)
c2.metric("Confidence", out["confidence"])
c3.metric("Queue pressure", f"{queue / berths:.2f}", help="Vessels in queue / berths")

st.subheader("Why")
st.info(out["sentence"])

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Factors moving the risk")
    s = out["shap"]
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:8][::-1]
    labels = {"util": "queue / berths", "queue_length": "queue length", "incoming_ratio": "incoming / berths",
              "incoming_24h": "incoming (24h)", "wind_speed_kmh": "wind speed", "rain_mm_hr": "rain",
              "is_monsoon": "monsoon season", "recent_avg_delay_hrs": "recent port delay",
              "vessel_size_teu": "vessel size", "berth_capacity": "berth capacity", "hour": "hour of day",
              "day_of_week": "day of week", "is_night": "night arrival", "is_weekend": "weekend"}
    names = [labels.get(i, i.replace("port_", "port: ")) for i in s.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(names, s.values, color=["#C44E52" if v > 0 else "#55A868" for v in s.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (red raises risk, green lowers it)")
    plt.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Recommended actions")
    for i, a in enumerate(out["actions"], 1):
        st.markdown(f"**{i}.** {a}")
    st.caption("Bands: above 70% = high, 40-70% = medium, below 40% = low. "
               "Confidence reflects how far the probability is from 50%.")
