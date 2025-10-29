import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# 🚍 BUS ETA PREDICTOR APP
# ===============================

st.set_page_config(page_title="Bus ETA Predictor", page_icon="🚌", layout="centered")

st.title("🚌 Bus ETA Predictor")
st.markdown("Predict the **Estimated Time of Arrival (ETA)** of a bus using a trained Random Forest model.")

# ===============================
# 🔹 Load model & scaler
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("final_random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {e}")
    st.stop()

# ===============================
# 🧾 Input form
# ===============================
st.subheader("Enter Bus Details:")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("🛣️ Distance to Destination (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
    speed = st.number_input("⚡ Current Bus Speed (km/h)", min_value=0.0, max_value=120.0, value=40.0, step=1.0)

with col2:
    traffic = st.selectbox("🚦 Traffic Level", ["Low", "Medium", "High"])
    stop_count = st.slider("🚌 Upcoming Bus Stops", 0, 10, 3)

# Convert traffic level to numeric
traffic_map = {"Low": 1, "Medium": 2, "High": 3}
traffic_val = traffic_map[traffic]

# ===============================
# 🔍 Predict ETA
# ===============================
if st.button("Predict ETA"):
    try:
        input_data = np.array([[distance, speed, traffic_val, stop_count]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        st.success(f"🕒 Estimated Time of Arrival: **{prediction[0]:.2f} minutes**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ===============================
# 🧠 Footer
# ===============================
st.markdown("---")
st.caption("Developed by Mohana Karthikeyan 🚀 | Powered by Random Forest + Streamlit")
