import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import time
from logic import run_middleware, get_keys, sign_data

# Page Configuration
st.set_page_config(layout="wide", page_title="AirDefend-X Security Gateway")

# Initialize Session State for persistence
if 'flights' not in st.session_state:
    # Load baseline data and drop empty rows
    st.session_state.flights = pd.read_csv('data/opensky.csv').dropna()
    st.session_state.logs = ["🛡️ System Online: Monitoring Global Airspace"]

# --- SIDEBAR: ATTACK INJECTION CONSOLE ---
st.sidebar.title("🎮 Attack Injection Console")
st.sidebar.info("Inject synthetic threats to test the RSA and AI Middleware layers.")

# 🚀 PHYSICS ATTACK (GHOST)
if st.sidebar.button("🚀 Inject Physics Attack (Ghost)"):
    # Use a real flight as a template
    atk = st.session_state.flights.iloc[0].copy()
    
    # Generate unique ID and impossible physics
    timestamp = int(time.time())
    atk['icao24'] = f"GHOST_{timestamp}"
    atk['velocity'] = 2500.0  # Impossible speed for commercial craft
    
    # Randomize position slightly so multiple ghosts don't overlap
    atk['lat'] += np.random.uniform(-1.5, 1.5)
    atk['lon'] += np.random.uniform(-1.5, 1.5)
    
    # Give it a VALID signature (passes RSA, must be caught by AI)
    priv, _ = get_keys()
    atk['signature'] = sign_data(priv, atk['icao24'], atk['velocity'])
    
    # Inject into session
    st.session_state.flights = pd.concat([pd.DataFrame([atk]), st.session_state.flights]).reset_index(drop=True)
    st.session_state.logs.append(f"⚠️ ALERT: High-Velocity (Ghost) detected: {atk['icao24']}")
    st.rerun()

# 🆔 IDENTITY ATTACK (SPOOF)
if st.sidebar.button("🆔 Inject Identity Attack (Spoof)"):
    # Use a real flight as a template
    atk = st.session_state.flights.iloc[1].copy()
    
    # Generate unique ID
    timestamp = int(time.time())
    atk['icao24'] = f"SPOOF_{timestamp}"
    
    # Randomize position
    atk['lat'] += np.random.uniform(-1.5, 1.5)
    atk['lon'] += np.random.uniform(-1.5, 1.5)
    
    # Give it a malformed signature (must be caught by RSA)
    atk['signature'] = "INVALID_RSA_HEADER_EXPLOIT"
    
    # Inject into session
    st.session_state.flights = pd.concat([pd.DataFrame([atk]), st.session_state.flights]).reset_index(drop=True)
    st.session_state.logs.append(f"⚠️ ALERT: RSA Signature Mismatch: {atk['icao24']}")
    st.rerun()

# 🔄 RESET AIRSPACE
if st.sidebar.button("🔄 Reset Airspace"):
    st.session_state.flights = pd.read_csv('data/opensky.csv').dropna()
    st.session_state.logs = ["✅ System Reset: Airspace Secure"]
    st.rerun()

# --- MIDDLEWARE EXECUTION ---
# Pass the current state through the security logic
df = run_middleware(st.session_state.flights)

# Icon Configuration
ICON_URL = "https://img.icons8.com/m_outlined/512/FFFFFF/airplane-mode-on.png"
icon_data = {"url": ICON_URL, "width": 242, "height": 242, "anchorY": 242, "mask": True}
df['icon_data'] = [icon_data for _ in range(len(df))]

# --- UI TABS ---
t1, t2, t3 = st.tabs(["🛡️ Middleware Processing", "📜 Security Logs", "📡 Final ATC Display"])
view = pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=2)

with t1:
    st.subheader("ADS-B Diagnostic Layer (Raw Signal Analysis)")
    mid_layer = pdk.Layer(
        "IconLayer", df, 
        get_icon="icon_data", get_size=10, size_scale=5,
        get_position='[lon, lat]', 
        get_color='[r, g, b]', # Uses colors assigned by logic.py
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(
        layers=[mid_layer], 
        initial_view_state=view, 
        tooltip={"text": "ID: {icao24}\nStatus: {status}\nSpeed: {velocity}"}
    ))
    
    st.markdown("""
    <div style='display:flex; gap:25px; justify-content:center; padding:10px; background:#1e1e1e; border-radius:10px;'>
        <b style='color:#00ff00'>● LEGITIMATE</b>
        <b style='color:#ff0000'>● IMPROPER PHYSICS (AI BLOCKED)</b>
        <b style='color:#ffa500'>● FAKE IDENTITY (RSA BLOCKED)</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Live Processing Stream")
    st.dataframe(df[['icao24', 'velocity', 'status', 'auth_passed']], use_container_width=True)

with t2:
    st.subheader("Real-Time Security Event Logs")
    for log in reversed(st.session_state.logs):
        if "ALERT" in log:
            st.error(log)
        else:
            st.info(log)

with t3:
    st.subheader("ATC Terminal (Secure Filtered View)")
    # Filter only legitimate aircraft for the final view
    atc_df = df[df['status'] == 'LEGITIMATE']
    
    atc_layer = pdk.Layer(
        "IconLayer", atc_df, 
        get_icon="icon_data", get_size=10, size_scale=5,
        get_position='[lon, lat]', 
        get_color='[0, 255, 0]', 
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[atc_layer], initial_view_state=view))
    
    c1, c2 = st.columns(2)
    c1.metric("Verified Aircraft", len(atc_df))
    c2.metric("Threats Mitigated", len(df) - len(atc_df))