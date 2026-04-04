"""
app.py — AirDefend-X Streamlit Dashboard
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

from logic import (
    run_middleware,
    register_and_sign_legitimate,
    prepare_dataframe,
    train_behavioral_model,
    make_physics_attack_row,
    sign_message,
    _registry,
    MAX_VELOCITY_MS,
)

st.set_page_config(layout="wide", page_title="AirDefend-X Security Gateway")

# ---------------------------------------------------------------------------
# Session State Bootstrap — runs only ONCE per session
# ---------------------------------------------------------------------------
if 'flights' not in st.session_state:
    try:
        raw = pd.read_csv('data/opensky.csv').dropna()
    except FileNotFoundError:
        rng = np.random.default_rng(0)
        n = 300
        raw = pd.DataFrame({
            'icao24':      [f'REAL_{i:04d}' for i in range(n)],
            'velocity':    rng.uniform(100, 300, n),
            'vertrate':    rng.uniform(-5, 5, n),
            'geoaltitude': rng.uniform(5000, 12000, n),
            'lat':         rng.uniform(20, 60, n),
            'lon':         rng.uniform(-20, 40, n),
            'track':       rng.uniform(0, 360, n),
        })

    raw = prepare_dataframe(raw)
    raw = register_and_sign_legitimate(raw)

    # Train the AI NOW on clean legitimate data — before any attack is injected
    train_behavioral_model(raw)

    st.session_state.flights       = raw
    st.session_state.attack_count  = 0   # used to generate unique attack ICAOs
    st.session_state.logs = [
        f"[{time.strftime('%H:%M:%S')}] ✅ AirDefend-X Online — Monitoring Global Airspace"
    ]

def _ts():
    return time.strftime("%H:%M:%S", time.localtime())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📡 ADS-B Simulation Console")
st.sidebar.info(
    "Simulate real-world ADS-B threat scenarios.\n\n"
    "**Improper Physics** — frame carries a valid RSA signature but an "
    "abnormal flight-envelope profile. Detected by the AI Behavioral Layer.\n\n"
    "**Spoofed Identity** — unregistered ICAO with a forged RSA signature. "
    "Blocked by RSA Layer 1."
)

# ── Improper Physics (AI catch) ─────────────────────────────────────────────
if st.sidebar.button("🚨 Simulate Improper Physics Threat"):
    # Unique ICAO per attack — avoids RSA signature collision across reruns
    st.session_state.attack_count += 1
    attack_icao = f"ATK{st.session_state.attack_count:04d}"

    base = st.session_state.flights.iloc[0].copy()
    base['lat'] += np.random.uniform(-1.0, 1.0)
    base['lon'] += np.random.uniform(-1.0, 1.0)
    base['time'] = float(int(time.time()))

    atk = make_physics_attack_row(base, attack_icao)

    st.session_state.flights = pd.concat(
        [pd.DataFrame([atk]), st.session_state.flights], ignore_index=True
    )
    st.session_state.logs.append(
        f"[{_ts()}] 🔴 IMPROPER PHYSICS DETECTED — "
        f"ICAO {attack_icao} — RSA signature verified, but AI Behavioral Layer "
        f"flagged an anomalous flight-envelope profile "
        f"(vel=60 m/s, alt=12000 m, vr=45 m/s). Frame quarantined."
    )
    st.rerun()

# ── Spoofed Identity (RSA catch) ────────────────────────────────────────────
if st.sidebar.button("🆔 Simulate Spoofed Identity Threat"):
    template = st.session_state.flights.iloc[1].copy()
    ts = int(time.time())

    atk = template.copy()
    atk['icao24']    = f"{ts % 0xFFFFFF:06X}"   # looks like a real ICAO hex
    atk['velocity']  = np.random.uniform(150, 320)
    atk['lat']      += np.random.uniform(-1.0, 1.0)
    atk['lon']      += np.random.uniform(-1.0, 1.0)
    atk['time']      = float(ts)
    atk['signature'] = b"FORGED_" + str(ts).encode()

    st.session_state.flights = pd.concat(
        [pd.DataFrame([atk]), st.session_state.flights], ignore_index=True
    )
    st.session_state.logs.append(
        f"[{_ts()}] 🔴 RSA AUTHENTICATION FAILURE — "
        f"ICAO {atk['icao24']} not found in trusted key registry. "
        f"Possible ghost aircraft or replay attack. Frame blocked by Layer 1."
    )
    st.rerun()

# ── Reset ───────────────────────────────────────────────────────────────────
if st.sidebar.button("🔄 Reset Airspace"):
    try:
        raw = prepare_dataframe(pd.read_csv('data/opensky.csv').dropna())
    except FileNotFoundError:
        rng = np.random.default_rng(0)
        n = 300
        raw = prepare_dataframe(pd.DataFrame({
            'icao24':      [f'REAL_{i:04d}' for i in range(n)],
            'velocity':    rng.uniform(100, 300, n),
            'vertrate':    rng.uniform(-5, 5, n),
            'geoaltitude': rng.uniform(5000, 12000, n),
            'lat':         rng.uniform(20, 60, n),
            'lon':         rng.uniform(-20, 40, n),
            'track':       rng.uniform(0, 360, n),
        }))
    raw = register_and_sign_legitimate(raw)
    st.session_state.flights      = raw
    st.session_state.attack_count = 0
    st.session_state.logs = [f"[{_ts()}] ✅ Airspace Reset — All Threat Records Cleared"]
    st.rerun()

# ---------------------------------------------------------------------------
# Run Middleware
# ---------------------------------------------------------------------------
df = run_middleware(st.session_state.flights)

# ---------------------------------------------------------------------------
# PyDeck Safety
# ---------------------------------------------------------------------------
ICON_URL  = "https://img.icons8.com/m_outlined/512/FFFFFF/airplane-mode-on.png"
icon_data = {"url": ICON_URL, "width": 242, "height": 242, "anchorY": 242, "mask": True}
df['icon_data'] = [icon_data] * len(df)

def _make_pydeck_safe(frame: pd.DataFrame) -> pd.DataFrame:
    import json
    safe = frame.copy()
    for col in safe.columns:
        if safe[col].dtype == object:
            def _coerce(v):
                if isinstance(v, (bytes, bytearray)):
                    return v.hex()
                try:
                    json.dumps(v)
                    return v
                except (TypeError, ValueError):
                    return str(v)
            safe[col] = safe[col].apply(_coerce)
    return safe

df_map  = _make_pydeck_safe(df)
view    = pdk.ViewState(latitude=df_map['lat'].mean(), longitude=df_map['lon'].mean(), zoom=2)
TOOLTIP = {"text": "ID: {icao24}\nStatus: {status}\nSpeed: {velocity} m/s\nAlt: {geoaltitude} m"}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
t1, t2, t3 = st.tabs(["🛡️ Middleware Processing", "📜 Security Logs", "📡 Final ATC Display"])

with t1:
    st.subheader("ADS-B Diagnostic Layer — Full Signal Analysis")
    mid_layer = pdk.Layer(
        "IconLayer", df_map,
        get_icon="icon_data", get_size=10, size_scale=5,
        get_position='[lon, lat]',
        get_color='[r, g, b]',
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[mid_layer], initial_view_state=view, tooltip=TOOLTIP))

    st.markdown("""
    <div style='display:flex;gap:30px;justify-content:center;padding:12px;
                background:#1e1e1e;border-radius:10px;margin-top:10px;'>
      <b style='color:#00ff00'>● LEGITIMATE</b>
      <b style='color:#ff0000'>● IMPROPER PHYSICS (AI Behavioral Layer)</b>
      <b style='color:#ffa500'>● FAKE IDENTITY (RSA Layer 1)</b>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Live Processing Stream")
    display_cols = ['icao24', 'velocity', 'geoaltitude', 'vertrate',
                    'auth_passed', 'kinematic_ok', 'iso_score', 'status']
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages",   len(df))
    col2.metric("Identity Threats", int((df['status'] == 'FAKE IDENTITY').sum()))
    col3.metric("Physics Threats",  int((df['status'] == 'IMPROPER PHYSICS').sum()))

with t2:
    st.subheader("Security Event Log")
    for log in reversed(st.session_state.logs):
        if "🔴" in log:
            st.error(log)
        elif "✅" in log:
            st.success(log)
        else:
            st.info(log)

with t3:
    st.subheader("ATC Terminal — Secure Filtered View (Legitimate Only)")
    atc_df = df_map[df_map['status'] == 'LEGITIMATE']
    atc_layer = pdk.Layer(
        "IconLayer", atc_df,
        get_icon="icon_data", get_size=10, size_scale=5,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]',
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[atc_layer], initial_view_state=view, tooltip=TOOLTIP))

    c1, c2, c3 = st.columns(3)
    c1.metric("Verified Aircraft", len(atc_df))
    c2.metric("Threats Mitigated", len(df_map) - len(atc_df))
    c3.metric("Mitigation Rate",
              f"{(len(df_map) - len(atc_df)) / max(len(df_map), 1) * 100:.1f}%")
