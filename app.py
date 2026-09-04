

"""
Project Soochak - Mine Assessment & Terrain Response
Smart India Hackathon 2026 | Problem Statement 26025
Team MATR

app.py:
Streamlit & Folium Interactive Dashboard for Real-Time Mine Subsidence
Monitoring, Prediction, and Early Warning System.
"""

import time
import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

# Deep Learning & Data Engine Imports
from deep_engine import run_dl_inference
from graph_topology import build_adjacency_matrix, NODE_COORDS
from data_engine import (
    MineEnvironmentSimulator,
    OperationalScenario,
    ThreatLevel,
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Project Soochak | Mine Subsidence Monitoring",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Mine Control Dashboard
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .stMetric {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(75, 85, 99, 0.4);
        padding: 12px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .badge-safe {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    .badge-watch {
        background-color: #075985;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #0284c7;
    }
    .badge-warning {
        background-color: #854d0e;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #d97706;
    }
    .badge-critical {
        background-color: #881337;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #dc2626;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .header-box {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border-left: 5px solid #3b82f6;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .recommendation-box {
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. SESSION STATE & SIMULATOR INITIALIZATION
# -----------------------------------------------------------------------------
if "simulator" not in st.session_state:
    with st.spinner("Initializing Project Soochak AI Engine & Training Isolation Forest..."):
        st.session_state.simulator = MineEnvironmentSimulator()
        st.session_state.current_data = st.session_state.simulator.step(OperationalScenario.NORMAL)
        st.session_state.auto_refresh = False

sim: MineEnvironmentSimulator = st.session_state.simulator


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & HARDWARE STACK
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/mine-cart.png", width=64)
    st.title("Project Soochak")
    st.caption("Mine Assessment & Terrain Response | SIH 2026 (PS ID 26025)")
    st.markdown("**Team MATR** - Hardware & AI Early Warning")
    st.markdown("---")

    st.subheader("🎮 Simulation Control")
    scenario_options = {
        "Normal Operations (Baseline Ambient)": OperationalScenario.NORMAL,
        "Controlled Blasting (False Alarm Test)": OperationalScenario.BLASTING,
        "Strata Creep (Pre-Subsidence Micro-Deformation)": OperationalScenario.STRATA_CREEP,
        "Critical Subsidence Event (Emergency Alert)": OperationalScenario.CRITICAL_SUBSIDENCE,
        "Sensor Hardware Glitch (Drift Suppression)": OperationalScenario.SENSOR_GLITCH,
    }

    selected_scenario_label = st.selectbox(
        "Operational Regime",
        list(scenario_options.keys()),
        index=0,
        help="Select geological or operational condition to test AI response."
    )
    current_scenario = scenario_options[selected_scenario_label]

    st.markdown("---")
    st.subheader("🧠 Select AI Model Engine")

    view_mode = st.radio(
        "Choose Analysis Mode:",
        [
            "Real-Time Anomaly Detection (Isolation Forest)",
            "+12h Predictive Risk (STGCN-LSTM)"
        ]
    )

    st.markdown("##### Manual Disturbance Injection")
    manual_disp = st.slider("Additional Displacement (mm)", 0.0, 10.0, 0.0, 0.1)
    manual_tilt = st.slider("Additional Tilt (deg)", 0.0, 5.0, 0.0, 0.1)
    manual_laser = st.slider("Laser Spot Drift (mm)", 0.0, 6.0, 0.0, 0.1)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        step_clicked = st.button("⚡ Next Step", use_container_width=True)
    with col_btn2:
        reset_clicked = st.button("🔄 Reset Base", use_container_width=True)

    auto_run = st.checkbox("🔄 Auto-Run Live Simulation", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_run

    if reset_clicked:
        st.session_state.simulator = MineEnvironmentSimulator()
        sim = st.session_state.simulator
        st.session_state.current_data = sim.step(OperationalScenario.NORMAL)
        st.rerun()

    if step_clicked:
        st.session_state.current_data = sim.step(
            scenario=current_scenario,
            manual_disp_offset=manual_disp,
            manual_tilt_offset=manual_tilt,
            manual_laser_offset=manual_laser,
        )
        st.rerun()

    st.markdown("---")
    st.subheader("🛠️ Hardware Stack Specs")
    st.markdown("""
    - **Controller**: ESP32 (ESP-NOW / LoRa Mesh)
    - **Tilt / Movement**: MPU6050 6-DoF IMU
    - **Displacement**: VL53L1X Laser ToF
    - **Strata Load**: HX711 + Strain Gauge Cell
    - **Optical Ref**: 980m Baseline Laser Receiver
    - **Atmosphere**: BME280 + DS18B20 + MQ-4
    """)

    st.markdown("---")
    st.caption("AI Model: Scikit-learn Isolation Forest (120 Estimators, Contamination 0.02) + Spatial Cross-Validation.")


# If Auto-run is active, advance step
if st.session_state.auto_refresh and not step_clicked:
    st.session_state.current_data = sim.step(
        scenario=current_scenario,
        manual_disp_offset=manual_disp,
        manual_tilt_offset=manual_tilt,
        manual_laser_offset=manual_laser,
    )

data = st.session_state.current_data
nodes = data["nodes"]
laser = data["laser_system"]
threat_level = data["threat_level"]
peak_score = data["peak_anomaly_score"]
rejection_event = data["rejection_event"]
false_alarms = data["false_alarms_rejected"]
history = data["history"]
logs = data["logs"]


# -----------------------------------------------------------------------------
# 4. HEADER & TOP METRICS ROW
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h2 style="margin: 0; color: #60a5fa; font-size: 1.8rem;">
                PROJECT SOOCHAK: MINE SUBSIDENCE MONITORING
            </h2>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Smart Multi-Sensor System with Isolation Forest ML & GIS Spatial Heatmap | Jharia Coalfield Panel 4-B
            </p>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 0.85rem; color: #94a3b8;">SIH 2026 Problem Statement ID: <b>26025</b></span><br/>
            <span style="font-size: 0.85rem; color: #38bdf8;">Team MATR</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if threat_level == "SAFE":
        st.markdown("**Mine Threat Level**")
        st.markdown("<span class='badge-safe'>🟢 SAFE OPERATIONS</span>", unsafe_allow_html=True)
    elif threat_level == "WATCH":
        st.markdown("**Mine Threat Level**")
        st.markdown("<span class='badge-watch'>🔵 WATCH LEVEL</span>", unsafe_allow_html=True)
    elif threat_level == "WARNING":
        st.markdown("**Mine Threat Level**")
        st.markdown("<span class='badge-warning'>🟠 WARNING (STRATA CREEP)</span>", unsafe_allow_html=True)
    else:
        st.markdown("**Mine Threat Level**")
        st.markdown("<span class='badge-critical'>🔴 CRITICAL EVACUATION</span>", unsafe_allow_html=True)

with c2:
    st.metric(
        label="Peak ML Anomaly Score",
        value=f"{peak_score:.3f}",
        delta=f"Threshold: 0.70",
        delta_color="inverse" if peak_score >= 0.70 else "normal",
    )

with c3:
    max_disp = max(n["displacement_mm"] for n in nodes)
    st.metric(
        label="Max Displacement (VL53L1X)",
        value=f"{max_disp:.2f} mm",
        delta="Rate: {:.2f} mm/min".format(max(n["displacement_rate_mm_min"] for n in nodes)),
    )

with c4:
    laser_dev = laser["total_deviation_mm"]
    st.metric(
        label="Laser Optical Ref Deviation",
        value=f"{laser_dev:.2f} mm",
        delta=f"{laser['status']}",
        delta_color="normal" if laser_dev < 1.0 else "inverse",
    )

with c5:
    st.metric(
        label="False Alarms Suppressed",
        value=f"{false_alarms}",
        delta="Persistent Spatial AI",
    )

# Dynamic Early Warning Advisory
if threat_level == "SAFE":
    st.markdown("""
    <div class="recommendation-box" style="background: rgba(6, 95, 70, 0.25); border-left: 4px solid #10b981; color: #a7f3d0;">
        <b>✅ STATUS NORMAL:</b> Underground strata equilibrium within normal geological limits. All longwall extraction operations authorized.
    </div>
    """, unsafe_allow_html=True)
elif threat_level == "WATCH":
    st.markdown("""
    <div class="recommendation-box" style="background: rgba(7, 89, 133, 0.25); border-left: 4px solid #0284c7; color: #bae6fd;">
        <b>ℹ️ ATTENTION REQUIRED:</b> Subtle baseline variances detected in peripheral nodes. Telemetry polling rate stepped up to 5s.
    </div>
    """, unsafe_allow_html=True)
elif threat_level == "WARNING":
    st.markdown("""
    <div class="recommendation-box" style="background: rgba(133, 77, 14, 0.25); border-left: 4px solid #f59e0b; color: #fde68a;">
        <b>⚠️ STRATA CREEP WARNING:</b> Continuous micro-deformation identified across Goaf Margins (avg displacement > 2mm). Geotechnical inspection of hydraulic supports in Panel 4-B advised.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="recommendation-box" style="background: rgba(136, 19, 55, 0.35); border-left: 4px solid #ef4444; color: #fecaca;">
        <b>🚨 CRITICAL SUBSIDENCE ALERT:</b> Coordinated multi-node ground movement (>5mm) confirmed by Long-Baseline Laser Reference (>2.5mm). Initiate immediate evacuation of Panel 4-B!
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. FOLIUM REAL-TIME HEATMAP & GIS VISUALIZATION
# -----------------------------------------------------------------------------
adj_matrix = build_adjacency_matrix(NODE_COORDS)

if view_mode == "+12h Predictive Risk (STGCN-LSTM)":
    st.info("🔮 **Deep Learning Engine Active:** Predicting +12 Hour Strata Movement across ESP32 Nodes using STGCN + LSTM.")
    sample_history = np.random.randn(24, 8, 5)
    dl_preds = run_dl_inference(sample_history, adj_matrix)
    
    heatmap_data = [
        [NODE_COORDS[i][0], NODE_COORDS[i][1], float(dl_preds[i])]
        for i in range(len(NODE_COORDS))
    ]
else:
    st.success("🟢 **Classical ML Engine Active:** Real-Time Isolation Forest Anomaly Detection.")
    heatmap_data = [
        [node["lat"], node["lon"], float(node["anomaly_score"])]
        for node in nodes
    ]

st.subheader("🗺️ Real-Time GIS Heatmap & Distributed Sensor Mesh")

center_lat = 23.7525
center_lon = 86.4200

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=16,
    tiles="CartoDB positron",
    control_scale=True,
)

panel_polygon = [
    [23.7542, 86.4155],
    [23.7548, 86.4235],
    [23.7500, 86.4242],
    [23.7495, 86.4162],
]
folium.Polygon(
    locations=panel_polygon,
    color="#3b82f6",
    weight=2,
    dash_array="5, 5",
    fill=True,
    fill_color="#1d4ed8",
    fill_opacity=0.08,
    popup="<b>Underground Longwall Panel 4-B</b><br>Target Coal Seam extraction block (Depth: 185m).",
).add_to(m)

goaf_polygon = [
    [23.7530, 86.4175],
    [23.7520, 86.4215],
    [23.7508, 86.4210],
    [23.7518, 86.4170],
]
folium.Polygon(
    locations=goaf_polygon,
    color="#f97316",
    weight=1.5,
    dash_array="3, 3",
    fill=True,
    fill_color="#ea580c",
    fill_opacity=0.12,
    popup="<b>Active Goaf Caving Zone</b><br>Subsidence-prone decompressed roof strata.",
).add_to(m)

laser_path = [
    [laser["tx_lat"], laser["tx_lon"]],
    [laser["rx_lat"], laser["rx_lon"]],
]
laser_line_color = "#ef4444" if laser["total_deviation_mm"] > 2.5 else ("#f59e0b" if laser["total_deviation_mm"] > 0.8 else "#10b981")
folium.PolyLine(
    locations=laser_path,
    color=laser_line_color,
    weight=3,
    dash_array="8, 4",
    opacity=0.85,
    popup=f"<b>Long-Baseline Laser Reference (980m)</b><br>Total Deviation: {laser['total_deviation_mm']:.2f} mm<br>Status: {laser['status']}",
).add_to(m)

folium.Marker(
    location=[laser["tx_lat"], laser["tx_lon"]],
    popup=f"<b>{laser['tx_name']}</b><br>Continuous collimated optical emitter.",
    icon=folium.Icon(color="darkblue", icon="bullseye", prefix="fa"),
).add_to(m)

folium.Marker(
    location=[laser["rx_lat"], laser["rx_lon"]],
    popup=f"<b>{laser['rx_name']}</b><br>High-precision photodiode detector array.<br>Spot Deviation: ({laser['spot_x_mm']}mm, {laser['spot_y_mm']}mm)",
    icon=folium.Icon(color="purple", icon="crosshairs", prefix="fa"),
).add_to(m)

heat_points = []
for node in nodes:
    lat, lon = node["lat"], node["lon"]
    score = node["anomaly_score"]
    heat_points.append([lat, lon, score])

    if score > 0.15:
        for offset_lat in [-0.0003, 0.0003]:
            for offset_lon in [-0.0003, 0.0003]:
                heat_points.append([lat + offset_lat, lon + offset_lon, score * 0.75])

if laser["total_deviation_mm"] > 1.0:
    mid_lat = (laser["tx_lat"] + laser["rx_lat"]) / 2
    mid_lon = (laser["tx_lon"] + laser["rx_lon"]) / 2
    heat_points.append([mid_lat, mid_lon, min(1.0, laser["total_deviation_mm"] / 5.0)])

gradient_config = {
    0.1: "#10b981",
    0.3: "#06b6d4",
    0.5: "#f59e0b",
    0.75: "#f97316",
    1.0: "#ef4444",
}

HeatMap(
    data=heat_points,
    min_opacity=0.35,
    max_val=1.0,
    radius=38,
    blur=26,
    gradient=gradient_config,
).add_to(m)

for node in nodes:
    score = node["anomaly_score"]
    status = node["status"]

    if status == "CRITICAL":
        color = "#ef4444"
    elif status == "WARNING":
        color = "#f97316"
    elif status == "WATCH":
        color = "#0284c7"
    else:
        color = "#10b981"

    popup_html = f"""
    <div style="font-family: Arial, sans-serif; min-width: 220px; font-size: 12px; line-height: 1.4;">
        <h4 style="margin: 0 0 6px 0; color: #1e293b; border-bottom: 2px solid {color}; padding-bottom: 4px;">
            Node {node['node_id']} - {node['name']}
        </h4>
        <b>Zone:</b> {node['panel_zone']} (Depth: {node['depth_m']}m)<br/>
        <b>Threat Status:</b> <span style="color:{color}; font-weight:bold;">{status}</span><br/>
        <b>ML Anomaly Score:</b> <b>{score:.3f}</b><br/>
        <hr style="margin: 6px 0; border: 0; border-top: 1px solid #e2e8f0;"/>
        <b>MPU6050 Tilt:</b> {node['tilt_magnitude_deg']}° (P: {node['pitch_deg']}°, R: {node['roll_deg']}°)<br/>
        <b>MPU6050 Vibration:</b> RMS {node['vibration_rms_g']}g | Peak {node['vibration_peak_g']}g<br/>
        <b>VL53L1X Displacement:</b> <b>{node['displacement_mm']:.2f} mm</b> ({node['displacement_rate_mm_min']:+.2f} mm/min)<br/>
        <b>HX711 Prop Load:</b> {node['load_kN']:.1f} kN (Δ {node['load_delta_kN']:+.1f} kN)<br/>
        <b>Gas / Environment:</b> CH₄ {node['methane_ppm']:.0f} ppm | Temp {node['temp_c']:.1f}°C<br/>
        <b>Hardware Health:</b> Battery {node['battery_pct']:.0f}% | RSSI {node['rssi_dbm']} dBm
    </div>
    """

    folium.CircleMarker(
        location=[node["lat"], node["lon"]],
        radius=9,
        color="#ffffff",
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=0.95,
        tooltip=f"<b>Node {node['node_id']}</b>: {node['panel_zone']} | Score: {score:.3f} ({status})",
        popup=folium.Popup(popup_html, max_width=320),
    ).add_to(m)

map_col, info_col = st.columns([3, 1])

with map_col:
    st_folium(m, width=None, height=540, returned_objects=[])

with info_col:
    st.markdown("##### 📍 Active Sensor Fleet")
    for n in nodes:
        badge_cls = (
            "badge-critical" if n["status"] == "CRITICAL"
            else ("badge-warning" if n["status"] == "WARNING"
                  else ("badge-watch" if n["status"] == "WATCH" else "badge-safe"))
        )
        st.markdown(f"""
        <div style="background: #111827; padding: 6px 10px; border-radius: 6px; margin-bottom: 6px; border: 1px solid #374151; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b>{n['node_id']}</b>: <span style="font-size:0.8rem; color:#9ca3af;">{n['panel_zone'][:18]}...</span>
            </div>
            <span class="{badge_cls}">{n['anomaly_score']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. ANALYTICS & TIME-SERIES TELEMETRY (PLOTLY)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Multi-Sensor Telemetry & Spatial Correlation")

tab_trends, tab_spatial, tab_table, tab_logs = st.tabs([
    "📈 Historical Telemetry Trends",
    "🎯 Spatial Cross-Correlation Radar",
    "📋 Sensor Fleet Telemetry Grid",
    "🛡️ False Alarm & Alert Audit Log",
])

if history:
    df_hist = pd.DataFrame(history)
else:
    df_hist = pd.DataFrame()

with tab_trends:
    if not df_hist.empty:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            fig_score = go.Figure()
            fig_score.add_trace(go.Scatter(
                x=df_hist["step"],
                y=df_hist["peak_anomaly_score"],
                mode="lines+markers",
                name="Peak Anomaly Score",
                line=dict(color="#ef4444", width=2.5),
            ))
            fig_score.add_trace(go.Scatter(
                x=df_hist["step"],
                y=df_hist["mean_anomaly_score"],
                mode="lines",
                name="Mean Mine Score",
                line=dict(color="#3b82f6", width=1.5, dash="dot"),
            ))
            fig_score.add_hline(y=0.70, line_dash="dash", line_color="#dc2626", annotation_text="Critical Threshold (0.70)")
            fig_score.add_hline(y=0.40, line_dash="dash", line_color="#d97706", annotation_text="Warning Threshold (0.40)")
            fig_score.update_layout(
                title="Isolation Forest Anomaly Score vs Time Steps",
                xaxis_title="Simulation Step",
                yaxis_title="Normalized Score [0, 1]",
                template="plotly_dark",
                height=320,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_score, use_container_width=True)

        with col_t2:
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter(
                x=df_hist["step"],
                y=df_hist["max_displacement_mm"],
                mode="lines+markers",
                name="Max Crack Displacement (mm)",
                line=dict(color="#f59e0b", width=2),
            ))
            fig_disp.add_trace(go.Scatter(
                x=df_hist["step"],
                y=df_hist["laser_deviation_mm"],
                mode="lines+markers",
                name="Laser Reference Deviation (mm)",
                line=dict(color="#10b981", width=2, dash="dash"),
            ))
            fig_disp.update_layout(
                title="Ground Displacement vs Long-Baseline Optical Drift",
                xaxis_title="Simulation Step",
                yaxis_title="Displacement (mm)",
                template="plotly_dark",
                height=320,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_disp, use_container_width=True)

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            fig_tilt = px.line(
                df_hist,
                x="step",
                y="max_tilt_deg",
                title="Peak MPU6050 Tilt Magnitude (°)",
                template="plotly_dark",
                markers=True,
            )
            fig_tilt.update_traces(line_color="#a855f7")
            fig_tilt.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig_tilt, use_container_width=True)

        with col_t4:
            fig_load = px.line(
                df_hist,
                x="step",
                y="mean_load_kN",
                title="Average Hydraulic Prop Load (kN)",
                template="plotly_dark",
                markers=True,
            )
            fig_load.update_traces(line_color="#38bdf8")
            fig_load.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig_load, use_container_width=True)

with tab_spatial:
    col_s1, col_s2 = st.columns([3, 2])

    with col_s1:
        df_nodes = pd.DataFrame(nodes)
        fig_bar = px.bar(
            df_nodes,
            x="node_id",
            y="anomaly_score",
            color="status",
            color_discrete_map={
                "SAFE": "#10b981",
                "WATCH": "#0284c7",
                "WARNING": "#f59e0b",
                "CRITICAL": "#ef4444",
            },
            title="Node-Level ML Anomaly Scores across Mining Panel",
            hover_data=["name", "displacement_mm", "tilt_magnitude_deg", "load_kN"],
            template="plotly_dark",
        )
        fig_bar.add_hline(y=0.70, line_dash="dash", line_color="#ef4444", annotation_text="Critical (0.70)")
        fig_bar.add_hline(y=0.40, line_dash="dash", line_color="#f59e0b", annotation_text="Warning (0.40)")
        fig_bar.update_layout(height=340, margin=dict(l=30, r=20, t=40, b=30))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_s2:
        crit_node = max(nodes, key=lambda n: n["anomaly_score"])
        categories = ["Displacement", "Tilt Angle", "Vibration RMS", "Load Delta", "Laser Shift"]
        norm_values = [
            min(1.0, crit_node["displacement_mm"] / 5.0),
            min(1.0, crit_node["tilt_magnitude_deg"] / 3.0),
            min(1.0, crit_node["vibration_rms_g"] / 0.8),
            min(1.0, max(0.0, crit_node["load_delta_kN"]) / 80.0),
            min(1.0, laser["total_deviation_mm"] / 3.0),
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_values,
            theta=categories,
            fill='toself',
            name=f"Node {crit_node['node_id']}",
            line_color="#f43f5e",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"Multi-Sensor Fingerprint ({crit_node['node_id']})",
            template="plotly_dark",
            height=340,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with tab_table:
    df_display = pd.DataFrame(nodes)[[
        "node_id", "name", "panel_zone", "depth_m", "displacement_mm",
        "displacement_rate_mm_min", "tilt_magnitude_deg", "vibration_rms_g",
        "load_kN", "load_delta_kN", "methane_ppm", "temp_c", "battery_pct",
        "anomaly_score", "status"
    ]]
    df_display.columns = [
        "Node ID", "Name", "Panel Zone", "Depth (m)", "Displacement (mm)",
        "Disp Rate (mm/m)", "Tilt (°)", "Vibration (g)", "Load (kN)",
        "Δ Load (kN)", "CH₄ (ppm)", "Temp (°C)", "Batt (%)", "ML Score", "Status"
    ]
    st.dataframe(
        df_display.style.format({
            "Displacement (mm)": "{:.2f}",
            "Disp Rate (mm/m)": "{:+.2f}",
            "Tilt (°)": "{:.2f}",
            "Vibration (g)": "{:.3f}",
            "Load (kN)": "{:.1f}",
            "Δ Load (kN)": "{:+.1f}",
            "CH₄ (ppm)": "{:.0f}",
            "Temp (°C)": "{:.1f}",
            "Batt (%)": "{:.0f}%",
            "ML Score": "{:.3f}",
        }),
        use_container_width=True,
        height=320,
    )

with tab_logs:
    st.markdown("##### 🛡️ Multi-Node AI Correlation & False Alarm Audit Log")
    if logs:
        for log in logs:
            b_type = log.get("type", "INFO")
            if b_type == "FILTERED_ALARM":
                icon = "🛡️"
                color = "#38bdf8"
            elif b_type == "SUBSIDENCE_ALERT":
                icon = "🚨"
                color = "#f87171"
            elif b_type == "STRATA_CREEP":
                icon = "⚠️"
                color = "#fbbf24"
            else:
                icon = "ℹ️"
                color = "#94a3b8"

            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.6); border-left: 3px solid {color}; padding: 8px 12px; border-radius: 4px; margin-bottom: 8px; font-size: 0.9rem;">
                <span style="color: #94a3b8; font-size: 0.8rem; font-family: monospace;">[{log['time']}]</span>
                <span style="font-weight: 600; color: {color}; margin-left: 8px;">{icon} {b_type}</span>:
                <span style="color: #e2e8f0; margin-left: 6px;">{log['message']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alert events recorded yet.")

# Auto-refresh rerun delay
if st.session_state.auto_refresh:
    time.sleep(2.0)
    st.rerun()


