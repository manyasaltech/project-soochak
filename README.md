# Project Soochak: Mine Assessment & Terrain Response
**Smart India Hackathon 2026 | Problem Statement ID: 26025**  
**Theme**: Smart Automation | **Category**: Hardware & AI  
**Team**: Team MATR  

---

## Overview
Project Soochak is an AI-enabled low-cost real-time mine subsidence monitoring, prediction, and early warning system for underground coal mines in India (demonstrated on Jharia Coalfield Panel 4-B).

## System Architecture
1. **Multi-Sensor Edge Fleet (data_engine.py)**:
   - **MPU6050**: Tilt angles (Pitch, Roll, Tilt magnitude) and 3-axis vibration acceleration (RMS & peak g).
   - **VL53L1X**: Laser Time-of-Flight relative crack displacement (mm) and rate of change.
   - **HX711 + Load Cell**: Hydraulic prop load and strata roof strain (kN).
   - **BME280 / DS18B20**: Ambient temperature (°C), humidity (%), barometric pressure (hPa).
   - **MQ-4**: Underground methane gas concentration (ppm).
   - **Long-Baseline Laser Reference System**: 980m independent optical baseline tracking (X, Y) spot deviation in millimeters.

2. **Machine Learning & Spatial Rejection Engine (data_engine.py)**:
   - **Isolation Forest**: Multi-sensor anomaly detection trained on normal baseline operation.
   - **Multi-Node Spatial Correlation**:
     - *Blasting Shock Rejection*: Filters out transient blasting vibrations when displacement and laser remain stable.
     - *Single-Node Glitch Suppression*: Prevents false mine evacuations from isolated sensor noise or detachment.
     - *Strata Creep Warning*: Early detection of pre-subsidence micro-deformation.
     - *Critical Subsidence Evacuation*: Multi-node coordinated deformation confirmed by optical laser deviation.

3. **Interactive GIS Heatmap Dashboard (app.py)**:
   - **Folium & Streamlit Integration**: Real-time Leaflet HeatMap displaying continuous anomaly score gradient over the mining panel.
   - **Sensor Node Markers**: Interactive popups with full hardware telemetry for all 8 nodes.
   - **Laser Reference Visualizer**: Optical path line between Transmitter Tx-1 and Detector Rx-1.
   - **Telemetry Analytics**: Plotly time-series charts, radar multi-sensor fingerprint, and real-time alert log.

---

## How to Launch

### Option 1: Direct Double-Click (Windows)
Double-click `run_app.bat` inside this folder.

### Option 2: Command Line
```powershell
cd C:\Users\KANAV\.gemini\antigravity\scratch\anomaly-heatmap-app
.venv\Scripts\streamlit.exe run app.py
```

Then open your browser at `http://localhost:8501`.
