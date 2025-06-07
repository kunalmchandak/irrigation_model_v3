from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import math
from datetime import datetime, timedelta

import joblib
import pandas as pd
import requests
import numpy as np
import sys, types

API_KEY = "f81d58a28607f320a77f8c4c9e76e77a"

def cyclical_encode(doy_series):
    theta = 2 * np.pi * doy_series.astype(float) / 365.0
    return np.column_stack([np.sin(theta), np.cos(theta)])

def doy_feature_names_out(_self, _input):
    return ["doy_sin", "doy_cos"]

main_mod = types.ModuleType("__main__")
main_mod.cyclical_encode = cyclical_encode
main_mod.doy_feature_names_out = doy_feature_names_out
sys.modules["__main__"] = main_mod

import joblib
model = joblib.load("model/water_predictor_model_v3.pkl")

# ─────── KC table stays the same ────────────────────────────────────────────
kc_table = {
    "wheat": {"seedling": 0.3, "vegetative": 0.7, "flowering": 1.1, "harvesting": 0.6},
    "tomato": {"seedling": 0.4, "vegetative": 0.75, "flowering": 1.05, "harvesting": 0.9},
    "paddy": {"seedling": 0.6, "vegetative": 1.05, "flowering": 1.2, "harvesting": 0.8},
    "maize": {"seedling": 0.35, "vegetative": 0.75, "flowering": 1.15, "harvesting": 0.5},
    "cotton": {"seedling": 0.4, "vegetative": 0.85, "flowering": 1.2, "harvesting": 0.6},
    "rice": {"seedling": 0.6, "vegetative": 1.1, "flowering": 1.2, "harvesting": 0.9},
    "soybean": {"seedling": 0.4, "vegetative": 0.85, "flowering": 1.05, "harvesting": 0.75},
}

# ─────── Penman–Monteith helpers (same maths as in the generator) ───────────
SIGMA  = 4.903e-9   # MJ K‑4 m‑2 day‑1
ALBEDO = 0.23
P_KPA  = 101.3      # rough sea‑level pressure; OK for Maharashtra plains
G_SOIL = 0          # daily timestep → soil heat flux ≈ 0

def _svp(T):              # saturation vapour pressure (kPa)
    return 0.6108 * math.exp((17.27 * T) / (T + 237.3))

def _delta_svp(T):        # slope Δ (kPa °C⁻1)
    return (4098 * _svp(T)) / ((T + 237.3) ** 2)

def _psy_const(P=P_KPA):  # psychrometric constant γ (kPa °C⁻1)
    return 0.000665 * P

def _ra(doy, lat_deg):
    """Extraterrestrial radiation Ra in MJ m‑2 day‑1 (FAO‑56 Eq 21‑25)."""
    lat = math.radians(lat_deg)
    dr  = 1 + 0.033 * math.cos(2 * math.pi / 365 * doy)
    δ   = 0.409 * math.sin(2 * math.pi / 365 * doy - 1.39)
    ws  = math.acos(-math.tan(lat) * math.tan(δ))
    Gsc = 0.0820
    return (24 * 60 / math.pi) * Gsc * dr * (
        ws * math.sin(lat) * math.sin(δ) +
        math.cos(lat) * math.cos(δ) * math.sin(ws)
    )

def et0_fao56(t_max, t_min, t_mean, rh, wind, rs, lat, doy, P=P_KPA):
    es   = (_svp(t_max) + _svp(t_min)) / 2
    ea   = es * rh / 100
    delta, gamma = _delta_svp(t_mean), _psy_const(P)

    rns = (1 - ALBEDO) * rs
    rso = (0.75 + 2e-5 * 0) * _ra(doy, lat)   # elevation≈0 m
    rnl = SIGMA * (
        ((t_max + 273.16)**4 + (t_min + 273.16)**4) / 2
    ) * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * (rs / rso) - 0.35)
    rn  = rns - rnl

    num = 0.408 * delta * (rn - G_SOIL) + gamma * (900 / (t_mean + 273)) * wind * (es - ea)
    den = delta + gamma * (1 + 0.34 * wind)
    return round(num / den, 3)          # mm day‑1

# ─────── Weather fetch (single call) ────────────────────────────────────────
def fetch_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_KEY}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    d = r.json()
    if "main" not in d or "wind" not in d or "clouds" not in d:
        raise ValueError("Incomplete weather payload")
    return {
        "lat": d["coord"]["lat"],
        "temp": d["main"]["temp"],
        "t_max": d["main"]["temp_max"],
        "t_min": d["main"]["temp_min"],
        "humidity": d["main"]["humidity"],
        "pressure": d["main"]["pressure"] / 10,
        "wind": d["wind"].get("speed", 0.0),
        "rain": d.get("rain", {}).get("1h", 0.0),
        "cloud": d["clouds"]["all"],
    }

# ─────── KC lookup ──────────────────────────────────────────────────────────
def kc(crop, stage):
    return kc_table.get(crop.lower(), {}).get(stage.lower(), 1.0)

# ─────── Interval heuristic (unchanged) ─────────────────────────────────────
def irrigation_interval(mm_per_day, stage, soil, rain_1h):
    if   mm_per_day < 0.25: base = 3
    elif mm_per_day < 0.5:  base = 2
    else:                   base = 1

    base += {"sandy":-1, "loamy":0, "clayey":1}.get(soil.lower(), 0)
    if stage.lower() == "flowering":  base -= 1
    if stage.lower() == "harvesting": base += 1
    if rain_1h >= 5: base += 2
    elif rain_1h >= 2: base += 1
    return max(base, 1)

# ─────── Master prediction routine ──────────────────────────────────────────
def predict_and_schedule(lat, lon, crop_type, crop_stage, soil_type="loamy"):
    w = fetch_weather(lat, lon)
    today_doy = datetime.now().timetuple().tm_yday
    rs = (1 - w["cloud"] / 100) * _ra(today_doy, w["lat"])   # MJ m‑2 d‑1

    eto = et0_fao56(
        t_max = w["t_max"],
        t_min = w["t_min"],
        t_mean = w["temp"],
        rh = w["humidity"],
        wind = w["wind"],
        rs = rs,
        lat = w["lat"],
        doy = today_doy,
        P = w["pressure"],
    )
    kc_val = kc(crop_type, crop_stage)

    # Build the exact feature frame expected by the v3 pipeline
    features = pd.DataFrame([{
        "temperature":      w["temp"],
        "humidity":         w["humidity"],
        "rainfall":         w["rain"],
        "wind_speed":       w["wind"],
        "solar_radiation":  rs,
        "day_of_year":      today_doy,
        "kc":               kc_val,
        "et0":              eto,
        "crop_type":        crop_type,
        "crop_stage":       crop_stage,
        "soil_type":        soil_type,
        "latitude": lat,
        "longitude": lon,
    }])

    water_pred = float(model.predict(features)[0])   # mm day‑1
    interval   = irrigation_interval(water_pred, crop_stage, soil_type, w["rain"])
    next_date = datetime.now() + timedelta(days=interval)

    return {
        "latitude": lat,
        "longitude": lon,
        "crop_type":         crop_type,
        "crop_stage":        crop_stage,
        "soil_type":         soil_type,
        "temperature_°C":    round(w["temp"], 2),
        "humidity_%":        round(w["humidity"], 1),
        "rainfall_mm_1h":    round(w["rain"], 2),
        "wind_speed_m_s":    round(w["wind"], 2),
        "solar_radiation_MJ_m2": round(rs, 2),
        "eto_mm_d":          eto,
        "kc":                kc_val,
        "predicted_water_mm_d": round(water_pred, 2),
        "interval_days":     interval,
        "next_irrigation":   next_date.strftime("%Y-%m-%d"),
    }

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(
    lat: float = Form(...),
    lon: float = Form(...),
    crop_type: str = Form(...),
    crop_stage: str = Form(...),
    soil_type: str = Form("loamy")
):
    return predict_and_schedule(lat, lon, crop_type, crop_stage, soil_type)
