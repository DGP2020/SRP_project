from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

app = Flask(__name__, static_folder="../frontend")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

JSON_MODEL = os.path.join(MODEL_DIR, "awg_xgb_model.json")
JOBLIB_MODEL = os.path.join(MODEL_DIR, "awg_xgb_model.joblib")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
TARGET = metadata["target"]

model = None

if os.path.exists(JOBLIB_MODEL):
    try:
        model = joblib.load(JOBLIB_MODEL)
    except Exception:
        model = None

if model is None and os.path.exists(JSON_MODEL):
    model = XGBRegressor()
    model.load_model(JSON_MODEL)

if model is None:
    raise FileNotFoundError("XGBoost model could not be loaded.")

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

def dew_point(temp, humidity):
    a = 17.27
    b = 237.7
    alpha = ((a * temp) / (b + temp)) + math.log(humidity / 100.0)
    return (b * alpha) / (a - alpha)

def get_value(data, key, default=0):
    value = data.get(key, default)
    if value is None or value == "":
        return default
    return float(value)

def build_features(data):
    month_value = data.get("month", 1)

    if isinstance(month_value, str):
        month = MONTHS.get(month_value, 1)
    else:
        month = int(month_value)

    day = get_value(data, "day", 1)
    solar_hours = np.nan
    tevp = get_value(data, "tevp")
    thot = get_value(data, "thot")
    tcold = get_value(data, "tcold")
    thtec = get_value(data, "thtec")
    solar_radiation = get_value(data, "solar_radiation")
    ambient_temperature = get_value(data, "ambient_temperature")
    humidity = get_value(data, "humidity")
    wind_speed = get_value(data, "wind_speed")
    pressure = get_value(data, "pressure")

    tdp = dew_point(ambient_temperature, humidity)

    day_of_year = (month - 1) * 30.44 + day
    doy_sin = math.sin(2 * math.pi * day_of_year / 365)
    doy_cos = math.cos(2 * math.pi * day_of_year / 365)

    values = {
        "Month": month,
        "Day": day,
        "Solar Hours": solar_hours,
        "Tevp": tevp,
        "Thot": thot,
        "Tcold": tcold,
        "Th,TEC": thtec,
        "Solar Radiation, I\n(W/m2)": solar_radiation,
        "Ambient Temperature, Ta (oC)": ambient_temperature,
        "Humidity, RH\n(%)": humidity,
        "Wind Speed (m/s)": wind_speed,
        "Absolute pressure \n(mm of Hg)": pressure,
        "Dew point Temperature (oC)": tdp,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos
    }

    return pd.DataFrame([[values.get(feature, np.nan) for feature in FEATURES]], columns=FEATURES), tdp

@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory("../frontend", path)

@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model": "XGBoost",
        "target": TARGET,
        "features": FEATURES
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received"}), 400

        X, dew_point_value = build_features(data)

        prediction = float(model.predict(X)[0])

        if not math.isfinite(prediction):
            return jsonify({"error": "Model returned an invalid prediction"}), 500

        prediction = max(0, prediction)

        daily_yield = prediction * 24

        return jsonify({
            "success": True,
            "prediction": round(prediction, 4),
            "yield_5min": round(prediction, 2),
            "daily_yield": round(daily_yield, 2),
            "dew_point": round(dew_point_value, 1),
            "target": TARGET
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("===================================")
    print("AWG XGBoost Backend")
    print("===================================")
    print("Model loaded successfully")
    print("Target:", TARGET)
    print("Features:", len(FEATURES))
    print("Server: http://127.0.0.1:5000")
    print("===================================")

    app.run(host="0.0.0.0", port=5000, debug=True)