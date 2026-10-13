import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
import math

app = Flask(__name__)
BASE = Path(__file__).parent
model = joblib.load(BASE / "solar_xgb_model.pkl")


@app.route("/")
def home():
    return send_from_directory(BASE, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hour = float(data["hour"])
    month = float(data["month"])
    solar = float(data["solar_radiation"])
    temp = float(data["temperature"])
    cloud = float(data["cloud_cover"])

    features = pd.DataFrame([{
        "hour": hour,
        "month": month,
        "solar_radiation": solar,
        "temperature": temp,
        "cloud_cover": cloud
    }])

    prediction = model.predict(features)[0]
    power = float(prediction)

    radiation_factor = solar / 850
    power *= radiation_factor

    print("Solar radiation received:", solar)
    print("XGBoost prediction:", float(prediction))
    print("Adjusted power:", power)

    return jsonify({"power": max(0, power)})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
