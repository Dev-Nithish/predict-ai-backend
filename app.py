from flask import Flask, jsonify
import threading
import time
import random
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------
# 🔹 Firebase & Model Setup
# --------------------------------------------------------------------
cred = credentials.Certificate("serviceAccountKey.json")  # Your Firebase key file
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://predict-ai-91ed3-default-rtdb.firebaseio.com/"
})

# ✅ Load trained models
pressure_model = load_model("pressure_sensor_model.h5", compile=False)
limit_model = load_model("limit_switch_model.h5", compile=False)

# ✅ Global state
sensor_history = {}
is_streaming = False

# --------------------------------------------------------------------
# 🔹 Streaming Logic (runs in background thread)
# --------------------------------------------------------------------
def stream_data():
    global is_streaming
    ref = db.reference("sensors")

    while is_streaming:
        timestamp = time.strftime("%H:%M:%S")

        # Generate random sensor values
        sensors = {
            "Pressure_Sensor_01_PS01": random.uniform(30, 80),
            "Pressure_Sensor_02_PS02": random.uniform(25, 90),
            "Limit_Switch_01_LS01": random.uniform(0, 1)
        }

        data_to_send = {}
        for sid, val in sensors.items():
            pred = predict_next(sid, val)
            data_to_send[sid] = {
                "pressure": val,
                "aiPrediction": pred,
                "timestamp": timestamp
            }

        ref.update(data_to_send)
        print(f"✅ Sent data @ {timestamp}")
        time.sleep(3)  # every 3 seconds

# --------------------------------------------------------------------
# 🔹 Prediction Helper
# --------------------------------------------------------------------
def predict_next(sensor_id, current_value):
    try:
        if sensor_id not in sensor_history:
            sensor_history[sensor_id] = []

        sensor_history[sensor_id].append(current_value)
        if len(sensor_history[sensor_id]) < 20:
            return current_value

        # Keep last 20 readings
        sensor_history[sensor_id] = sensor_history[sensor_id][-20:]
        X_input = np.array(sensor_history[sensor_id]).reshape(1, 20, 1)

        # Predict using correct model
        if "Pressure_Sensor" in sensor_id:
            pred = pressure_model.predict(X_input, verbose=0)[0][0]
        elif "Limit_Switch" in sensor_id:
            pred = limit_model.predict(X_input, verbose=0)[0][0]
        else:
            pred = current_value

        return float(pred)
    except Exception as e:
        print(f"⚠️ Prediction error for {sensor_id}: {e}")
        return current_value

# --------------------------------------------------------------------
# 🔹 Flask API Endpoints
# --------------------------------------------------------------------
app = Flask(__name__)

@app.route("/start", methods=["GET"])
def start_stream():
    global is_streaming
    if not is_streaming:
        is_streaming = True
        threading.Thread(target=stream_data, daemon=True).start()
        return jsonify({"status": "started", "message": "Streaming data to Firebase"})
    else:
        return jsonify({"status": "running", "message": "Already streaming"})

@app.route("/stop", methods=["GET"])
def stop_stream():
    global is_streaming
    is_streaming = False
    return jsonify({"status": "stopped", "message": "Streaming stopped"})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Predict.AI backend running ✅"})

# --------------------------------------------------------------------
# 🔹 Run Flask Server
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
