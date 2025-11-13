# ============================================================
#  send_data_to_firebase.py
#  Sends exactly 50 readings + LSTM predictions (at 5,10,15,...)
#  Adds alert counter (alert_count) when DANGER/CRITICAL occurs
#  Includes safe legacy loader for older Keras models
# ============================================================

import time
import math
import random
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import joblib
import tensorflow as tf
import h5py, json

# ------------------------------------------------------------
#  Firebase setup
# ------------------------------------------------------------
cred = credentials.Certificate(r"E:\Predict.Ai\ai-training\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://predict-ai-91ed3-default-rtdb.firebaseio.com/'
})

# ------------------------------------------------------------
#  Safe Legacy Model Loader
# ------------------------------------------------------------
def safe_load_model(path):
    import h5py, json, tensorflow as tf
    from tensorflow.keras.models import Sequential, model_from_json
    from tensorflow.keras.utils import get_custom_objects

    try:
        print(f" Loading model: {path}")
        return tf.keras.models.load_model(path, compile=False)

    except Exception as e:
        print(f" Direct load failed: {e}")
        print(" Attempting legacy deserialization...")

        with h5py.File(path, "r") as f:
            model_json = f.attrs.get("model_config")
            if model_json is None:
                raise e

            if isinstance(model_json, bytes):
                model_json = model_json.decode("utf-8")

            model_config = json.loads(model_json)

            # Clean invalid keys like time_major
            def clean_dict(d):
                if isinstance(d, dict):
                    return {
                        k: clean_dict(v)
                        for k, v in d.items()
                        if k != "time_major"
                    }
                elif isinstance(d, list):
                    return [clean_dict(v) for v in d]
                else:
                    return d

            cleaned_config = clean_dict(model_config)

            if "class_name" not in cleaned_config:
                cleaned_config = {
                    "class_name": "Sequential",
                    "config": cleaned_config.get("config", cleaned_config)
                }

            get_custom_objects()['Sequential'] = Sequential

            try:
                model = model_from_json(json.dumps(cleaned_config))
                print(f" Legacy model successfully deserialized from {path}")
                return model
            except Exception as inner_e:
                print(f" Legacy deserialization failed: {inner_e}")
                raise inner_e

# ------------------------------------------------------------
#  Load trained models & scalers safely
# ------------------------------------------------------------
pressure_model = safe_load_model("pressure_model_new.h5")
limit_model = safe_load_model("limit_switch_model_new.h5")

pressure_scaler = joblib.load("pressure_scaler.pkl")
limit_scaler = joblib.load("limit_switch_scaler.pkl")

# ------------------------------------------------------------
#  YOUR REQUESTED UPDATE
# ------------------------------------------------------------
SEQ_LEN = 5
PRED_INTERVAL = 5
TOTAL_READINGS = 50
sensor_history = {}

# ------------------------------------------------------------
#  Status helper
# ------------------------------------------------------------
def get_status(value: float) -> str:
    if value < 4:
        return "DANGER"
    elif value > 10:
        return "CRITICAL"
    else:
        return "NORMAL"

# ------------------------------------------------------------
#  UPDATED predict_next() — EXACTLY AS YOU REQUESTED
# ------------------------------------------------------------
def predict_next(sensor_id: str):
    history = sensor_history.get(sensor_id, [])

    if len(history) < SEQ_LEN:
        return None

    # Predict only every 5th reading
    if len(history) % PRED_INTERVAL != 0:
        return None

    arr = np.array(history[-SEQ_LEN:]).reshape(-1, 1)

    # Pressure sensor prediction
    if "Pressure_Sensor" in sensor_id and pressure_model:
        scaled = pressure_scaler.transform(arr)
        X_input = scaled.reshape(1, SEQ_LEN, 1)
        pred_scaled = pressure_model.predict(X_input, verbose=0)[0][0]
        pred_real = pressure_scaler.inverse_transform([[pred_scaled]])[0][0]
        return float(max(0.0, min(10.0, pred_real)))

    # Limit switch prediction
    elif "Limit_Switch" in sensor_id and limit_model:
        scaled = limit_scaler.transform(arr)
        X_input = scaled.reshape(1, SEQ_LEN, 1)
        pred_scaled = limit_model.predict(X_input, verbose=0)[0][0]
        pred_real = limit_scaler.inverse_transform([[pred_scaled]])[0][0]
        return float(max(0.0, min(1.0, pred_real)))

    return None

# ------------------------------------------------------------
#  Data simulation
# ------------------------------------------------------------
def simulate_sensor_data():
    t = time.time()
    base_pressure = 7 + math.sin(t / 5) * 3 + random.uniform(-0.3, 0.3)
    data = {}

    for i in range(1, 5):
        sensor_id = f"Pressure_Sensor_0{i}_PS0{i}"
        data[sensor_id] = round(max(0, min(15, base_pressure + random.uniform(-0.5, 0.5))), 2)

    for i in range(1, 5):
        sensor_id = f"Limit_Switch_0{i}_LS0{i}"
        val = random.choice([0.0, 1.0]) + random.uniform(-0.05, 0.05)
        data[sensor_id] = round(max(0, min(1, val)), 2)

    return data

# ------------------------------------------------------------
#  Send to Firebase
# ------------------------------------------------------------
def send_to_firebase(read_num):
    timestamp = time.strftime("%H:%M:%S")
    data = simulate_sensor_data()
    print(f"\n Reading {read_num}/{TOTAL_READINGS} @ {timestamp}")

    for sensor_id, value in data.items():
        if sensor_id not in sensor_history:
            sensor_history[sensor_id] = []
        sensor_history[sensor_id].append(value)
        sensor_history[sensor_id] = sensor_history[sensor_id][-SEQ_LEN:]

        predicted = predict_next(sensor_id)
        status = get_status(value)

        ref = db.reference(f"sensors/{sensor_id}")

        # Alert counter
        if "alert_counter" not in sensor_history:
            sensor_history["alert_counter"] = {}

        if sensor_id not in sensor_history["alert_counter"]:
            sensor_history["alert_counter"][sensor_id] = 0

        if status in ["CRITICAL", "DANGER"]:
            sensor_history["alert_counter"][sensor_id] += 1

        current_count = sensor_history["alert_counter"][sensor_id]

        ref.update({
            "actual": value,
            "predicted": predicted if predicted is not None else ref.child("predicted").get() or value,
            "status": status,
            "alert_count": current_count,
            "timestamp": timestamp
        })

        cfg_ref = db.reference(f"configuration/sensors/{sensor_id}")
        if not cfg_ref.get():
            cfg_ref.set({"active": True})

# ------------------------------------------------------------
#  Main loop
# ------------------------------------------------------------
if __name__ == "__main__":
    print(" Starting 50-reading data stream...\n")
    for i in range(1, TOTAL_READINGS + 1):
        send_to_firebase(i)
        time.sleep(1.5)
    print("\n Completed 50 readings successfully! Stopping program.")
