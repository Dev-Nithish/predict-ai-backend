from flask import Flask, jsonify
import threading
import time
import random
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
import os
import json
import h5py

# ✅ Global state
is_streaming = False
sensor_history = {}


# --------------------------------------------------------------------
# 🔹 Firebase & Model Setup
# --------------------------------------------------------------------
firebase_config_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if firebase_config_json:
    print("✅ Using credentials from environment variable")
    firebase_config = json.loads(firebase_config_json)
    cred = credentials.Certificate(firebase_config)
else:
    print("✅ Using local serviceAccountKey.json file")
    cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://predict-ai-91ed3-default-rtdb.firebaseio.com/"
})

# --------------------------------------------------------------------
# 🩵 Safe Model Loader — fixes “batch_shape” error
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# 🩵 Safe Model Loader — handles legacy Keras 2.x models
# --------------------------------------------------------------------
def safe_load_model(path):
    import tensorflow as tf
    import json, h5py

    try:
        print(f"⏳ Loading model: {path}")
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"⚠️ Direct load failed: {e}")
        print("🔁 Attempting deep legacy deserialization...")

        with h5py.File(path, "r") as f:
            model_json = f.attrs.get("model_config")
            if model_json is None:
                raise e

            # Decode from bytes to str if needed
            if isinstance(model_json, bytes):
                model_json = model_json.decode("utf-8")

            try:
                model_config = json.loads(model_json)
            except json.JSONDecodeError:
                print("❌ Invalid JSON in model config.")
                raise

        # 🧹 Clean out incompatible or missing keys recursively
        def clean_dict(d):
            if isinstance(d, dict):
                clean = {}
                for k, v in d.items():
                    if k in ["batch_shape", "class_name", "build_input_shape"]:
                        continue
                    clean[k] = clean_dict(v)
                return clean
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return d

        cleaned_config = clean_dict(model_config)

        # 🧠 Ensure structure is valid for model_from_config
        if "class_name" not in cleaned_config:
            cleaned_config = {"class_name": "Sequential", "config": cleaned_config.get("config", cleaned_config)}

        from tensorflow.keras.models import model_from_config
        try:
            model = model_from_config(cleaned_config)
            print(f"✅ Successfully reconstructed legacy model: {path}")
            return model
        except Exception as inner_e:
            print(f"❌ Legacy deserialization still failed: {inner_e}")
            raise inner_e


# --------------------------------------------------------------------
# 🔹 Streaming Logic (runs in background thread)
# --------------------------------------------------------------------
def stream_data():
    import traceback
    global is_streaming
    ref = db.reference("sensors")
    print("🔁 Stream loop started...")

    while is_streaming:
        try:
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

            # ✅ Safe Firebase update
            try:
                ref.update(data_to_send)
                print(f"✅ Data sent @ {timestamp}: {data_to_send}")
            except Exception as fb_err:
                print(f"❌ Firebase update failed: {fb_err}")
                traceback.print_exc()

            time.sleep(3)  # every 3 seconds

        except Exception as e:
            print(f"⚠️ Error inside stream_data loop: {e}")
            traceback.print_exc()
            time.sleep(3)  # small delay before retry


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

        sensor_history[sensor_id] = sensor_history[sensor_id][-20:]
        X_input = np.array(sensor_history[sensor_id]).reshape(1, 20, 1)

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
# 🔹 Run Flask Server (Render-compatible)
# --------------------------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
