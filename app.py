from flask import Flask, jsonify
import threading
import time
import random
import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import os
import json
import h5py
import tensorflow as tf
import traceback
from tensorflow.keras.models import model_from_json
from flask_cors import CORS  # ✅ import first

# --------------------------------------------------------------------
# 🔹 Initialize Flask App
# --------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS immediately after creating the app

# --------------------------------------------------------------------
# 🔹 Global state & Model Placeholders
# --------------------------------------------------------------------
is_streaming = False
sensor_history = {}
pressure_model = None
limit_model = None


# --------------------------------------------------------------------
# 🩵 Safe Model Loader — handles legacy Keras 2.x models
# --------------------------------------------------------------------
def safe_load_model(path):
    """Safely loads a Keras model, attempting legacy deserialization if direct load fails."""
    try:
        print(f"⏳ Loading model: {path}")
        # Setting compile=False is often necessary when loading models without the original training environment
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"⚠️ Direct load failed: {e}")
        print("🔁 Attempting deep legacy deserialization...")

        try:
            with h5py.File(path, "r") as f:
                model_json = f.attrs.get("model_config")
                if model_json is None:
                    raise Exception("No model_config found in H5 attributes.")

                # Decode from bytes to str if needed
                if isinstance(model_json, bytes):
                    model_json = model_json.decode("utf-8")

                model_config = json.loads(model_json)

            # 🧹 Clean out incompatible or missing keys recursively
            def clean_dict(d):
                if isinstance(d, dict):
                    clean = {}
                    for k, v in d.items():
                        # Keys like 'batch_shape' can cause conflicts when loading from config
                        if k in ["batch_shape", "class_name", "build_input_shape"]:
                            continue
                        clean[k] = clean_dict(v)
                    return clean
                elif isinstance(d, list):
                    return [clean_dict(v) for v in d]
                else:
                    return d

            cleaned_config = clean_dict(model_config)

            # 🧠 Ensure structure is valid for model_from_config (Keras 2 legacy)
            if "class_name" not in cleaned_config:
                cleaned_config = {"class_name": "Sequential", "config": cleaned_config.get("config", cleaned_config)}

            model = tf.keras.models.model_from_config(cleaned_config)
            print(f"✅ Successfully reconstructed legacy model: {path}")
            return model
        except Exception as inner_e:
            print(f"❌ Legacy deserialization still failed: {inner_e}")
            raise inner_e


# --------------------------------------------------------------------
# 🔹 Firebase Initialization (With Private Key Fix)
# --------------------------------------------------------------------
firebase_config_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if firebase_config_json:
    print("✅ Using credentials from environment variable")
    firebase_config = json.loads(firebase_config_json)
    
    # === CRITICAL FIX FOR PEM ERROR (InvalidData(InvalidByte(0, 92))) ===
    # Replace escaped newline characters (\\n) with actual newlines (\n) 
    # to correctly load the PEM-formatted private key.
    if 'private_key' in firebase_config:
        firebase_config['private_key'] = firebase_config['private_key'].replace('\\n', '\n')
    # ===================================================================

    cred = credentials.Certificate(firebase_config)
else:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not found. Using local file.")
    # Fallback for local development
    cred = credentials.Certificate("serviceAccountKey.json")

try:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://predict-ai-91ed3-default-rtdb.firebaseio.com/"
    })
except ValueError as e:
    print(f"❌ Firebase Initialization Failed: {e}")
    # Re-raise the error to stop the application if Firebase is critical
    # This prevents the stream from running without proper authentication
    raise

# --------------------------------------------------------------------
# 🔹 Model Loading
# --------------------------------------------------------------------
try:
    # Load models into the global variables defined at the top
    pressure_model = safe_load_model("pressure_model.h5")
    limit_model = safe_load_model("limit_model.h5")
except Exception as e:
    # This is a non-fatal warning, but predictions will fail
    print(f"❌ WARNING: Failed to load one or both prediction models. Check model files: {e}")

# --------------------------------------------------------------------
# 🔹 Prediction Helper
# --------------------------------------------------------------------
def predict_next(sensor_id, current_value):
    """
    Appends current value to history, prepares input, and predicts the next value
    using the appropriate loaded model.
    """
    try:
        if sensor_id not in sensor_history:
            sensor_history[sensor_id] = []

        sensor_history[sensor_id].append(current_value)
        
        # We need a sequence length of 20 for prediction
        if len(sensor_history[sensor_id]) < 20:
            # Not enough data yet, return current value as prediction
            return current_value

        # Keep only the last 20 elements
        sensor_history[sensor_id] = sensor_history[sensor_id][-20:]
        # Reshape data for the LSTM model: (1 sample, 20 timesteps, 1 feature)
        X_input = np.array(sensor_history[sensor_id]).reshape(1, 20, 1)

        pred = current_value # Default prediction

        if "Pressure_Sensor" in sensor_id and pressure_model:
            # Use the loaded pressure model
            pred = pressure_model.predict(X_input, verbose=0)[0][0]
        elif "Limit_Switch" in sensor_id and limit_model:
            # Use the loaded limit model
            pred = limit_model.predict(X_input, verbose=0)[0][0]

        # Convert numpy float to native Python float for JSON serialization
        return float(pred)
    except Exception as e:
        print(f"⚠️ Prediction error for {sensor_id}: {e}")
        # Return current value on any prediction failure
        return current_value

# --------------------------------------------------------------------
# 🔹 Streaming Logic (runs in background thread)
# --------------------------------------------------------------------
def get_status(value: float) -> str:
    """Classify pressure value into status ranges."""
    if value < 4:
        return "DANGER"
    elif value <= 8:
        return "NORMAL"
    else:
        return "CRITICAL"


def stream_data():
    global is_streaming
    ref = db.reference("sensors")
    print("🔁 Stream loop started...")

    while is_streaming:
        try:
            timestamp = time.strftime("%H:%M:%S")

            # 🔧 Generate realistic 0–10 range readings
            base = 7 + math.sin(time.time() / 5) * 3 + random.uniform(-0.3, 0.3)
            sensors = {
                "Pressure_Sensor_01_PS01": round(max(0, min(10, base + random.uniform(-0.5, 0.5))), 2),
                "Pressure_Sensor_02_PS02": round(max(0, min(10, base + random.uniform(-0.4, 0.4))), 2),
                "Limit_Switch_01_LS01": round(random.choice([0.0, 1.0]) + random.uniform(-0.05, 0.05), 2)
            }

            data_to_send = {}
            for sid, val in sensors.items():
                pred = predict_next(sid, val)
                status = get_status(val)
                data_to_send[sid] = {
                    "actual": val,
                    "predicted": pred,      # ✅ unified key name
                    "status": status,
                    "active": True,         # ✅ helps Angular detect deactivation
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
    port = int(os.environ.get("PORT", 5000))
    # Flask is set to run on 0.0.0.0 to be accessible on the server's network
    app.run(host="0.0.0.0", port=port)



import os
import json
from firebase_admin import credentials, initialize_app

# 1. Retrieve the credentials string from the environment variable
# Assuming your variable name is FIREBASE_CREDENTIALS_JSON
firebase_config_str = os.environ.get('FIREBASE_CREDENTIALS_JSON')

if firebase_config_str:
    # 2. Parse the string into a Python dictionary
    firebase_config = json.loads(firebase_config_str)
    
    # 3. CRITICAL FIX: Replace the literal escaped newlines ('\\n') 
    # with actual newline characters ('\n') in the private_key.
    firebase_config['private_key'] = firebase_config['private_key'].replace('\\n', '\n')

    # 4. Initialize the certificate with the fixed dictionary
    cred = credentials.Certificate(firebase_config)
    initialize_app(cred)
    
    print("✅ Successfully initialized Firebase Admin SDK.")
else:
    # Handle the case where the environment variable is missing
    print("❌ FIREBASE_CREDENTIALS_JSON environment variable not found.")
    # You might want to raise an exception or exit here
    
# Your main application code continues here...
# ...
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
