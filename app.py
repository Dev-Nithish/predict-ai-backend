from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import time
import os
import signal
import json
import traceback
import threading

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ML / Data
import tensorflow as tf
import numpy as np
import h5py

# =====================================================
# 🔧 Utility: Feed Output Logger
# =====================================================
def log_feed_output(process):
    """Continuously read and print output from the subprocess."""
    for line in iter(process.stdout.readline, ''):
        if isinstance(line, bytes):
            line = line.decode(errors="ignore")
        print(f"[FEED] {line.strip()}")
    process.stdout.close()

# =====================================================
# 🔹 Initialize Flask App
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# 🔹 Firebase Initialization
# =====================================================
firebase_config_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")

if firebase_config_json:
    print("✅ Using credentials from environment variable")
    firebase_config = json.loads(firebase_config_json)

    if 'private_key' in firebase_config:
        firebase_config['private_key'] = firebase_config['private_key'].replace('\\n', '\n')

    cred = credentials.Certificate(firebase_config)
else:
    print("⚠️ Using local credentials: serviceAccountKey.json")
    cred = credentials.Certificate("serviceAccountKey.json")

try:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://predict-ai-91ed3-default-rtdb.firebaseio.com/"
    })
except ValueError:
    pass  # Firebase already initialized

# =====================================================
# 🔹 Global State
# =====================================================
feed_process = None

# =====================================================
# 🧠 Safe Model Loader
# =====================================================
def safe_load_model(path):
    try:
        print(f"⏳ Loading model: {path}")
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")
        return None

# =====================================================
# 🔹 Optional: Preload Models
# =====================================================
try:
    pressure_model = safe_load_model("pressure_sensor_model_new.h5")
    limit_model = safe_load_model("limit_switch_model_new.h5")
except Exception as e:
    print(f"⚠️ Model preload failed: {e}")

# =====================================================
# 🟢 START FEED
# =====================================================
@app.route("/start_feed", methods=["POST"])
def start_feed():
    global feed_process

    if feed_process and feed_process.poll() is None:
        print("⚠️ Feed already running")
        return jsonify({"status": "already_running"}), 200

    try:
        feed_process = subprocess.Popen(
            ["python", "send_data_to_firebase.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        threading.Thread(target=log_feed_output, args=(feed_process,), daemon=True).start()
        print("🚀 Started send_data_to_firebase.py")

        return jsonify({"status": "started"}), 200
    except Exception as e:
        print(f"❌ Failed to start feed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =====================================================
# 🔴 STOP FEED
# =====================================================
@app.route("/stop_feed", methods=["POST"])
def stop_feed():
    global feed_process

    if not feed_process or feed_process.poll() is not None:
        print("⚠️ No feed currently running")
        return jsonify({"status": "not_running"}), 200

    try:
        feed_process.terminate()
        time.sleep(1)

        if feed_process.poll() is None:
            os.kill(feed_process.pid, signal.SIGTERM)

        print("🛑 Feed stopped.")
        feed_process = None
        return jsonify({"status": "stopped"}), 200
    except Exception as e:
        print(f"❌ Failed to stop feed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =====================================================
# 🔍 FEED STATUS CHECK
# =====================================================
@app.route("/feed_status", methods=["GET"])
def feed_status():
    global feed_process
    running = feed_process is not None and feed_process.poll() is None
    return jsonify({"running": running})

# =====================================================
# 🏠 ROOT ENDPOINT
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Predict.AI backend running ✅"})

# =====================================================
# 🚀 Run Flask App (Render-Compatible)
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
