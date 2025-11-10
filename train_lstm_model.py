# 📄 File: train_lstm_model.py
# 📍 Location: E:\Predict.Ai\ai-training\train_lstm_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# 1. Load or simulate dataset
# ----------------------------
try:
    df = pd.read_csv("sensor_data.csv")
    print("✅ Loaded existing dataset (sensor_data.csv)")
except FileNotFoundError:
    print("⚠️ sensor_data.csv not found — generating dummy dataset...")
    # Simulate example sensor data (1000 time steps, 3 features)
    time = np.arange(1000)
    data = {
        "temperature": np.sin(time / 50) + np.random.normal(0, 0.1, 1000),
        "pressure": np.cos(time / 80) + np.random.normal(0, 0.1, 1000),
        "vibration": np.random.normal(0, 0.3, 1000),
        "label": (np.random.rand(1000) > 0.8).astype(int),  # binary classification
    }
    df = pd.DataFrame(data)
    df.to_csv("sensor_data.csv", index=False)
    print("✅ Dummy dataset created and saved as sensor_data.csv")

# ----------------------------
# 2. Prepare the data
# ----------------------------
SEQ_LEN = 20  # number of timesteps per sequence
features = ["temperature", "pressure", "vibration"]
target = "label"

X, y = [], []
for i in range(len(df) - SEQ_LEN):
    X.append(df[features].iloc[i:i+SEQ_LEN].values)
    y.append(df[target].iloc[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)
print(f"✅ Data prepared: X shape = {X.shape}, y shape = {y.shape}")

# ----------------------------
# 3. Define the LSTM model
# ----------------------------
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# 4. Train the model
# ----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# 5. Save the model (Updated for TF 2.15 compatibility)
# ----------------------------
MODEL_NAME = "pressure_model_new.h5"

# ✅ Ensure we save in the new format, without old optimizer metadata
model.save(MODEL_NAME, include_optimizer=False, save_format="h5")

print(f"✅ Model re-saved cleanly as {MODEL_NAME}")
print("You can now safely load it with TensorFlow 2.15+ (no legacy errors).")
