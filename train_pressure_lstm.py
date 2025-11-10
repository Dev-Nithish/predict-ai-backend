import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Simulated pressure data (smooth trend + noise) ---
time_steps = 3000
base_pressure = np.linspace(2, 10, time_steps)  # gradual increase
noise = np.random.normal(0, 0.3, time_steps)
pressure = base_pressure + noise
pressure = np.clip(pressure, 0, 12).reshape(-1, 1)

# --- Normalize the data ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(pressure)
joblib.dump(scaler, "pressure_scaler.pkl")
print("✅ Saved scaler: pressure_scaler.pkl")

# --- Prepare sequences ---
SEQ_LEN = 20
X, y = [], []
for i in range(len(scaled_data) - SEQ_LEN):
    X.append(scaled_data[i:i + SEQ_LEN])
    y.append(scaled_data[i + SEQ_LEN])
X, y = np.array(X), np.array(y)
print(f"✅ Pressure dataset ready: X={X.shape}, y={y.shape}")

# --- Build LSTM model ---
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, verbose=1)

# --- Save model ---
model.save("pressure_model_new.h5")
print("✅ Pressure LSTM saved as pressure_model_new.h5")
