import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- Generate realistic ON/OFF toggle data ---
time_steps = 2000
signal = []
state = 0
for _ in range(time_steps):
    if np.random.rand() < 1/50:  # occasional toggle
        state = 1 - state
    signal.append(state + np.random.normal(0, 0.05))  # add small noise

limit_switch = np.clip(signal, 0, 1).reshape(-1, 1)

# --- Normalize ---
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(limit_switch)
joblib.dump(scaler, "limit_switch_scaler.pkl")
print("✅ Saved scaler: limit_switch_scaler.pkl")

# --- Prepare sequences ---
SEQ_LEN = 20
X, y = [], []
for i in range(len(scaled_data) - SEQ_LEN):
    X.append(scaled_data[i:i+SEQ_LEN])
    y.append(scaled_data[i+SEQ_LEN])
X, y = np.array(X), np.array(y)
print(f"✅ Dataset ready: X={X.shape}, y={y.shape}")

# --- Build LSTM ---
model = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # keeps output between 0 and 1
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# --- Train ---
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# --- Save model ---
model.save("limit_switch_model_new.h5")
print("✅ Saved LSTM model: limit_switch_model_new.h5")
