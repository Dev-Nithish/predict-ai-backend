# train_pressure_seq5.py
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# 1) LOAD / SIMULATE DATA
# -------------------------
# If you have real sensor CSV, load it here. Otherwise simulated data:
time_steps = 5000
base = np.linspace(2, 9.5, time_steps)                    # smooth trend 0..10-ish
noise = np.random.normal(0, 0.25, time_steps)             # small noise
pressure = base + noise
pressure = np.clip(pressure, 0, 10).reshape(-1, 1)

# -------------------------
# 2) SCALE
# -------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(pressure)
joblib.dump(scaler, "pressure_scaler.pkl")
print("Saved scaler: pressure_scaler.pkl")

# -------------------------
# 3) CREATE SEQUENCES (SEQ_LEN=5)
# -------------------------
SEQ_LEN = 5
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN])   # predict immediate next (6th)
X = np.array(X)  # shape (N, 5, 1)
y = np.array(y)  # shape (N, 1)
print("Prepared dataset:", X.shape, y.shape)

# shuffle (optional)
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# -------------------------
# 4) MODEL
# -------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# -------------------------
# 5) TRAIN
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]
history = model.fit(
    X, y,
    epochs=60,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# 6) SAVE
# -------------------------
MODEL_NAME = "pressure_model.h5"   # keep stable name (or pressure_model_new.h5)
model.save(MODEL_NAME, include_optimizer=False)
print(f"Saved model: {MODEL_NAME}")
