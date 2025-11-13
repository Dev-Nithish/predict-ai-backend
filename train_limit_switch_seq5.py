# train_limit_seq5.py
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
# 1) SIMULATE ON/OFF SIGNAL
# -------------------------
time_steps = 4000
signal = []
state = 0
for _ in range(time_steps):
    if np.random.rand() < 1/50:
        state = 1 - state
    signal.append(state + np.random.normal(0, 0.03))  # small jitter
limit_switch = np.clip(signal, 0, 1).reshape(-1,1)

# -------------------------
# 2) SCALE
# -------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(limit_switch)
joblib.dump(scaler, "limit_switch_scaler.pkl")
print("Saved scaler: limit_switch_scaler.pkl")

# -------------------------
# 3) SEQUENCES (SEQ_LEN=5)
# -------------------------
SEQ_LEN = 5
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN])
X = np.array(X)
y = np.array(y)
print("Prepared dataset:", X.shape, y.shape)

# shuffle
perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# -------------------------
# 4) MODEL
# -------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, 1)),
    LSTM(32, return_sequences=False),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    epochs=40,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------
# 6) SAVE
# -------------------------
MODEL_NAME = "limit_switch_model.h5"  # or keep your existing name
model.save(MODEL_NAME, include_optimizer=False)
print(f"Saved model: {MODEL_NAME}")
