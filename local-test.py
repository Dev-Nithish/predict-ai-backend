import numpy as np, joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("pressure_scaler.pkl")
model = load_model("pressure_model_new.h5", compile=False)

# create a smooth 20-step window near 8.0
window = (np.linspace(7.5, 8.5, 20) + np.random.normal(0,0.1,20)).reshape(-1,1)
print("window first/last:", window[0][0], window[-1][0])
scaled = scaler.transform(window)
pred_scaled = model.predict(scaled.reshape(1,20,1), verbose=0)[0][0]
pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
print("pred_scaled:", pred_scaled, "pred_real:", pred_real)
