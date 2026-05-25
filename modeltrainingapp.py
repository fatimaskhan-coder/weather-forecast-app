import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
df = pd.read_csv("C:/Users/Hp/OneDrive - Higher Education Commission/DailyDelhiClimateTrain.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"\nDataset shape: {df.shape}")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")
df.info()
features = ["meantemp", "humidity", "wind_speed", "meanpressure"]
df = df[features].dropna()
#  Scale Features 
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
 
# Save scaler params for the web app
np.save("scaler_min.npy",   scaler.data_min_)
np.save("scaler_scale.npy", scaler.data_range_)
# Sliding Window Sequences 
SEQ_LEN = 30  # use last 30 days to predict next day
 
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])      # shape: (30, 4)
        y.append(data[i + seq_len][0])        # next day meantemp (scaled)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
 
X, y = make_sequences(data_scaled, SEQ_LEN)
print(f"\nX shape: {X.shape}")   # (samples, 30, 4)
print(f"y shape: {y.shape}")     # (samples,)
 
# Train / Validation split (80/20)
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
 
print(f"\nTrain samples: {len(X_train)}, Val samples: {len(X_val)}")
# Build Stacked LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])
 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)
 
model.summary()
# Callbacks ──────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]
#  Train 
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1)
#Evaluate 
def inverse_temp(scaled_vals):
    """Inverse transform only the temperature column."""
    dummy = np.zeros((len(scaled_vals), len(features)))
    dummy[:, 0] = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]
 
preds_scaled = model.predict(X_val).flatten()
preds_real   = inverse_temp(preds_scaled)
true_real    = inverse_temp(y_val)
 
mae  = mean_absolute_error(true_real, preds_real)
rmse = np.sqrt(mean_squared_error(true_real, preds_real))
 
print(f"\n{'='*40}")
print(f"  MAE  : {mae:.2f} °C")
print(f"  RMSE : {rmse:.2f} °C")
print(f"{'='*40}\n")
model.save("lstm_weather.keras")
 
results = {
    "mae":  round(float(mae),  2),
    "rmse": round(float(rmse), 2),
    "history": {
        "train_loss": [round(float(v), 6) for v in history.history["loss"]],
        "val_loss":   [round(float(v), 6) for v in history.history["val_loss"]],
        "train_mae":  [round(float(v), 6) for v in history.history["mae"]],
        "val_mae":    [round(float(v), 6) for v in history.history["val_mae"]],
    },
    "sample_preds": [round(float(v), 2) for v in preds_real[:60]],
    "sample_true":  [round(float(v), 2) for v in true_real[:60]],
}
 
with open("results.json", "w") as f:
    json.dump(results, f)
 
print("✅ Model saved → lstm_weather.keras")
print("✅ Results saved → results.json")