# 🌤️ Weather Temperature Forecasting using LSTM (Deep Learning)

A end-to-end deep learning project that predicts the next day's temperature using a Stacked LSTM neural network trained on real Delhi climate data.

---

## 📌 Problem Statement

Weather forecasting is one of the most impactful applications of machine learning. Traditional forecasting systems rely on complex physics-based models that require enormous computational resources. In this project, I explored whether a deep learning model — specifically a **Stacked LSTM (Long Short-Term Memory)** network — can learn temporal patterns from historical weather data and accurately predict the next day's mean temperature.

Given the last **30 days** of weather readings (temperature, humidity, wind speed, atmospheric pressure), the model predicts **tomorrow's mean temperature** in °C.

---

## 📂 Dataset

**Source:** [Daily Climate Time Series Data — Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

**Location:** Delhi, India | **Duration:** 2013–2017

| Feature | Description |
|---|---|
| `meantemp` | Mean temperature of the day (°C) — **target variable** |
| `humidity` | Mean humidity (%) |
| `wind_speed` | Mean wind speed (km/h) |
| `meanpressure` | Mean atmospheric pressure (hPa) |

---

## 🧠 Deep Learning Concepts Used

### 1. Sliding Window Sequences
Raw time series data can't be directly fed into a neural network. We use a **sliding window** approach: for each day, we take the previous 30 days as input (X) and the next day's temperature as the label (y).

```
Day 1–30  →  predict Day 31
Day 2–31  →  predict Day 32
Day 3–32  →  predict Day 33
...
```

This transforms the dataset into supervised learning sequences of shape `(samples, 30, 4)`.

### 2. MinMax Scaling
All features are scaled to the range [0, 1] using `MinMaxScaler` before training. This prevents features with large values (like pressure ~1013 hPa) from dominating features with small values (like wind speed ~10 km/h). After prediction, we **inverse transform** back to real °C values.

### 3. Stacked LSTM Architecture
LSTMs are a type of Recurrent Neural Network (RNN) designed to learn **long-term dependencies** in sequential data. Unlike standard RNNs, LSTMs have a **cell state** and three gates (input, forget, output) that control what information to keep or discard over time.

We stack **two LSTM layers**:
- Layer 1 learns low-level temporal patterns (day-to-day changes)
- Layer 2 learns higher-level patterns (weekly/seasonal trends)

### 4. Dropout Regularization
After each LSTM layer, a **Dropout(0.2)** layer randomly switches off 20% of neurons during training. This prevents the model from memorizing the training data (overfitting) and forces it to learn more generalized patterns.

### 5. Callbacks
- **EarlyStopping** — stops training when validation loss stops improving (patience=10), and restores the best weights
- **ReduceLROnPlateau** — cuts the learning rate in half when val loss plateaus (patience=5), helping the model fine-tune

---

## 🏗️ Model Architecture

```
Input:  (30 timesteps × 4 features)
        ↓
LSTM Layer 1  —  64 units, return_sequences=True
        ↓
Dropout (0.2)
        ↓
LSTM Layer 2  —  64 units, return_sequences=False
        ↓
Dropout (0.2)
        ↓
Dense (32 units, ReLU activation)
        ↓
Dense (1 unit)  →  Predicted Temperature (°C)

Loss:      Mean Squared Error (MSE)
Optimizer: Adam (lr=0.001)
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn pandas numpy flask
```

### 2. Download the dataset
Get `DailyDelhiClimateTrain.csv` from [Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) and place it in the project folder.

### 3. Train the model
```bash
python train_model.py
```
This trains for up to 100 epochs with early stopping and saves:
- `lstm_weather.keras` — trained model weights
- `scaler_min.npy` / `scaler_scale.npy` — scaler parameters
- `results.json` — training history + validation predictions

### 4. Launch the web app
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## 🌐 Web App

Built with **Flask**, the app lets you:
- Enter today's temperature, humidity, wind speed and pressure
- Get an instant prediction of tomorrow's temperature
- Visualize how closely the model's predictions match actual values on the validation set

---

## 📁 Project Structure

```
weather_lstm/
├── DailyDelhiClimateTrain.csv   # Raw dataset (download from Kaggle)
├── train_model.py               # Data preprocessing + LSTM training
├── app.py                       # Flask web application
├── lstm_weather.keras           # Saved model (after training)
├── scaler_min.npy               # Scaler min values
├── scaler_scale.npy             # Scaler range values
├── results.json                 # Training history + predictions
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
