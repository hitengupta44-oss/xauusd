```python
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta
import pytz

# ===== Flask keep-alive (Render) =====
from flask import Flask
import threading

app = Flask(__name__)

@app.route("/")
def home():
    return "XAUUSD Producer running"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask).start()

# ================= CONFIG =================

BACKEND_URL = "https://xauusd-klue.onrender.com/update"

API_KEY = os.getenv("TWELVEDATA_API_KEY")

SYMBOL = "XAU/USD"
INTERVAL = "5min"

LOOKBACK = 60
PRED_MINUTES = 6
RETRAIN_INTERVAL = 1800

IST = pytz.timezone("Asia/Kolkata")

print("Producer started — XAUUSD 5m")

model = None
scaler = MinMaxScaler()
last_candle_time = None
last_train_time = None


# ================= MODEL =================

def build_model(n_features):
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ================= FETCH DATA =================

def fetch_data():

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 500,
        "apikey": API_KEY
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "values" not in data:
        print("API Error:", data)
        return None

    df = pd.DataFrame(data["values"])

    df["time"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert(IST)

    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float
    })

    df["volume"] = 1

    df = df.sort_values("time")

    return df[["time","open","high","low","close","volume"]]


# ================= MAIN LOOP =================

while True:

    try:

        df = fetch_data()

        if df is None or len(df) < 100:
            print("Data issue")
            time.sleep(60)
            continue

        current_time = df.iloc[-1]["time"]

        if last_candle_time == current_time:
            print("Waiting for new candle...")
            time.sleep(60)
            continue

        last_candle_time = current_time
        print("New candle:", current_time)

        # ===== Indicators =====
        df["EMA20"] = df["close"].ewm(span=20).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ===== Sequences =====
        X, y = [], []

        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i-LOOKBACK:i])
            y.append(scaled[i,0])

        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            time.sleep(60)
            continue

        # ===== Train =====
        if model is None:
            model = build_model(len(features))
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = time.time()

        if time.time() - last_train_time > RETRAIN_INTERVAL:
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            last_train_time = time.time()

        # ===== Send real candles =====

        real60 = df.tail(60)

        for _, row in real60.iterrows():

            requests.post(BACKEND_URL, json={
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "ema20": float(row["EMA20"]),
                "sma50": float(row["SMA50"]),
                "vwap": float(row["VWAP"]),
                "rsi": float(row["RSI"]),
                "signal": None,
                "type": "real"
            }, timeout=5)

        # ===== Predictions =====

        volatility = df["RET"].std()
        last_price = df.iloc[-1]["close"]

        for _ in range(PRED_MINUTES):

            pred_price = last_price * (1 + np.random.normal(0, volatility))

            future_time = df.iloc[-1]["time"] + timedelta(minutes=5)

            requests.post(BACKEND_URL, json={
                "time": future_time.isoformat(),
                "open": float(last_price),
                "high": float(max(last_price, pred_price)),
                "low": float(min(last_price, pred_price)),
                "close": float(pred_price),
                "ema20": None,
                "sma50": None,
                "vwap": None,
                "rsi": None,
                "signal": None,
                "type": "prediction"
            }, timeout=5)

            last_price = pred_price

        print("Sent 60 real + 6 prediction")

    except Exception as e:
        print("Error:", e)

    time.sleep(60)
```
