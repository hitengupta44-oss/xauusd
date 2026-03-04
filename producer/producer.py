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
    return "XAUUSD Producer running (5m IST)"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask).start()

# ================= CONFIG =================

BACKEND_URL = "https://your-backend.onrender.com/update"   # CHANGE THIS

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
BASE_URL = "https://api-fxpractice.oanda.com/v3"

SYMBOL = "XAU_USD"
GRANULARITY = "M5"

LOOKBACK = 60        # 300 mins
PRED_MINUTES = 6     # next 30 mins
RETRAIN_INTERVAL = 1800

IST = pytz.timezone("Asia/Kolkata")

print("Producer started — XAUUSD 5m IST")

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
    url = f"{BASE_URL}/instruments/{SYMBOL}/candles"

    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}"
    }

    params = {
        "count": 500,
        "granularity": GRANULARITY,
        "price": "M"
    }

    r = requests.get(url, headers=headers, params=params)
    data = r.json()

    if "candles" not in data:
        print("OANDA error:", data)
        return None

    records = []
    for c in data["candles"]:
        if not c["complete"]:
            continue

        utc_time = pd.to_datetime(c["time"], utc=True)
        ist_time = utc_time.tz_convert(IST)

        records.append({
            "time": ist_time,
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": float(c["volume"])
        })

    df = pd.DataFrame(records)
    return df

# ================= MAIN LOOP =================

while True:
    try:
        df = fetch_data()

        if df is None or len(df) < 100:
            print("Data issue")
            time.sleep(60)
            continue

        # Last closed 5-min candle
        current_time = df.iloc[-1]["time"]

        # Run only when new candle arrives
        if last_candle_time == current_time:
            print("Waiting for new 5m candle...")
            time.sleep(60)
            continue

        last_candle_time = current_time
        print("New candle:", current_time)

        # ===== Indicators =====
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["VWAP"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"] = df["close"].pct_change()

        df = df.dropna()

        features = ["RET","EMA20","SMA50","RSI","VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ===== Sequences =====
        X, y = [], []
        for i in range(LOOKBACK, len(saled := scaled)):
            X.append(saled[i-LOOKBACK:i])
            y.append(saled[i,0])

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

        # ===== Send last 60 real =====
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
        last_seq = scaled[-LOOKBACK:]
        temp_df = df.copy()

        for _ in range(PRED_MINUTES):

            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)),
                verbose=0
            )[0][0]

            pred_ret = scaler.inverse_transform(
                np.hstack([[pred_scaled], np.zeros(len(features)-1)]).reshape(1,-1)
            )[0][0]

            pred_price = last_price * (1 + pred_ret)

            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=5)

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
