import os
import pandas as pd
import numpy as np
import requests
import time
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import ta
import pytz

from flask import Flask
import threading

# ================= FLASK SERVER (Render requires open port) =================

app = Flask(__name__)

@app.route("/")
def home():
    return "XAUUSD Producer Running"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask, daemon=True).start()


# ================= CONFIG =================

BACKEND_URL = "https://xauusd-klue.onrender.com/update"
BACKEND_HOME = "https://xauusd-klue.onrender.com/"
SERVICE_URL = "https://producer-ngxd.onrender.com/"

API_KEY = os.getenv("TWELVEDATA_API_KEY")

SYMBOL = "XAU/USD"
INTERVAL = "5min"

LOOKBACK = 60
PRED_MINUTES = 6           # 6 x 5-min candles = 30 minutes forward
RETRAIN_INTERVAL = 1800

IST = pytz.timezone("Asia/Kolkata")

print("Producer started — XAUUSD 5m | 30-candle LSTM predictions")


# ================= VARIABLES =================

model = None
scaler = MinMaxScaler()
last_candle_time = None
last_train_time = None

last_self_ping = 0
last_backend_ping = 0


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
        "apikey": API_KEY,
        # Force UTC from API so we always know the source timezone
        "timezone": "UTC"
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "values" not in data:
        print("API error:", data)
        return None

    df = pd.DataFrame(data["values"])

    # ── TIMEZONE FIX ──────────────────────────────────────────────────────────
    # TwelveData returns timestamps in the requested timezone (UTC here).
    # We parse as UTC first, then convert to IST — this avoids the
    # "New York / naive datetime" offset error that occurred when the API
    # defaulted to America/New_York without an explicit tz= param.
    df["time"] = pd.to_datetime(df["datetime"], utc=True)        # parse as UTC
    df["time"] = df["time"].dt.tz_convert(IST)                   # convert → IST
    # ─────────────────────────────────────────────────────────────────────────

    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    df["volume"] = 1.0          # TwelveData XAU/USD has no real volume; use 1

    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


# ================= MAIN LOOP =================

while True:
    try:

        # ===== SELF KEEP ALIVE =====
        if time.time() - last_self_ping > 300:
            try:
                requests.get(SERVICE_URL, timeout=5)
                print("Self ping successful")
            except:
                print("Self ping failed")
            last_self_ping = time.time()

        # ===== BACKEND KEEP ALIVE =====
        if time.time() - last_backend_ping > 300:
            try:
                requests.get(BACKEND_HOME, timeout=5)
                print("Backend ping successful")
            except:
                print("Backend ping failed")
            last_backend_ping = time.time()

        # ===== FETCH DATA =====
        df = fetch_data()

        if df is None or len(df) < 100:
            print("Data issue — retrying in 60s")
            time.sleep(60)
            continue

        # ===== NEW CANDLE CHECK (use last CLOSED candle, i.e. iloc[-2]) =====
        current_time = df.iloc[-2]["time"]

        if last_candle_time == current_time:
            print("Waiting for new candle...")
            time.sleep(60)
            continue

        last_candle_time = current_time
        print("New closed candle:", current_time)

        # ===== INDICATORS =====
        df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["SMA50"] = df["close"].rolling(50).mean()
        df["RSI"]   = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["VWAP"]  = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df["RET"]   = df["close"].pct_change()

        df = df.dropna().reset_index(drop=True)

        # ===== FEATURES =====
        features = ["RET", "EMA20", "SMA50", "RSI", "VWAP"]
        scaled = scaler.fit_transform(df[features])

        # ===== SEQUENCES =====
        X, y = [], []
        for i in range(LOOKBACK, len(scaled)):
            X.append(scaled[i - LOOKBACK:i])
            y.append(scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        if len(X) == 0:
            print("Not enough sequences — retrying in 60s")
            time.sleep(60)
            continue

        # ===== TRAIN MODEL =====
        if model is None:
            model = build_model(len(features))
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            last_train_time = time.time()
            print("Model trained (initial)")

        if time.time() - last_train_time > RETRAIN_INTERVAL:
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            last_train_time = time.time()
            print("Model retrained")

        # ===== SEND LAST 60 REAL CANDLES =====
        real60 = df.tail(60).reset_index(drop=True)

        for i, row in real60.iterrows():
            signal = None
            if i > 0:
                prev = real60.iloc[i - 1]
                if row["EMA20"] > row["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                    signal = "BUY"
                elif row["EMA20"] < row["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                    signal = "SELL"

            requests.post(BACKEND_URL, json={
                "time":   row["time"].isoformat(),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "ema20":  float(row["EMA20"]),
                "sma50":  float(row["SMA50"]),
                "vwap":   float(row["VWAP"]),
                "rsi":    float(row["RSI"]),
                "signal": signal,
                "type":   "real"
            }, timeout=5)

        # ===== PREDICTIONS (LSTM + VOLATILITY — 30 candles × 5 min) =====
        volatility  = df["RET"].std()
        last_price  = df.iloc[-1]["close"]

        # Seed sequence from the last LOOKBACK scaled rows
        last_seq = scaled[-LOOKBACK:]          # shape (LOOKBACK, n_features)
        temp_df  = df.copy()

        for _ in range(PRED_MINUTES):

            # 1. LSTM prediction (scaled RET)
            pred_scaled = model.predict(
                last_seq.reshape(1, LOOKBACK, len(features)),
                verbose=0
            )[0][0]

            # 2. Inverse-transform only the RET column
            dummy = np.hstack([[pred_scaled], np.zeros(len(features) - 1)]).reshape(1, -1)
            pred_ret = scaler.inverse_transform(dummy)[0][0]

            # 3. Clip extreme returns (±2.5σ)
            pred_ret = np.clip(pred_ret, -2.5 * volatility, 2.5 * volatility)
            pred_price = last_price * (1 + pred_ret)

            # 4. EMA anchor (prevent wild drift)
            ema = temp_df.iloc[-1]["EMA20"]
            pred_price = 0.9 * pred_price + 0.1 * ema

            # 5. Add micro-noise for realistic candle variation
            pred_price *= (1 + np.random.normal(0, volatility / 2))

            # 6. Realistic wick sizing
            body  = abs(pred_price - last_price)
            wick  = max(body * 0.5, last_price * volatility * 0.2)
            high_p = max(last_price, pred_price) + wick
            low_p  = min(last_price, pred_price) - wick

            # 7. Advance time by 5 minutes (XAUUSD interval)
            future_time = temp_df.iloc[-1]["time"] + timedelta(minutes=5)

            new_row = {
                "time":   future_time,
                "open":   last_price,
                "high":   high_p,
                "low":    low_p,
                "close":  pred_price,
                "volume": temp_df.iloc[-1]["volume"]
            }

            # 8. Append and recompute indicators so the next step is consistent
            temp_df = pd.concat(
                [temp_df, pd.DataFrame([new_row])], ignore_index=True
            )
            temp_df["EMA20"] = temp_df["close"].ewm(span=20, adjust=False).mean()
            temp_df["SMA50"] = temp_df["close"].rolling(50).mean()
            temp_df["RSI"]   = ta.momentum.RSIIndicator(temp_df["close"]).rsi()
            temp_df["VWAP"]  = (
                (temp_df["close"] * temp_df["volume"]).cumsum()
                / temp_df["volume"].cumsum()
            )
            temp_df["RET"] = temp_df["close"].pct_change()

            latest = temp_df.iloc[-1]
            prev   = temp_df.iloc[-2]

            # 9. Signal detection on predicted candle
            signal = None
            if latest["EMA20"] > latest["SMA50"] and prev["EMA20"] <= prev["SMA50"]:
                signal = "BUY"
            elif latest["EMA20"] < latest["SMA50"] and prev["EMA20"] >= prev["SMA50"]:
                signal = "SELL"

            requests.post(BACKEND_URL, json={
                "time":   latest["time"].isoformat(),
                "open":   float(latest["open"]),
                "high":   float(latest["high"]),
                "low":    float(latest["low"]),
                "close":  float(latest["close"]),
                "ema20":  float(latest["EMA20"]),
                "sma50":  float(latest["SMA50"]),
                "vwap":   float(latest["VWAP"]),
                "rsi":    float(latest["RSI"]),
                "signal": signal,
                "type":   "prediction"
            }, timeout=5)

            last_price = pred_price

            # 10. Roll the sequence forward with the new candle's scaled features
            valid = temp_df[features].dropna()
            last_seq = scaler.transform(valid.tail(LOOKBACK))

        print("Sent 60 real + 6 prediction candles (30 mins ahead)")

    except Exception as e:
        print("Error:", e)

    time.sleep(60)
