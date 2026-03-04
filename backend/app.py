```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="XAUUSD Backend (5m)")

# Allow frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Storage =====
REAL_DATA = {}
PRED_DATA = {}

# ===== Limits =====
MAX_REAL = 60   # Last 300 minutes (60 x 5m)
MAX_PRED = 6    # Next 30 minutes (6 x 5m)

last_real_time = None


@app.get("/")
def home():
    return {
        "status": "XAUUSD backend running",
        "timeframe": "5 minutes",
        "history": "300 minutes",
        "prediction": "30 minutes"
    }


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA, last_real_time

    t = payload.get("time")
    typ = payload.get("type")

    if not t or not typ:
        return {"status": "ignored"}

    # ===== REAL DATA =====
    if typ == "real":
        REAL_DATA[t] = payload

        # Clear predictions when new candle arrives
        if last_real_time is None or t > last_real_time:
            last_real_time = t
            PRED_DATA = {}

        while len(REAL_DATA) > MAX_REAL:
            REAL_DATA.pop(next(iter(REAL_DATA)))

    # ===== PREDICTIONS =====
    elif typ == "prediction":
        PRED_DATA[t] = payload

        while len(PRED_DATA) > MAX_PRED:
            PRED_DATA.pop(next(iter(PRED_DATA)))

    return {"status": "ok"}


@app.get("/data")
def get_data():
    real = [REAL_DATA[k] for k in sorted(REAL_DATA.keys())]
    pred = [PRED_DATA[k] for k in sorted(PRED_DATA.keys())]
    return real + pred


@app.get("/stats")
def stats():
    return {
        "real_candles": len(REAL_DATA),
        "predictions": len(PRED_DATA),
        "last_real_time": last_real_time
    }
```
