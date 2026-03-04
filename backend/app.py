from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

REAL_DATA = {}
PRED_DATA = {}

MAX_REAL = 60
MAX_PRED = 6
last_real_time = None


@app.get("/")
def home():
    return {"status": "backend running"}


@app.post("/update")
def update(payload: dict):
    global REAL_DATA, PRED_DATA, last_real_time

    t = payload["time"]
    typ = payload.get("type")

    if typ == "real":
        REAL_DATA[t] = payload

        if last_real_time is None or t > last_real_time:
            last_real_time = t
            PRED_DATA = {}

        while len(REAL_DATA) > MAX_REAL:
            REAL_DATA.pop(next(iter(REAL_DATA)))

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
