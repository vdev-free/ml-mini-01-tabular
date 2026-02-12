from __future__ import annotations
from fastapi import FastAPI
from pathlib import Path
import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


class SegmentRequest(BaseModel):
    purchases_30d: float = Field(ge=0)
    spend_30d: float = Field(ge=0)

class SegmentResponse(BaseModel):
    segment: str
    cluster: int

app = FastAPI(title='mini03-customer-segmentation')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ART_DIR = Path('artifacts/mini03')
MODEL_PATH = ART_DIR / "kmeans.pkl"
SCALER_PATH = ART_DIR / "scaler.pkl"

model: KMeans | None = None
scaler: StandardScaler | None = None

@app.on_event('startup')
def load_artifacts() -> None:
    global model, scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

CLUSTER_TO_SEGMENT = {
    1: "VIP",
    0: "Regular",
    2: "Low",
}

@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}

@app.post('/segment', response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    assert model is not None and scaler is not None

    x = pd.DataFrame(
    [{"purchases_30d": req.purchases_30d, "spend_30d": req.spend_30d}]
    )
    x_scaled = scaler.transform(x)
    cluster = int(model.predict(x_scaled)[0])
    seg = CLUSTER_TO_SEGMENT.get(cluster, 'Unknown')

    return SegmentResponse(segment=seg, cluster=cluster)