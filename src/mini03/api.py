from __future__ import annotations
from fastapi import FastAPI
from pathlib import Path
import joblib
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

class SegmentRequest(BaseModel):
    purchases_30d: float = Field(ge=0)
    spend_30d: float = Field(ge=0)

class SegmentResponse(BaseModel):
    segment: str
    cluster: int

ART_DIR = Path('artifacts/mini03')
MODEL_PATH = ART_DIR / "kmeans.pkl"
SCALER_PATH = ART_DIR / "scaler.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    app.state.scaler = joblib.load(SCALER_PATH)
    yield

app = FastAPI(title='mini03-customer-segmentation', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLUSTER_TO_SEGMENT = {
    1: "VIP",
    0: "Regular",
    2: "Low",
}

@app.get("/health")
def health() -> dict:
    ok = hasattr(app.state, "model") and hasattr(app.state, "scaler")
    return {"status": "ok" if ok else "not_ready", "artifacts_loaded": ok}

@app.post('/segment', response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:

    x = pd.DataFrame(
    [{"purchases_30d": req.purchases_30d, "spend_30d": req.spend_30d}]
    )
    
    scaler: StandardScaler = app.state.scaler
    model: KMeans = app.state.model

    x_scaled = scaler.transform(x)
    cluster = int(model.predict(x_scaled)[0])

    seg = CLUSTER_TO_SEGMENT.get(cluster, 'Unknown')

    return SegmentResponse(segment=seg, cluster=cluster)