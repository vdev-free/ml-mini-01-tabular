from fastapi import FastAPI, HTTPException
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from mini02.schemas import HousingFeatures, PredictResponse, PredictBatchResponse

app = FastAPI(title="mini02-regression")

MODEL_PATH = Path('artifacts/mini02/model.joblib')
model = joblib.load(MODEL_PATH)

@app.get('/health')
def health() -> dict:
    return{'status': 'ok'}

@app.post('/predict', response_model=PredictResponse)
def predict(features: HousingFeatures) -> PredictResponse:
    payload = features.model_dump()
    df = pd.DataFrame([payload])

    prediction = float(model.predict(df)[0])

    return(PredictResponse(prediction=prediction))

@app.post('/predict_batch', response_model=PredictBatchResponse)
def predict_batch(features_list: list[HousingFeatures]) -> PredictBatchResponse:
    payload = [item.model_dump() for item in features_list]
    df = pd.DataFrame(payload)

    preds = model.predict(df)
    predictions = [float(p) for p in preds]

    return PredictBatchResponse(predictions=predictions)
