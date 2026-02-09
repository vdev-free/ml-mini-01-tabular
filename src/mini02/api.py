from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd
from mini02.schemas import HousingFeatures, PredictResponse

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