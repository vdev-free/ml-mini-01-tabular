from __future__ import annotations

import joblib
from pathlib import Path

from mini03.data import make_customers
from mini03.features import scale_features
from mini03.cluster import fit_kmeans

ARTIFACT_DIR = Path('artifacts/mini03')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save() -> None:
    df = make_customers(n=1000)
    scaled_df, scaler = scale_features(df)
    model, labels = fit_kmeans(scaled_df, k=3)

    joblib.dump(model, ARTIFACT_DIR / 'kmeans.pkl')
    joblib.dump(scaler, ARTIFACT_DIR / 'scaler.pkl')

    print("Saved model and scaler")

if __name__ == "__main__":
    train_and_save()