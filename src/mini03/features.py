from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    features = df[["purchases_30d", "spend_30d"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled,
        columns=["purchases_scaled", "spend_scaled"]
    )

    return scaled_df, scaler