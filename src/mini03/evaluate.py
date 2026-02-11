from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score


def silhouette(scaled_df: pd.DataFrame, labels: pd.Series) -> float:
    return float(silhouette_score(scaled_df, labels))
