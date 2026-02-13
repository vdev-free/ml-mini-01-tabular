from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

def fit_kmeans(
        scaled_df: pd.DataFrame,
        k: int = 3,
        seed: int = 42,
) -> tuple[KMeans, pd.Series]:
    model = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    labels = model.fit_predict(scaled_df)

    return model, pd.Series(labels, name='cluster')
    