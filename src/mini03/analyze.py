from __future__ import annotations

import pandas as pd


def describe_clusters(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("cluster")[["purchases_30d", "spend_30d"]]
        .mean()
        .sort_values("spend_30d", ascending=False)
    )

    return summary
