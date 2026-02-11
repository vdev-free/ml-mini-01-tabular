from __future__ import annotations

import numpy as np
import pandas as pd


def make_customers(seed: int = 42, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # 3 "типи" клієнтів (ми їх не показуємо моделі, це для реалістичності)
    n1 = int(n * 0.5)  # звичайні
    n2 = int(n * 0.35) # активні
    n3 = n - n1 - n2   # VIP

    freq = np.concatenate([
        rng.poisson(lam=2, size=n1),     # купує рідко
        rng.poisson(lam=8, size=n2),     # купує частіше
        rng.poisson(lam=18, size=n3),    # купує дуже часто
    ])

    spend = np.concatenate([
        rng.normal(loc=60, scale=20, size=n1),    # витрати невеликі
        rng.normal(loc=350, scale=120, size=n2),  # середні
        rng.normal(loc=1200, scale=300, size=n3), # великі
    ])

    # трохи "захисту" від негативних значень
    freq = np.clip(freq, 0, None)
    spend = np.clip(spend, 0, None)

    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "purchases_30d": freq,
        "spend_30d": spend,
    })

    return df