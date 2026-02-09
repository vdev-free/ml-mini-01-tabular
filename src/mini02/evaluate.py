from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, preds) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_true, preds)

        return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
        }