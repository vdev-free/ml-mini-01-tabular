from typing import Dict
from sklearn.metrics import accuracy_score


def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc}
