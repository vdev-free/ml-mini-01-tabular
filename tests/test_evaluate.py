from mini01.evaluate import evaluate_predictions

def test_evaluate_predictions_accurancy() -> None:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = evaluate_predictions(y_true, y_pred)

    assert 'accuracy' in metrics
    assert metrics['accuracy'] == 0.75