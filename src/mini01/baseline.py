from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import mlflow
from mini01.evaluate import evaluate_predictions

def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mini01-tabular")

    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ]
    )

    with mlflow.start_run(run_name='baseline-logreg'):
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            
        print('metrics:', metrics)


if __name__ == '__main__':
    main()
