from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mini02.evaluate import evaluate_predictions
import pandas as pd

def main() -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mini02-regression")


    data = fetch_california_housing(as_frame=True)

    X = data.data
    y = data.target

    def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['rooms_per_household'] = X['AveRooms'] / X['AveOccup']
        X["bedrooms_ratio"] = X["AveBedrms"] / X["AveRooms"]
        return X
    
    feature_engineering = FunctionTransformer(
        add_ratio_features,
        validate=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
        )
    
    model = Pipeline(
        steps=[
            ('features', feature_engineering),
            ('scl', StandardScaler()),
            ('model', LinearRegression() )
        ]
    )
    
    with mlflow.start_run(run_name='baseline-linreg'):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate_predictions(y_test, preds)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        print('metrics', metrics)


if __name__ == '__main__':
    main()