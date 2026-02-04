from sklearn.datasets import load_breast_cancer
import pandas as pd


def main() -> None:
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print()
    print("First 5 rows of X:")
    print(X.head())


if __name__ == "__main__":
    main()

