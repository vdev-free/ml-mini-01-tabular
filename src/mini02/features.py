import pandas as pd

def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
     X = X.copy()
     X['rooms_per_household'] = X['AveRooms'] / X['AveOccup']
     X["bedrooms_ratio"] = X["AveBedrms"] / X["AveRooms"]
     return X