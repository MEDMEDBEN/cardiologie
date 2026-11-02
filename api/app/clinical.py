import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "height", "weight", "bmi", "height_missing", "weight_missing"]


def clean_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Mêmes règles de nettoyage que le notebook projet_AASD."""
    X = X.copy()
    X.loc[(X["height"] < 40) | (X["height"] > 230), "height"] = np.nan
    X.loc[(X["weight"] < 2) | (X["weight"] > 250), "weight"] = np.nan
    X.loc[(X["age"] >= 18) & (X["weight"] < 30), "weight"] = np.nan
    X.loc[(X["age"] >= 18) & (X["height"] < 120), "height"] = np.nan
    X["bmi"] = X["weight"] / (X["height"] / 100) ** 2
    X.loc[(X["bmi"] < 10) | (X["bmi"] > 70), "bmi"] = np.nan
    X["height_missing"] = X["height"].isna().astype(int)
    X["weight_missing"] = X["weight"].isna().astype(int)
    return X[FEATURES]


def build_features(age, sex, height=None, weight=None) -> pd.DataFrame:
    X = pd.DataFrame([{
        "age": float(age), "sex": int(sex),
        "height": np.nan if height is None else float(height),
        "weight": np.nan if weight is None else float(weight),
    }])
    return clean_frame(X)
