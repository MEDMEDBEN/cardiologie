"""Ré-entraîne le modèle clinique exactement comme le notebook projet_AASD et le sauvegarde.

Usage (dans le conteneur) : python -m training.train_clinique
"""
import ast
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.clinical import FEATURES, clean_frame

BASE = "https://physionet.org/files/ptb-xl/1.0.1/"
CACHE = os.environ.get("DATA_DIR", "/tmp/ptbxl")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def fetch(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print("Téléchargement", name)
        urllib.request.urlretrieve(BASE + name, path)
    return path


def main():
    df = pd.read_csv(fetch("ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    stmt = pd.read_csv(fetch("scp_statements.csv"), index_col=0)
    stmt = stmt[stmt["diagnostic"] == 1]

    def to_superclass(codes):
        return sorted({stmt.loc[c, "diagnostic_class"] for c in codes if c in stmt.index})

    df["superclass"] = df["scp_codes"].apply(to_superclass)
    df = df[df["superclass"].map(len) > 0]
    for c in CLASSES:
        df[c] = df["superclass"].apply(lambda s, c=c: int(c in s))

    X = df[["age", "sex", "height", "weight"]].copy()
    if 13666 in X.index:
        X.loc[13666, "height"] = np.nan
    X = clean_frame(X)
    Y = df[CLASSES]

    tr = df["strat_fold"].between(1, 8)
    va = df["strat_fold"] == 9
    te = df["strat_fold"] == 10

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"))),
    ])
    model.fit(X[tr], Y[tr])

    for name, m in (("validation (pli 9)", va), ("test (pli 10)", te)):
        f1 = f1_score(Y[m], model.predict(X[m]), average="macro", zero_division=0)
        print(f"F1 macro {name} : {f1:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, "clinique.joblib")
    joblib.dump(model, out)
    print("Modèle sauvegardé :", out, "| variables :", FEATURES)


if __name__ == "__main__":
    main()
