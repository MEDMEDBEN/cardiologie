import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .clinical import build_features
from .ecgnet import ECGNet

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLINIQUE_THRESHOLDS = [0.5] * 5
ECG_THRESHOLDS = [0.36, 0.31, 0.28, 0.46, 0.30]
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
MAX_UPLOAD = 5_000_000

state = {"clinique": None, "ecg": None, "mean": None, "std": None}


def load_models():
    p = os.path.join(MODELS_DIR, "clinique.joblib")
    if os.path.exists(p):
        state["clinique"] = joblib.load(p)
    w = os.path.join(MODELS_DIR, "ecg_cnn_v2.pt")
    n = os.path.join(MODELS_DIR, "ecg_norm.npz")
    if os.path.exists(w) and os.path.exists(n):
        torch.set_num_threads(2)
        model = ECGNet()
        model.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        model.eval()
        z = np.load(n)
        state["ecg"] = model
        state["mean"] = np.asarray(z["mean"], dtype=np.float32).reshape(1, 12)
        state["std"] = np.asarray(z["std"], dtype=np.float32).reshape(1, 12)


@asynccontextmanager
async def lifespan(_app):
    load_models()
    yield


app = FastAPI(title="SYSTOLE API", version="1.0.0", root_path=os.environ.get("ROOT_PATH", ""), lifespan=lifespan)


def result(p, thresholds, model):
    return {
        "model": model,
        "classes": [
            {"code": c, "proba": round(float(p[i]), 4), "seuil": thresholds[i], "positif": bool(p[i] >= thresholds[i])}
            for i, c in enumerate(CLASSES)
        ],
        "avertissement": "Exercice pédagogique — pas un outil de diagnostic.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "clinique": state["clinique"] is not None, "ecg": state["ecg"] is not None, "classes": CLASSES}


class ClinicalIn(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1, description="0 = homme, 1 = femme")
    height: Optional[float] = Field(None, description="cm")
    weight: Optional[float] = Field(None, description="kg")


@app.post("/predict/clinique")
def predict_clinique(x: ClinicalIn):
    if state["clinique"] is None:
        raise HTTPException(503, "Modèle clinique absent : lancez python -m training.train_clinique")
    X = build_features(x.age, x.sex, x.height, x.weight)
    p = np.asarray(state["clinique"].predict_proba(X))[0]
    return result(p, CLINIQUE_THRESHOLDS, "clinique")


def read_signal(raw: bytes, name: str) -> np.ndarray:
    if name.endswith(".npy"):
        sig = np.load(io.BytesIO(raw), allow_pickle=False)
    else:
        lines = [l for l in raw.decode("utf-8", errors="ignore").strip().splitlines() if l.strip()]
        if lines and any(ch.isalpha() for ch in lines[0].replace("e-", "").replace("E-", "")):
            lines = lines[1:]
        sep = ";" if lines and lines[0].count(";") >= 11 else ","
        sig = np.array([[float(v) for v in l.split(sep)] for l in lines], dtype=np.float32)
    sig = np.asarray(sig, dtype=np.float32).squeeze()
    if sig.shape == (12, 1000):
        sig = sig.T
    if sig.shape != (1000, 12):
        raise HTTPException(400, f"Forme attendue (1000, 12), reçue {tuple(sig.shape)}")
    if not np.isfinite(sig).all():
        raise HTTPException(400, "Le signal contient des valeurs manquantes ou infinies.")
    return sig


@app.post("/predict/ecg")
async def predict_ecg(file: UploadFile = File(...)):
    if state["ecg"] is None:
        raise HTTPException(503, "Modèle ECG absent : copiez ecg_cnn_v2.pt et ecg_norm.npz dans models/")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "Fichier trop volumineux (max 5 Mo).")
    try:
        sig = read_signal(raw, (file.filename or "").lower())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Fichier illisible : .npy ou .csv de 1000 × 12 attendu.")
    x = (sig - state["mean"]) / state["std"]
    t = torch.from_numpy(np.ascontiguousarray(x.T)).unsqueeze(0)
    with torch.no_grad():
        p = torch.sigmoid(state["ecg"](t))[0].numpy()
    return result(p, ECG_THRESHOLDS, "ecg")
