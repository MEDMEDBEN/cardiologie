"""Entraîne le CNN ECG (ECGNet) sur PTB-XL et sauvegarde les artefacts de l'API.

Produit dans MODELS_DIR :
  - ecg_cnn_v2.pt   state_dict du meilleur modèle (sélection sur le pli 9)
  - ecg_norm.npz    mean / std par dérivation, calculés sur les plis 1-8

Usage (depuis le dossier où se trouvent les paquets app/ et training/) :
    python -m training.train_ecg
    DATA_DIR=/content/ptbxl MODELS_DIR=/content/models python -m training.train_ecg

Signal attendu : 1000 × 12 à 100 Hz (records100 de PTB-XL).
Sur CPU c'est très long : préférer un GPU (Colab, machine locale).
"""
import ast
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from app.ecgnet import ECGNet

BASE = "https://physionet.org/files/ptb-xl/1.0.1/"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CACHE = os.environ.get("DATA_DIR", "/tmp/ptbxl")
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = int(os.environ.get("BATCH", "64"))
PATIENCE = int(os.environ.get("PATIENCE", "6"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = int(os.environ.get("SEED", "42"))


def download(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "systole-train/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        f.write(r.read())


def fetch(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print("Téléchargement", name)
        download(BASE + name, path)
    return path


def load_db():
    df = pd.read_csv(fetch("ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    stmt = pd.read_csv(fetch("scp_statements.csv"), index_col=0)
    stmt = stmt[stmt["diagnostic"] == 1]
    df["superclass"] = df["scp_codes"].apply(
        lambda codes: {stmt.loc[c, "diagnostic_class"] for c in codes if c in stmt.index})
    df = df[df["superclass"].map(len) > 0].copy()
    for c in CLASSES:
        df[c] = df["superclass"].apply(lambda s, c=c: float(c in s))
    return df


def download_records(df):
    missing = [f for f in df["filename_lr"] if not os.path.exists(os.path.join(CACHE, f))]
    if not missing:
        return
    print(f"Téléchargement de {len(missing)} enregistrements 100 Hz…")

    def get(name):
        dst = os.path.join(CACHE, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(3):
            try:
                download(BASE + name, dst)
                return
            except Exception:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(get, missing), 1):
            if i % 2000 == 0:
                print(f"  {i}/{len(missing)}")


def stats(df):
    total, total_sq, n = np.zeros(12), np.zeros(12), 0
    for name in df["filename_lr"]:
        x = np.load(os.path.join(CACHE, name)).astype(np.float64)
        total += x.sum(0)
        total_sq += (x ** 2).sum(0)
        n += x.shape[0]
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class Records(Dataset):
    def __init__(self, df, mean, std):
        self.names = df["filename_lr"].tolist()
        self.labels = df[CLASSES].to_numpy(np.float32)
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        x = np.load(os.path.join(CACHE, self.names[i])).astype(np.float32)  # (1000, 12)
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x.T)), torch.from_numpy(self.labels[i])


def evaluate(model, loader, criterion, dev):
    model.eval()
    losses, probas, targets = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            losses.append(criterion(out, yb).item())
            probas.append(torch.sigmoid(out).cpu().numpy())
            targets.append(yb.cpu().numpy())
    p, y = np.concatenate(probas), np.concatenate(targets)
    return (float(np.mean(losses)),
            float(roc_auc_score(y, p, average="macro")),
            float(f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Périphérique :", dev)

    df = load_db()
    download_records(df)

    tr = df[df["strat_fold"].between(1, 8)]
    va = df[df["strat_fold"] == 9]
    te = df[df["strat_fold"] == 10]
    print(f"plis 1-8 : {len(tr)} · pli 9 : {len(va)} · pli 10 : {len(te)}")

    mean, std = stats(tr)
    print("mean :", np.round(mean, 3))
    print("std  :", np.round(std, 3))

    train_loader = DataLoader(Records(tr, mean, std), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(Records(va, mean, std), batch_size=BATCH)
    test_loader = DataLoader(Records(te, mean, std), batch_size=BATCH)

    model = ECGNet().to(dev)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc, best_state, bad = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        val_loss, auc, f1 = evaluate(model, val_loader, criterion, dev)
        print(f"époque {epoch:02d} · train {running / len(tr):.4f} · val {val_loss:.4f} · "
              f"AUC {auc:.4f} · F1 {f1:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"arrêt anticipé (patience {PATIENCE})")
                break

    model.load_state_dict(best_state)
    test_loss, test_auc, test_f1 = evaluate(model, test_loader, criterion, dev)
    print(f"pli 9 AUC {best_auc:.4f} · pli 10 AUC {test_auc:.4f} · F1 {test_f1:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(best_state, os.path.join(MODELS_DIR, "ecg_cnn_v2.pt"))
    np.savez(os.path.join(MODELS_DIR, "ecg_norm.npz"), mean=mean, std=std)
    print("Sauvegardé :", os.path.join(MODELS_DIR, "ecg_cnn_v2.pt"))
    print("Sauvegardé :", os.path.join(MODELS_DIR, "ecg_norm.npz"))


if __name__ == "__main__":
    main()
