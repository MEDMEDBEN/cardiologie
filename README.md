# SYSTOLE — Diagnostic cardiaque multimodal (PTB-XL)

Tableau de bord + API d'inférence pour le projet AASD : classification multi-label de 5 superclasses
(NORM, MI, STTC, CD, HYP) à partir du signal ECG (CNN 1D résiduel) et du profil clinique (régression logistique).

> Exercice de TP — **pas un outil de diagnostic médical**.

Production : https://cardiologue.mohamedbennabi.tech

## Structure

```
web/                 Tableau de bord (HTML autonome) servi par Nginx
  index.html
  uploads/predictions_tabulaire_test.csv
api/                 API FastAPI (inférence réelle)
  app/main.py        /health, /predict/clinique, /predict/ecg
  app/ecgnet.py      Architecture du CNN (identique au rapport)
  app/clinical.py    Nettoyage + variables du modèle clinique
  training/train_clinique.py   Ré-entraîne le modèle clinique (plis 1–8) et le sauvegarde
models/              Poids (non versionnés) : clinique.joblib, ecg_cnn_v2.pt, ecg_norm.npz
deploy/              Nginx + script d'installation VPS
docker-compose.yml
```

## 1. DNS

Chez votre registrar (domaine `mohamedbennabi.tech`), ajoutez :

| Type | Nom          | Valeur          |
|------|--------------|-----------------|
| A    | cardiologue  | IP_DE_VOTRE_VPS |

Vérifiez : `dig +short cardiologue.mohamedbennabi.tech` doit renvoyer l'IP du VPS.

## 2. Exporter les fichiers du CNN depuis Colab

Le serveur a besoin de **deux fichiers** produits dans votre notebook ECG :

1. `ecg_cnn_v2.pt` (déjà sauvegardé par la partie D du rapport).
2. `ecg_norm.npz` — moyenne / écart-type par dérivation (plis 1–8). Ajoutez après la partie C :

```python
np.savez('ecg_norm.npz', mean=mean, std=std)
from google.colab import files
files.download('ecg_cnn_v2.pt'); files.download('ecg_norm.npz')
```

## 3. Installer sur le VPS (Ubuntu 22.04 / 24.04)

```bash
ssh root@IP_DU_VPS
git clone https://github.com/MEDMEDBEN/cardiologie.git /opt/systole
cd /opt/systole
```

Depuis votre PC, envoyez les poids du CNN :

```bash
scp ecg_cnn_v2.pt ecg_norm.npz root@IP_DU_VPS:/opt/systole/models/
```

Puis sur le VPS :

```bash
sudo EMAIL=vous@exemple.com bash deploy/setup.sh
```

Le script installe Docker, Nginx et Certbot, construit l'API, **ré-entraîne le modèle clinique**
(télécharge PTB-XL v1.0.1 depuis PhysioNet, ~7 Mo), configure Nginx et le certificat HTTPS.

## 3 bis. VPS où le port 80 est déjà pris (mode 8080)

Aucun Nginx système, aucun Certbot, aucun conflit avec les services existants :

```bash
cd /opt/systole && git pull
docker compose up -d --build          # api + web (nginx:alpine)
sudo ufw allow 8080/tcp               # si UFW est actif
```

Accès : `http://DOMAINE:8080` (front) et `http://DOMAINE:8080/api/health`.
Le conteneur `web` sert `web/` et proxifie `/api/` vers `api:8000`.
Ne pas lancer `deploy/setup.sh` dans ce mode : il configure Nginx sur le port 80.
HTTPS sur 8080 : à gérer par le reverse proxy déjà présent sur 80/443, pointé vers `127.0.0.1:8080`.

## 4. Vérifier

```bash
curl https://cardiologue.mohamedbennabi.tech/api/health
# {"status":"ok","clinique":true,"ecg":true,...}
curl -X POST https://cardiologue.mohamedbennabi.tech/api/predict/clinique \
  -H 'Content-Type: application/json' -d '{"age":64,"sex":0,"height":172,"weight":80}'
```

Documentation interactive : https://cardiologue.mohamedbennabi.tech/api/docs

## 5. Mettre à jour

```bash
cd /opt/systole && git pull
docker compose up -d --build
sudo systemctl reload nginx
```

## API

| Méthode | Route                 | Entrée                                             |
|---------|-----------------------|----------------------------------------------------|
| GET     | /api/health           | —                                                  |
| POST    | /api/predict/clinique | JSON `{age, sex (0 homme, 1 femme), height?, weight?}` |
| POST    | /api/predict/ecg      | multipart `file` : .npy ou .csv, 1000 × 12, 100 Hz, mV |

Seuils : clinique 0,5 ; ECG NORM 0,36 · MI 0,31 · STTC 0,28 · CD 0,46 · HYP 0,30 (choisis sur le pli 9).

## Résultats de référence (pli 10)

| Modèle | AUC macro | F1 macro |
|---|---|---|
| CNN ECG | 0,928 | 0,748 |
| Clinique (LR balanced) | — | 0,461 |

## Références

- Wagner P. et al. (2020). PTB-XL, a large publicly available ECG dataset. *Scientific Data*.
- Strodthoff N. et al. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE JBHI*.
