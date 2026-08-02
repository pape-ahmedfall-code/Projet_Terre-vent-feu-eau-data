# Data Platform — Prédiction du Risque d'Incendie en France 

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![DVC](https://img.shields.io/badge/DVC-Tracked-orange)
![Framework](https://img.shields.io/badge/PyTorch-MLP_53--102--126-red)
![Database](https://img.shields.io/badge/DuckDB-Consolidated-yellow)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-Passing-brightgreen)

## 📌 Présentation du Projet

Cette plateforme MLOps de bout en bout anticipe et évalue le risque d'incendie de forêt à l'échelle communale en France. Elle croise plus de 50 ans d'historiques d'incendies issus de la base **BDIFF (1973–2024)** avec les données météorologiques quotidiennes **SIM2/MétéoNet** et le référentiel géographique des communes de l'INSEE.

Le système s'appuie sur un réseau de neurones (MLP) à 3 couches ($53 \rightarrow 102 \rightarrow 126$) entraîné sous PyTorch avec Focal Loss, une explicabilité SHAP intégrée et une application web interactive Streamlit conteneurisée via Docker.

---

## 🏗️ Architecture du Système & Pipeline MLOps

```text
┌────────────────────────┐
│   Sources Brutes       │
│ (BDIFF, Météo, INSEE)  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Ingestion SQL Massive  │
│       (DuckDB)         │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  Feature Engineering   │
│ (Negative Sampling,    │
│  HDBSCAN, Lags, Sin/Cos)│
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Modélisation PyTorch   │
│ (MLP + Focal Loss +    │
│  Optuna + Metadata)    │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   Application Web      │
│  (Streamlit + SHAP)    │
└────────────────────────┘

---

🛠️ Stack Technique
- Langage & Environnement : Python 3.11, Virtualenv, Docker (CPU optimized)

- Gestion des Données & SQL : DuckDB, Pandas, NumPy, PyArrow/Parquet

- Machine Learning & Deep Learning : PyTorch (MLP 53-102-126, Dropout 0.337), Scikit-Learn (HDBSCAN), Optuna, SHAP

- Visualisation & UI : Streamlit, Folium / Plotly

- MLOps & DevOps : Git, DVC (Data Version Control), GitHub Actions (CI/CD avec Pytest)

---

📂 Structure du Répertoire

├── .dvc/                   # Configuration DVC
├── .github/workflows/      # Pipelines CI/CD (ci-cd.yml)
├── data/                   # Géré par DVC (exclu du suivi Git direct)
│   ├── raw/                # Données brutes BDIFF et Météo
│   └── processed/          # Matrices filtrées, parquet, base DuckDB
├── models/                 # Artefacts du modèle (model_focal.pt, scaler.joblib, model_metadata.json)
├── tests/                  # Tests unitaires automatisés (test_pipeline.py)
├── app.py                  # Application Streamlit principale
├── sql_consolidation.py    # Script d'ingestion SQL DuckDB
├── CADRAGE_TECHNIQUE.md    # Note de cadrage RNCP 40573 & choix d'architecture
├── Dockerfile              # Fichier de conteneurisation de l'application (PyTorch CPU)
├── .dockerignore           # Contexte de build optimisé par liste blanche (<120 ko)
├── data.dvc                # Pointeur DVC pour le répertoire data
├── models.dvc              # Pointeur DVC pour le répertoire models
└── requirements.txt        # Dépendances Python du projet

---

🚀 Installation et Démarrage Rapide
Prérequis
- Git

- Python 3.11+

- Docker Desktop

---

1. Clonage du Projet

git clone [https://github.com/votre-user/projet_plateforme.git](https://github.com/votre-user/projet_plateforme.git)
cd projet_plateforme

---

2. Configuration de l'Environnement Virtuel

python -m venv .venv
# Sur Windows (cmd) :
.venv\Scripts\activate
# Sur Linux/MacOS :
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

---

3. Récupération des Données et Artefacts via DVC

dvc pull

---

⚙️ Exécution des Pipelines

A. Ingestion et Consolidation SQL (DuckDB)
Pour ré-exécuter la jointure massive (69 millions+ de lignes) entre le référentiel INSEE, la matrice Negative Sampling et les séries temporelles météo :

python sql_consolidation.py

--

B. Lancement de l'Application Streamlit en Local

streamlit run app.py

L'interface sera accessible sur http://localhost:8501.

---

🐳 Déploiement via Docker

1. Construction de l'Image Docker
L'image s'appuie sur une liste blanche .dockerignore garantissant un transfert de contexte ultra-léger (~119 ko).

docker build -t fire-risk-mlps:1.0 .

---

2. Lancement du Conteneur
Bash

docker run -d -p 8501:8501 --name fire_risk_app fire-risk-mlps:1.0

Accédez à l'application via http://localhost:8501.

---

🧪 Qualité du Code & CI/CD

Les tests unitaires vérifient le chargement des artefacts (model_metadata.json, scaler.joblib, model_focal.pt), l'alignement des dimensions ($53 \rightarrow 102 \rightarrow 126$) et la validité de l'inférence.

Pour exécuter les tests localement : pytest tests/

Le pipeline GitHub Actions (.github/workflows/ci-cd.yml) déclenche automatiquement l'exécution des tests et le build Docker à chaque push ou pull_request sur la branche main ou master.

---

📊 Méthodologie & Spécifications du Modèle

1. Negative Sampling : Équilibrage de la matrice d'apprentissage par génération de paires (commune, date) sans incendie.
2. Features Spatio-Temporelles :
  - Spatial : Clustering HDBSCAN sur coordonnées centroïdes (6 clusters identifiés via la métrique Haversine).
  - Temporel : Encodage cyclique ($\sin/\cos$) du jour de l'année et du mois, indicateur is_weekend.
  - Sécheresse Cumulée : Lags et moyennes glissantes à 7, 15 et 30 jours sur les précipitations et températures.

3. Architecture & Métadonnées MLP :
  - Structure : 53 variables d'entrée $\rightarrow$ couche 1 (102) $\rightarrow$ couche 2 (126) $\rightarrow$ sortie ( activation Sigmoid).
  - Hyperparamètres : Dropout de 0,337, perte Focal Loss, et seuil de décision opérationnel fixe réglé à 0,5000.
  - Les métadonnées d'architecture sont lues dynamiquement depuis models/model_metadata.json.

---

📄 Licence et Documentation RNCP

Pour consulter la note d'architecture complète, la justification des choix d'éco-conception et la matrice des risques, référez-vous au fichier CADRAGE_TECHNIQUE.md.
