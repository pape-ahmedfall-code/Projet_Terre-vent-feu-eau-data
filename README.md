# Data Platform — Prédiction du Risque d'Incendie en France 

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![DVC](https://img.shields.io/badge/DVC-Tracked-orange)
![Framework](https://img.shields.io/badge/PyTorch-MLP-red)
![Database](https://img.shields.io/badge/DuckDB-Consolidated-yellow)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-Passing-brightgreen)

## 📌 Présentation du Projet

Cette plateforme MLOps de bout en bout anticipe et évalue le risque d'incendie de forêt à l'échelle communale en France. Développée dans le cadre de la certification **RNCP 40573**, elle croise plus de 50 ans d'historiques d'incendies issus de la base **BDIFF (1973–2024)** avec les données météorologiques quotidiennes **SIM2/MétéoNet** et le référentiel géographique des communes de l'INSEE.

Le système propose une architecture prédictive basée sur un réseau de neurones (MLP) sous PyTorch, combinée à une analyse explicable via SHAP et servie par une application web interactive Streamlit conteneurisée.

---

## Architecture du Système & Pipeline MLOps

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
                          │  (MLP + Focal Loss +   │
                          │    Optuna Tuning)      │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │   Application Web      │
                          │  (Streamlit + SHAP)    │
                          └────────────────────────┘

---

## 🛠️ Stack Technique

* **Langage & Environnement** : Python 3.11, Virtualenv, Docker
* **Gestion des Données & SQL** : DuckDB, Pandas, NumPy, PyArrow/Parquet
* **Machine Learning & Deep Learning** : PyTorch (MLP), Scikit-Learn (HDBSCAN), Optuna, SHAP
* **Visualisation & UI** : Streamlit, Folium / Plotly
* **MLOps & DevOps** : Git, DVC (Data Version Control), GitHub Actions (CI/CD avec Pytest)

---

## 📂 Structure du Répertoire

```text
├── .dvc/                   # Configuration DVC
├── .github/workflows/      # Pipelines CI/CD (ci-cd.yml)
├── data/                   # Géré par DVC (exclu du suivi Git direct)
│   ├── raw/                # Données brutes BDIFF et Météo
│   └── processed/          # Matrices filtrées, parquet, base DuckDB
├── models/                 # Modèles entraînés et scalers (.pt, .joblib) - Géré par DVC
├── src/                    # Scripts sources du pipeline
│   ├── ingestion/          # Ingestion et jointures DuckDB
│   ├── features/           # Encodage cyclique, HDBSCAN, Lags
│   └── models/             # Entraînement PyTorch et optimisation Optuna
├── tests/                  # Tests unitaires automatisés (pytest)
├── app.py                  # Application Streamlit principale
├── sql_consolidation.py    # Script d'ingestion SQL DuckDB
├── CADRAGE_TECHNIQUE.md    # Note de cadrage RNCP 40573 & choix d'architecture
├── Dockerfile              # Fichier de conteneurisation de l'application
├── .dockerignore           # Context de build optimisé (<1MB)
├── data.dvc                # Pointeur DVC pour le répertoire data
├── models.dvc              # Pointeur DVC pour le répertoire models
└── requirements.txt        # Dépendances Python du projet

---

🚀 Installation et Démarrage Rapide
Prérequis
  - Git

  - Python 3.11+

  - Docker Desktop (optionnel pour l'exécution conteneurisée)

---

1. Clonage du Projet

git clone [https://github.com/votre-user/projet_plateforme.git](https://github.com/votre-user/projet_plateforme.git)
cd projet_plateforme

---

2. Configuration de l'Environnement Virtuel

python -m venv .venv
# Sur Windows Command Prompt :
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

B. Lancement de l'Application Streamlit en Local

streamlit run app.py

L'interface sera accessible sur http://localhost:8501.

---

🐳 Déploiement via Docker

1. Construction de l'Image Docker
L'image s'appuie sur un fichier .dockerignore strict garantissant un contexte de build ultra-léger (<1Mo).

docker build -t fire-risk-app:latest .

2. Lancement du Conteneur

docker run -d -p 8501:8501 --name fire_risk_container fire-risk-app:latest

Accédez à l'application via http://localhost:8501.

🧪 Qualité du Code & CI/CD

Les tests unitaires vérifient la validité des sorties de features, le non-chevauchement des dates dans le split temporel et la conformité du modèle PyTorch.

Pour exécuter les tests localement :

pytest

Le pipeline GitHub Actions (.github/workflows/ci-cd.yml) déclenche automatiquement l'exécution de pytest à chaque push ou pull_request sur la branche master.

📊 Méthodologie & Performances ML

1. Negative Sampling : Génération des couples (commune, date) sans départ de feu pour équilibrer l'apprentissage ($Cible=0$).

2. Features Spatio-Temporelles :

 - Spatial : Clustering HDBSCAN sur les coordonnées centroïdes (6 clusters identifiés via la métrique Haversine).
 - Temporel : Encodage cyclique ($\sin/\cos$) du jour de l'année et du mois, indicateur is_weekend.
 - Sécheresse Cumulée : Lags et moyennes glissantes à 7, 15 et 30 jours sur les précipitations et températures.
3. Evaluation sur Split Chronologique :
 - Train : 2006 – 2021 | Test : 2022 – 2024 (Prévention absolue du data leakage).
 - Rappel (Classe Feu) : ~82% | AUPRC : ~0.48.

📄 Licence et Documentation RNCP

Pour consulter la note d'architecture complète, la justification des compétences et la matrice des risques EBIOS, référez-vous au fichier CADRAGE_TECHNIQUE.md.