# Note de Cadrage Technique & MLOps — Plateforme de Prédiction du Risque d'Incendie
**Référentiel RNCP 40573 — Alignement Datasets, Machine Learning & Architecture SI**

---

## 1. Contexte Stratégique & Analyse du Besoin Métier 

Face à l'intensification du dérèglement climatique, la **Direction Générale de la Sécurité Civile et de la Gestion des Crises (DGSCGC)** nécessite un outil d'anticipation du risque d'incendie de forêt à l'échelle communale. 

L'objectif principal est de passer d'une logique de réaction post-déclaration à une **stratégie de prévention prédictive géo-temporelle**.

### Objectifs Clés
* **Consolidation multi-sources** : Harmonisation des historiques d'incendies (BDIFF) et des données météorologiques historiques (SIM2/MétéoNet).
* **Modélisation non-linéaire** : Prédiction de la probabilité d'occurrence d'incendie sur l'ensemble du territoire français.
* **Explicabilité métier** : Fournir aux décideurs opérationnels les facteurs de risque dominants (vent, humidité, sécheresse cumulée) via l'approche SHAP.

---

## 2. Architecture du Système d'Information & Cartographie des Flux 

Le pipeline d'ingénierie suit un flux de données strict pour garantir la séparation des responsabilités et éviter le *data leakage*.

[BDIFF (CSV)] --------┐
                      ├──> [Ingestion DuckDB] ──> [Negative Sampling & Feature Eng.] ──> [HDBSCAN Spatial]
[Météo SIM2 (Parquet)] ┘                                         
                                                                 │
                                                                 ▼
[Streamlit Frontend] <── [Inférence PyTorch + SHAP] <── [Modèle Entraîné (MLP)] <── [Optuna Tuning & Split Temporel]

---

### Cartographie du SI et Matrice des Flux 
1. **Ingestion & Data Lake** : Chargement des fichiers Parquet/CSV via **DuckDB** pour des jointures SQL massives (69M+ de lignes) sur la clé composite `(code_insee, date)`.
2. **Feature Store / Processing** : Calcul des variables cycliques, des fenêtres glissantes (lags) et de l'attribution des clusters HDBSCAN.
3. **MLOps Core** : Suivi des artefacts par **DVC**, versionnage de code par **Git**, et conteneurisation légère sous **Docker**.

---

## 3. Matrice de Risques Projet & Mitigation

| Risque Identifié | Impact | Probabilité | Stratégie de Mitigation |
| :--- | :---: | :---: | :--- |
| **Data Leakage Temporel** | Majeur | Forte | Strict split chronologique (Train: 2006–2021, Test: 2022–2024). Interdiction du K-Fold classique. |
| **Déséquilibre de Classe Extrême** | Majeur | Forte | Utilisation d'une fonction de perte **Focal Loss** et fixation du seuil de décision métier à **0,5000**. |
| **Biais d'Interpolation Météo** | Moyen | Moyenne | Attribution spatiale par métrique Haversine / Plus Proche Voisin par rapport au centroïde INSEE. |
| **Dérive du Modèle (Data Drift)** | Moyen | Moyenne | MLOps avec suivi DVC des datasets bruts/modèles et tests automatisés sur les schémas d'entrée. |

---

## 4. Choix de Modélisation & Architecture ML/DL 

### A. Clustering Spatial (Contagion Géographique)
Un algorithme **HDBSCAN** (métrique Haversine sur coordonnées en radians) est exécuté sur les centroïdes des communes (`latitude_centre`, `longitude_centre`).
* **Résultat** : Identification de **6 clusters géographiques majeurs** représentant les zones à forte densité historique de risques (ex: arc méditerranéen, massif des Landes).
* **Usage** : L'ID de cluster est injecté sous forme de feature catégorielle pour capturer l'effet de contagion spatiale.

### B. Architecture Neural Network (PyTorch MLP)
Pour capturer les interactions complexes et non-linéaires entre la météo et la saisonnalité :
* **Réseau** : Perceptron Multicouche (MLP) à 3 couches cachées avec **53 nœuds en entrée**, des couches cachées de **102 et 126 nœuds**, activation **Sigmoid** finale, et un taux de **Dropout de 0,337**.
* **Perte & Seuil** : **Focal Loss** pour pénaliser les erreurs sur la classe minoritaire et seuil de décision fixe réglé à **0,5000**.
* **Hypertuning & Métadonnées** : Recherche d'hyperparamètres via **Optuna** avec export dynamique des configurations dans `model_metadata.json`.

---

## 5. Explicabilité Métier avec SHAP 

L'intégration de la bibliothèque **SHAP (SHapley Additive exPlanations)** au sein du frontend Streamlit permet d'expliquer chaque prédiction individuelle :
* **Facteurs Aggravants Principaux** : Vent moyen (`f_wind_mean`), vitesse des rafales, et lags de sécheresse à 30 jours.
* **Facteurs Atténuants** : Humidité relative élevée et précipitations récentes (lags 7 jours).

---

## 6. Éco-Conception & Ingestion Sécurisée 

* **Éco-conception** : Ingestion SQL optimisée par **DuckDB** réduisant la consommation de mémoire RAM (traitement par blocs). Image Docker optimisée via PyTorch CPU (`--extra-index-url`) et un `.dockerignore` strict en liste blanche (**119 ko de contexte de build**).
* **Versionnage DVC** : Découplage complet entre le code source (Git) et les volumes de données/artefacts (`data.dvc`, `models.dvc`).
* **Qualité & CI/CD** : Exécution automatisée des tests unitaires (`pytest`) via **GitHub Actions** pour vérifier la cohérence des dimensions et du pipeline à chaque *push*.