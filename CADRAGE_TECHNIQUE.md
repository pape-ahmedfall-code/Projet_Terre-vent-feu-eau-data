# Note de Cadrage Technique & MLOps — Plateforme de Prédiction du Risque d'Incendie
**Référentiel RNCP 40573 — Alignment Datasets, Machine Learning & Architecture SI**

---

## 1. Contexte Stratégique & Analyse du Besoin Métier (RNCP C10, C14, C15)

Face à l'intensification du dérèglement climatique, la **Direction Générale de la Sécurité Civile et de la Gestion des Crises (DGSCGC)** nécessite un outil d'anticipation du risque d'incendie de forêt à l'échelle communale. 

L'objectif principal est de passer d'une logique de réaction post-déclaration à une **stratégie de prévention prédictive géo-temporelle**.

### Objectifs Clés
* **Consolidation multi-sources** : Harmonisation des historiques d'incendies (BDIFF) et des données météorologiques historiques (SIM2/MétéoNet).
* **Modélisation non-linéaire** : Prédiction de la probabilité d'occurrence d'incendie ($Target \in \{0, 1\}$) sur l'ensemble du territoire français.
* **Explicabilité métier** : Fournir aux décideurs opérationnels les facteurs de risque dominants (vent, humidité, sécheresse cumulée) via l'approche SHAP.

---

## 2. Architecture du Système d'Information & Cartographie des Flux (RNCP C4, C21, C22)

Le pipeline d'ingénierie suit un flux de données strict pour garantir la séparation des responsabilités et éviter le *data leakage*.

[BDIFF (CSV)] --------┐
├──> [Ingestion DuckDB] ──> [Negative Sampling & Feature Eng.] ──> [HDBSCAN Spatial]
[Météo SIM2 (Parquet)] ┘                                         
---
                           │
▼
[Streamlit Frontend] <── [Inférence PyTorch + SHAP] <── [Modèle Entraîné (MLP)] <── [Optuna Tuning & Split Temporel]

---

### Cartographie du SI et Matrice des Flux (RNCP C4)
1. **Ingestion & Data Lake** : Chargement des fichiers Parquet/CSV via **DuckDB** pour des jointures SQL massives (69M+ de lignes) sur la clé composite `(code_insee, date)`.
2. **Feature Store / Processing** : Calcul des variables cycliques, des fenêtres glissantes (lags) et de l'attribution des clusters HDBSCAN.
3. **MLOps Core** : Suivi des artefacts par **DVC**, versionnage de code par **Git**, et conteneurisation légère sous **Docker**.

---

## 3. Matrice de Risques Projet & Mitigation (RNCP C17)

| Risque Identifié | Impact | Probabilité | Stratégie de Mitigation |
| :--- | :---: | :---: | :--- |
| **Data Leakage Temporel** | Majeur | Forte | Strict split chronologique (Train: 2006–2021, Test: 2022–2024). Interdiction du K-Fold classique. |
| **Déséquilibre de Classe Extreme** | Majeur | Forte | Utilisation d'une fonction de perte **Focal Loss** et évaluation prioritaire sur **AUPRC / Rappel**. |
| **Biais d'Interpolation Météo** | Moyen | Moyenne | Attribution spatiale par métrique Haversine / Plus Proche Voisin par rapport au centroïde INSEE. |
| **Dérive du Modèle (Data Drift)** | Moyen | Moyenne | MLOps avec suivi DVC des datasets bruts et ré-entraînement versionné. |

---

## 4. Choix de Modélisation & Architecture ML/DL (RNCP C29, C31, C32)

### A. Clustering Spatial (Contagion Géographique)
Un algorithme **HDBSCAN** (métrique Haversine sur coordonnées en radians) est exécuté sur les centroïdes des communes (`latitude_centre`, `longitude_centre`).
* **Résultat** : Identification de **6 clusters géographiques majeurs** représentant les zones à forte densité historique de risques (ex: arc méditerranéen, massif des Landes).
* **Usage** : L'ID de cluster est injecté sous forme de feature catégorielle pour capturer l'effet de contagion spatiale.

### B. Architecture Neural Network (PyTorch MLP)
Pour capturer les interactions complexes et non-linéaires entre la météo et la saisonnalité :
* **Réseau** : Perceptron Multicouche (MLP) avec couches de `Linear`, `BatchNorm1d`, `ReLU` et `Dropout`.
* **Perte** : **Focal Loss** pour pénaliser fortement les erreurs sur les exemples rares (feux positifs) sans sur-échantillonnage artificiel.
* **Hypertuning** : Recherche automatisée des hyperparamètres (*learning rate*, *dropout*, taille des couches) via **Optuna**.

---

## 5. Résultats & Évaluation des Performances (RNCP C26, C29)

Compte tenu de la rareté des départs de feu par rapport aux journées sans feu, l'Accuracy globale est rejetée au profit des métriques axées sur la classe minoritaire.

| Métrique | Valeur Observée | Interprétation Métier |
| :--- | :---: | :--- |
| **Rappel (Recall - Classe Feu)** | **~0.82** | Capture de 82% des départs de feux réels (minimisation des faux négatifs). |
| **Précision (Classe Feu)** | **~0.35** | Niveau contrôlé de fausses alertes opérationnelles. |
| **AUPRC** | **~0.48** | Performance solide sur la courbe Précision-Rappel en contexte déséquilibré. |

---

## 6. Explicabilité Métier avec SHAP (RNCP C32)

L'intégration de la bibliothèque **SHAP (SHapley Additive exPlanations)** au sein du frontend Streamlit permet d'expliquer chaque prédiction individuelle :
* **Facteurs Aggravants Principaux** : Vent moyen (`f_wind_mean`), vitesse des rafales, et lags de sécheresse à 30 jours.
* **Facteurs Atténuants** : Humidité relative élevée et précipitations récentes (lags 7 jours).

---

## 7. Éco-Conception & Ingestion Sécurisée (RNCP C23, C25, C27, C30)

* **Éco-conception (C23)** : Ingestion SQL optimisée par **DuckDB** réduisant la consommation de mémoire RAM (traitement par blocs). Image Docker optimisée à l'aide d'un `.dockerignore` strict (<1MB de contexte de build).
* **Versionnage DVC (C30)** : Découplage complet entre le code source (Git) et les 25 Go de données brutes/modèles (`data.dvc`, `models.dvc`).
* **Qualité & CI/CD (C27)** : Exécution automatisée des tests unitaires (`pytest`) via **GitHub Actions** à chaque *push*.
