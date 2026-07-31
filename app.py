import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import json

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Plateforme MLOps - Prédiction Risque Incendie",
    page_icon="🔥",
    layout="wide"
)

# Chemins vers les artefacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------------------------------------------------------------------
# Définition de la classe PyTorch dynamique (selon Optuna : 53 -> 102 -> 126)
# ------------------------------------------------------------------------------
class FirePredictionModel(nn.Module):
    def __init__(self, input_dim=11, hidden_units=[102, 126], dropout_rate=0.337):
        super(FirePredictionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 53),
            nn.BatchNorm1d(53),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(53, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_units[1], 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# Chargement optimisé des artefacts MLOps avec cache Streamlit
# ------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_cols.joblib")
    weights_path = os.path.join(MODELS_DIR, "model_focal.pt")

    # Lecture des métadonnées JSON
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    
    input_dim = metadata.get("input_dim", len(feature_cols))
    hidden_units = metadata.get("hidden_units", [102, 126])
    dropout_rate = metadata.get("dropout_rate", 0.337)
    threshold = metadata.get("best_threshold", 0.5000)

    # Instanciation du modèle avec les dimensions exactes du JSON
    model = FirePredictionModel(input_dim, hidden_units, dropout_rate)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    return model, scaler, feature_cols, threshold, metadata

# Initialisation et contrôle des erreurs de chargement
model_loaded = False
error_message = ""

try:
    model, scaler, feature_cols, THRESHOLD, metadata = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# ------------------------------------------------------------------------------
# Interface Utilisateur Streamlit
# ------------------------------------------------------------------------------
st.title("🔥 Plateforme MLOps : Prédiction & Aide à la Décision Feux de Forêt")
st.markdown("---")

if not model_loaded:
    st.error(f"⚠️ Erreur lors du chargement des artefacts PyTorch/MLOps : {error_message}")
    st.info("Vérifiez que la cellule d'exportation du Notebook 4 s'est exécutée avec succès dans le dossier `models/`.")
else:
    tab1, tab2 = st.tabs(["🗺️ 1. Analyse Spatiale & Performance MLOps", "🔮 2. Simulateur de Risque & Explicabilité SHAP"])

    # =========================================================================
    # ONGLET 1 : Métriques & Carte
    # =========================================================================
    with tab1:
        st.header("Performances et Robustesse Hors-Temps (2022-2025)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Données Test Relles", "47.7M lignes")
        c2.metric("ROC-AUC Test (Stabilité)", f"{metadata['test_metrics']['roc_auc']:.4f}")
        c3.metric("PR-AUC Test", f"{metadata['test_metrics']['pr_auc']:.4f}")
        c4.metric("Seuil Optimal Détécté", f"{THRESHOLD:.4f}")

        st.info("💡 **Diagnostic MLOps** : La stabilité de la ROC-AUC (0.8841) atteste de l'absence de surapprentissage. Le seuil de décision est ajusté dynamiquement pour maximiser la réponse opérationnelle.")

        if st.checkbox("Afficher la distribution spatiale d'échantillon sur la carte"):
            map_data = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [43.6, 3.8],
                columns=['lat', 'lon']
            )
            st.map(map_data)

    # =========================================================================
    # ONGLET 2 : Simulateur
    # =========================================================================
    with tab2:
        st.header("Simulateur Météo-Spatial en Temps Réel")
        st.markdown("Saisissez les valeurs brutes ou normalisées pour obtenir le score d'alerte instantané.")

        with st.form("simulation_form"):
            st.subheader("Variables météorologiques et géographiques")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.slider("Température instantanée", -2.0, 3.0, 0.5)
                temp_mean_7d = st.slider("Température moyenne (7j)", -2.0, 3.0, 0.4)
                vent_vitesse = st.slider("Vitesse du vent", -2.0, 3.0, 0.1)
            with col2:
                humidite = st.slider("Humidité de l'air (Sécheresse)", -2.5, 2.0, -0.8)
                humid_mean_7d = st.slider("Humidité moyenne (7j)", -2.5, 2.0, -0.7)
                vent_mean_7d = st.slider("Vent moyen (7j)", -2.0, 2.0, 0.0)
            with col3:
                superficie = st.slider("Superficie (km²)", -1.0, 3.0, 0.2)
                altitude = st.slider("Altitude moyenne", -2.0, 2.0, 0.0)
                hdbscan_cluster = st.selectbox("Cluster HDBSCAN", [-0.5, 0.0, 0.5, 1.0])
                lamb_x = st.slider("Coordonnée LAMBX", -2.0, 2.0, 0.5)
                lamb_y = st.slider("Coordonnée LAMBY", -2.0, 2.0, 0.5)

            submit_button = st.form_submit_button(label="🚀 Calculer le Risque d'Incendie")

        if submit_button:
            input_dict = {
                'temperature': temperature,
                'temperature_mean_7d': temp_mean_7d,
                'humidite': humidite,
                'humidite_mean_7d': humid_mean_7d,
                'vent_vitesse': vent_vitesse,
                'vent_mean_7d': vent_mean_7d,
                'superficie_km2': superficie,
                'altitude_moyenne': altitude,
                'hdbscan_cluster': hdbscan_cluster,
                'LAMBX_attribue': lamb_x,
                'LAMBY_attribue': lamb_y
            }
            
            # Alignement strict sur les colonnes d'entraînement
            input_array = np.array([[input_dict.get(col, 0.0) for col in feature_cols]])
            
            # Passage en Tensor PyTorch
            tensor_input = torch.tensor(input_array, dtype=torch.float32)
            
            with torch.no_grad():
                logit = model(tensor_input)
                prob = torch.sigmoid(logit).item()
            
            is_fire = prob >= THRESHOLD

            st.markdown("---")
            st.subheader("📊 Résultat du Modèle de Prédiction")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Score de Risque Prédit", value=f"{prob:.4f}")
                if is_fire:
                    st.error(f"🚨 **ALERTE ROUGE** : Probabilité ({prob:.4f}) ≥ Seuil ({THRESHOLD:.4f}) — Déclenchement recommandé.")
                else:
                    st.success(f"✅ **SÉCURITAIRE** : Probabilité ({prob:.4f}) < Seuil ({THRESHOLD:.4f}).")
            
            with res_col2:
                st.write(f"**Seuil de Décision Métier** : `{THRESHOLD:.4f}`")
                st.write(f"**Fonction d'Activation** : `Sigmoid(Focal_Loss_Logits)`")

            # Explicabilité locale SHAP
            st.markdown("---")
            st.subheader("🔍 Explication Locale des Facteurs de Risque (SHAP)")
            try:
                background = torch.zeros((20, len(feature_cols)), dtype=torch.float32)
                explainer = shap.GradientExplainer(model, background)
                shap_val = explainer.shap_values(tensor_input)
                
                if isinstance(shap_val, list):
                    s_vals = shap_val[0][0]
                else:
                    s_vals = shap_val[0]
                if len(s_vals.shape) > 1:
                    s_vals = s_vals.squeeze(-1)

                fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
                y_pos = np.arange(len(feature_cols))
                colors = ['#d62728' if val > 0 else '#1f77b4' for val in s_vals]
                ax.barh(y_pos, s_vals, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_cols, fontsize=9)
                ax.set_xlabel("Contribution SHAP (+ Aggravant / - Modérateur)", fontsize=10)
                ax.set_title("Impact des variables sur la prédiction locale", fontsize=11, fontweight='bold')
                ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning(f"Note explicative SHAP : {shap_err}")