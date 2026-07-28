import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Plateforme MLOps - Prédiction Risque Incendie",
    page_icon="🔥",
    layout="wide"
)

# Chemin du dossier des artefacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Définition de la classe PyTorch correspondant à l'architecture entraînée

class FirePredictionModel(nn.Module):
    def __init__(self, input_dim=11):
        super(FirePredictionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 53),       # 53 neurones (selon le checkpoint Optuna)
            nn.BatchNorm1d(53),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(53, 119),             # 119 neurones
            nn.BatchNorm1d(119),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(119, 1)               # Couche de sortie
        )

    def forward(self, x):
        return self.net(x)

# 1. Chargement des artefacts avec cache Streamlit
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    input_dim = len(feature_cols)

    model = FirePredictionModel(input_dim)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_focal.pt"), map_location=torch.device('cpu')))
    model.eval()
    return model, scaler, feature_cols

# Initialisation explicite des variables de contrôle
model = None
scaler = None
feature_cols = None
model_loaded = False
error_message = ""

# Tentative de chargement
try:
    model, scaler, feature_cols = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)

# En-tête principal
st.title("🔥 Plateforme MLOps : Prédiction & Aide à la Décision Feux de Forêt (2022-2025)")
st.markdown("---")

if not model_loaded:
    st.error(f"⚠️ Erreur lors du chargement des artefacts : {error_message}")
else:
    # Création des onglets principaux
    tab1, tab2 = st.tabs(["🗺️ 1. Carte & Analyse Spatiale (HDBSCAN)", "🔮 2. Simulateur de Risque & Explication SHAP"])

    # =========================================================================
    # ONGLET 1 : Carte et Analyse Spatiale (HDBSCAN)
    # =========================================================================
    with tab1:
        st.header("Analyse Spatiale et Vulnérabilité du Territoire")
        st.markdown("""
        Cet onglet présente la cartographie des zones à risque basées sur le clustering spatial **HDBSCAN** 
        et l'historique des observations géographiques (coordonnées Lambert & superficies).
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Observations Globales Analysées", "47.7 Millions")
        col2.metric("Seuil Métier Appliqué (Recall ≥ 70%)", "0.3309")
        col3.metric("ROC-AUC Global (Test 2022-2025)", "0.8854")
        
        st.info("💡 **Note métier** : Les clusters HDBSCAN permettent d'isoler des zones homogènes soumises à des régimes de topographie et de vulnérabilité similaires.")
        
        if st.checkbox("Afficher un échantillon de la distribution spatiale des clusters"):
            map_data = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [43.6, 3.8],
                columns=['lat', 'lon']
            )
            st.map(map_data)

    # =========================================================================
    # ONGLET 2 : Simulateur de Risque & Explication SHAP
    # =========================================================================
    with tab2:
        st.header("Simulateur de Risque Météo-Spatial par Commune")
        st.markdown("Modifiez les paramètres météorologiques et géographiques pour simuler le score de risque en temps réel.")

        with st.form("simulation_form"):
            st.subheader("Paramètres d'entrée du modèle")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                temperature = st.slider("Température (°C / Normalisée)", -2.0, 3.0, 0.5)
                temp_mean_7d = st.slider("Température Moyenne (7j)", -2.0, 3.0, 0.4)
                vent_vitesse = st.slider("Vitesse du Vent", -2.0, 3.0, 0.1)
            with c2:
                humidite = st.slider("Humidité de l'Air (Sécheresse)", -2.5, 2.0, -0.8)
                humid_mean_7d = st.slider("Humidité Moyenne (7j)", -2.5, 2.0, -0.7)
                vent_mean_7d = st.slider("Vent Moyen (7j)", -2.0, 2.0, 0.0)
            with c3:
                superficie = st.slider("Superficie de la zone (km²)", -1.0, 3.0, 0.2)
                altitude = st.slider("Altitude Moyenne", -2.0, 2.0, 0.0)
                hdbscan_cluster = st.selectbox("Cluster HDBSCAN", [-0.5, 0.0, 0.5, 1.0])
                lamb_x = st.slider("Coordonnée LAMBX", -2.0, 2.0, 0.5)
                lamb_y = st.slider("Coordonnée LAMBY", -2.0, 2.0, 0.5)

            submit_button = st.form_submit_button(label="Lancer la Prédiction & l'Explication SHAP")

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
            
            input_vector = np.array([[input_dict.get(col, 0.0) for col in feature_cols]])
            input_scaled = scaler.transform(input_vector)
            
            tensor_input = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                logit = model(tensor_input)
                prob = torch.sigmoid(logit).item()
                
            THRESHOLD = 0.3309
            is_fire = prob >= THRESHOLD
            
            st.markdown("---")
            st.subheader("📊 Résultats de la Simulation")
            
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Probabilité Prédite de Feu", value=f"{prob:.4f}")
                if is_fire:
                    st.error("🚨 **ALERTE ROUGE** : Risque d'incendie élevé détecté (Déclenchement alerte secours).")
                else:
                    st.success("✅ **SÉCURITAIRE** : Risque faible en deçà du seuil critique.")
            
            with res_col2:
                st.write(f"**Seuil de décision métier** : {THRESHOLD}")
                st.write(f"**Statut** : {'Alerte Activée' if is_fire else 'Pas d alerte'}")

            # Calcul SHAP Local pour l'explication en direct
            st.markdown("---")
            st.subheader("🔍 Explication Locale (Valeurs SHAP de la Prédiction)")
            try:
                background = torch.zeros((10, len(feature_cols)), dtype=torch.float32)
                explainer = shap.DeepExplainer(model, background)
                shap_val = explainer.shap_values(tensor_input)
                
                if isinstance(shap_val, list):
                    s_vals = shap_val[0][0]
                else:
                    s_vals = shap_val[0]
                if len(s_vals.shape) > 1:
                    s_vals = s_vals.squeeze(-1)

                fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
                y_pos = np.arange(len(feature_cols))
                colors = ['red' if val > 0 else 'blue' for val in s_vals]
                ax.barh(y_pos, s_vals, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_cols, fontsize=9)
                ax.set_xlabel("Impact SHAP (Pousse vers le Feu [+] ou le Calme [-])", fontsize=10)
                ax.set_title("Facteurs contributifs pour cette commune", fontsize=11, fontweight='bold')
                ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning(f"Explication locale SHAP désactivée : {shap_err}")