import os
import json
import pytest
import torch
import torch.nn as nn
import joblib
import numpy as np

# Définition de l'architecture exacte issue d'Optuna (53 -> 102 -> 126)
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

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def test_artefacts_existence():
    """Vérifie que tous les artefacts requis par l'API et la CI/CD sont présents dans models/"""
    assert os.path.exists(os.path.join(MODELS_DIR, "model_focal.pt")), "Modèle PyTorch manquant"
    assert os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib")), "Scaler manquant"
    assert os.path.exists(os.path.join(MODELS_DIR, "feature_cols.joblib")), "Feature columns manquantes"
    assert os.path.exists(os.path.join(MODELS_DIR, "model_metadata.json")), "Fichier de métadonnées JSON manquant"

def test_model_inference():
    """Vérifie le chargement dynamique des poids PyTorch et l'inférence du modèle"""
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    
    # Validation du nombre de variables
    assert len(feature_cols) == metadata["input_dim"], "Discordance sur le nombre de variables d'entrée"

    # Instanciation dynamique du modèle
    model = FirePredictionModel(
        input_dim=metadata["input_dim"],
        hidden_units=metadata["hidden_units"],
        dropout_rate=metadata["dropout_rate"]
    )
    
    # Chargement des poids enregistrés
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_focal.pt"), map_location=torch.device('cpu')))
    model.eval()

    # Création d'un vecteur d'entrée factice
    dummy_input = np.zeros((1, len(feature_cols)))
    tensor_input = torch.tensor(dummy_input, dtype=torch.float32)

    with torch.no_grad():
        logit = model(tensor_input)
        prob = torch.sigmoid(logit).item()

    # Contrôles de validité de la sortie
    assert isinstance(prob, float), "La sortie d'inférence doit être un float"
    assert 0.0 <= prob <= 1.0, "La probabilité doit appartenir à [0, 1]"