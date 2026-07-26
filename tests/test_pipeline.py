import os
import pytest
import torch
import joblib
import numpy as np

# Définition de la structure exacte validée
class FirePredictionModel(torch.nn.Module):
    def __init__(self, input_dim=11):
        super(FirePredictionModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 53),
            torch.nn.BatchNorm1d(53),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(53, 119),
            torch.nn.BatchNorm1d(119),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(119, 1)
        )

    def forward(self, x):
        return self.net(x)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def test_artefacts_existence():
    """Vérifie que tous les fichiers requis sont bien présents dans models/"""
    assert os.path.exists(os.path.join(MODELS_DIR, "model_focal.pt")), "Modèle PyTorch manquant"
    assert os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib")), "Scaler manquant"
    assert os.path.exists(os.path.join(MODELS_DIR, "feature_cols.joblib")), "Feature columns manquantes"

def test_model_inference():
    """Vérifie l'inférence du modèle PyTorch et le format de sortie"""
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    
    model = FirePredictionModel(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "model_focal.pt"), map_location=torch.device('cpu')))
    model.eval()

    # Création d'un vecteur factice aux bonnes dimensions (11 variables)
    dummy_input = np.zeros((1, len(feature_cols)))
    scaled_input = scaler.transform(dummy_input)
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)

    with torch.no_grad():
        logit = model(tensor_input)
        prob = torch.sigmoid(logit).item()

    # La probabilité doit être un float compris stricto sensu entre 0 et 1
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0