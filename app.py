# app.py
# Lance avec:  streamlit run app.py

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from joblib import load

MODEL_PATH = "Cognitive_1.pth"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.json"

st.set_page_config(page_title="CognitiveTrack – Démo Alzheimer", page_icon="🧠", layout="centered")
st.title("CognitiveTrack – Démo de prédiction (Alzheimer)")

st.markdown("""
Cette démonstration illustre un **modèle tabulaire** prédictif entraîné sur un jeu de données synthétique.
L'outil **ne fournit aucun diagnostic médical**. Il montre uniquement comment un modèle peut estimer une probabilité à partir de variables simples.
""")

# Chargement des artefacts
@st.cache_resource
def load_artifacts():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        features = json.load(f)["features"]
    scaler = load(SCALER_PATH)

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    model = MLP(input_dim=len(features), num_classes=2)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, scaler, features

try:
    model, scaler, features = load_artifacts()
except Exception as e:
    st.error(f"Impossible de charger le modèle ou le scaler: {e}")
    st.stop()

st.subheader("Entrer les informations")

# Interface des entrées utilisateur, dans l'ordre exact des features
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=110, value=70)
    Gender = st.selectbox("Genre (0=H, 1=F)", options=[0,1], index=0)
    BMI = st.number_input("BMI (Indice de masse corporelle)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    Smoking = st.selectbox("Fumeur (0=Non, 1=Oui)", options=[0,1], index=0)
    AlcoholConsumption = st.slider("Consommation d'alcool (0–10)", min_value=0, max_value=10, value=1)

with col2:
    PhysicalActivity = st.slider("Activité physique (0–10)", min_value=0, max_value=10, value=3)
    SleepQuality = st.slider("Qualité du sommeil (0–10)", min_value=0, max_value=10, value=4)
    FamilyHistoryAlzheimers = st.selectbox("Antécédents familiaux (0=Non, 1=Oui)", options=[0,1], index=0)
    MMSE = st.slider("MMSE (0–30)", min_value=0, max_value=30, value=28)
    FunctionalAssessment = st.slider("Autonomie fonctionnelle (0–30)", min_value=0, max_value=30, value=10)
    ADL = st.slider("ADL (Activités quotidiennes, 0–10)", min_value=0, max_value=10, value=5)

# Assemblage des features dans l'ordre attendu
input_row = np.array([[Age, Gender, BMI, Smoking, AlcoholConsumption,
                       PhysicalActivity, SleepQuality, FamilyHistoryAlzheimers,
                       MMSE, FunctionalAssessment, ADL]], dtype=np.float32)

# Normalisation via le scaler appris à l'entraînement
try:
    input_scaled = scaler.transform(input_row).astype(np.float32)
except Exception as e:
    st.error(f"Erreur de normalisation des données: {e}")
    st.stop()

# Prédiction
if st.button("Estimer la probabilité"):
    x_t = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).numpy()[0]
        pred = int(np.argmax(probs))

    st.markdown("### Résultat")
    st.write(f"Classe prédite: **{pred}** (0 = sain, 1 = Alzheimer)")
    st.write(f"Probabilité 'Sain' (classe 0): {probs[0]:.3f}")
    st.write(f"Probabilité 'Alzheimer' (classe 1): {probs[1]:.3f}")

    # Barre de progression simple sur la classe 1
    st.progress(float(probs[1]))

st.divider()
st.caption("Cette démo est fournie à des fins pédagogiques. Elle ne constitue en aucun cas un dispositif médical ni un avis clinique.")
