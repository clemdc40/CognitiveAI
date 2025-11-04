import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================================
# Chargement du modèle et du préprocesseur
# ================================
@st.cache_resource
def load_models():
    bundle = torch.load("alz_suivi_hybride_bundle.pth", map_location="cpu")
    preproc = joblib.load("alz_preprocessor_suivi.pkl")
    gbm = joblib.load("alz_suivi_gbm.joblib")

    # Reconstruction du réseau neuronal
    class AlzheimerMultiHead(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.backbone = torch.nn.Sequential(
                torch.nn.Linear(input_size, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
            )
            self.medical_head = torch.nn.Linear(64, 1)
            self.psy_head = torch.nn.Linear(64, 1)
            self.behav_head = torch.nn.Linear(64, 1)
            self.global_mixer = torch.nn.Sequential(torch.nn.Linear(3, 1))

        def forward(self, x):
            h = self.backbone(x)
            med = self.medical_head(h)
            psy = self.psy_head(h)
            beh = self.behav_head(h)
            med_p = torch.sigmoid(med)
            psy_p = torch.sigmoid(psy)
            beh_p = torch.sigmoid(beh)
            mix_in = torch.cat([med_p, psy_p, beh_p], dim=1)
            mix = self.global_mixer(mix_in)
            return med, psy, beh, mix

    model = AlzheimerMultiHead(bundle["input_dim"])
    model.load_state_dict(bundle["nn_state_dict"])
    model.eval()

    temps = {
        "T_med": torch.tensor(bundle["T_med"]),
        "T_psy": torch.tensor(bundle["T_psy"]),
        "T_beh": torch.tensor(bundle["T_beh"]),
        "T_glb": torch.tensor(bundle["T_glb"]),
    }

    return model, preproc, gbm, bundle, temps


# ================================
# Prédiction patient
# ================================
def predict_patient(patient_dict, model, preproc, gbm, temps, threshold):
    df = pd.DataFrame([patient_dict])
    Xp = preproc.transform(df)
    xt = torch.tensor(Xp.toarray() if hasattr(Xp, "toarray") else Xp, dtype=torch.float32)

    with torch.no_grad():
        med, psy, beh, glb = model(xt)
        med_p = torch.sigmoid(med / temps["T_med"]).item()
        psy_p = torch.sigmoid(psy / temps["T_psy"]).item()
        beh_p = torch.sigmoid(beh / temps["T_beh"]).item()
        glb_p = torch.sigmoid(glb / temps["T_glb"]).item()

    X_ext = np.column_stack([Xp.toarray() if hasattr(Xp, "toarray") else Xp, [glb_p], [med_p], [psy_p], [beh_p]])
    final_prob = gbm.predict_proba(X_ext)[:, 1][0]
    label = "Élevé" if final_prob >= threshold else ("Modéré" if final_prob >= 0.3 else "Faible")

    return {
        "final_prob": final_prob,
        "label": label,
        "medical": med_p,
        "psy": psy_p,
        "behavior": beh_p,
        "global": glb_p,
    }


# ================================
# Interface Streamlit
# ================================
st.set_page_config(page_title="Suivi Alzheimer", page_icon="🧠", layout="centered")

st.title("🧠 Outil de Suivi Alzheimer (Hybride NN + Boosting)")
st.write("Cet outil estime le **risque global d'Alzheimer** à partir des facteurs médicaux, psychologiques et comportementaux d’un patient.")

model, preproc, gbm, bundle, temps = load_models()
threshold = bundle["threshold"]

# ----------- Saisie patient -----------
st.header("📋 Informations du patient")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge", 40, 100, 74)
    gender = st.selectbox("Genre", ["Male", "Female"])
    bmi = st.number_input("IMC", 15.0, 40.0, 26.5)
    activity = st.selectbox("Niveau d'activité physique", ["Low", "Medium", "High"])
    smoking = st.selectbox("Statut tabagique", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Consommation d'alcool", ["Never", "Occasionally", "Regularly"])
with col2:
    diabetes = st.selectbox("Diabète", ["No", "Yes"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    cholesterol = st.selectbox("Niveau de cholestérol", ["Normal", "High"])
    cognitive = st.slider("Score cognitif", 0, 100, 45)
    depression = st.selectbox("Niveau de dépression", ["Low", "Moderate", "High"])
    sleep = st.selectbox("Qualité du sommeil", ["Poor", "Average", "Good"])
    social = st.selectbox("Engagement social", ["Low", "Medium", "High"])
    stress = st.selectbox("Niveau de stress", ["Low", "Medium", "High"])

patient_dict = {
    "Age": age,
    "Gender": gender,
    "BMI": bmi,
    "Physical Activity Level": activity,
    "Smoking Status": smoking,
    "Alcohol Consumption": alcohol,
    "Diabetes": diabetes,
    "Hypertension": hypertension,
    "Cholesterol Level": cholesterol,
    "Cognitive Test Score": cognitive,
    "Depression Level": depression,
    "Sleep Quality": sleep,
    "Social Engagement Level": social,
    "Stress Levels": stress,
}

if st.button("🔍 Calculer le risque"):
    res = predict_patient(patient_dict, model, preproc, gbm, temps, threshold)
    prob = res["final_prob"] * 100

    st.markdown("---")
    st.subheader("📊 Résultats du patient")
    color = {"Faible": "🟢", "Modéré": "🟡", "Élevé": "🔴"}[res["label"]]
    st.markdown(f"### {color} Risque {res['label']} ({prob:.1f}%)")
    st.caption(f"Seuil de décision : {threshold*100:.1f}%")

    # Jauges sous-scores
    st.write("#### Sous-scores (réseau neuronal calibré)")
    c1, c2, c3 = st.columns(3)
    c1.metric("🩺 Médical", f"{res['medical']*100:.1f}%")
    c2.metric("🧠 Psychologique", f"{res['psy']*100:.1f}%")
    c3.metric("🏃 Comportemental", f"{res['behavior']*100:.1f}%")

    # Détails
    st.write("#### Facteurs observés :")
    df_factors = pd.DataFrame(list(patient_dict.items()), columns=["Facteur", "Valeur"])
    st.table(df_factors)

    st.success("Rapport généré avec succès ✅")

else:
    st.info("👉 Remplis les informations ci-dessus et clique sur **Calculer le risque**.")
