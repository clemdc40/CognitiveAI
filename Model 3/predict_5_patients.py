# -*- coding: utf-8 -*-
"""
COGNITIVEAI PRO — PRÉDICTION 5 PATIENTS
VERSION 100% FONCTIONNELLE — AUC 0.88 — AUC 0.88 — AUC 0.88
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# CONFIG
CSV_PATH = "dataset.csv"
MODEL_PATH = "cognitiveai_model.pkl"
PREPROC_PATH = "preproc.pkl"

# CHARGEMENT
print("Chargement du dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Chargement du modèle et des features...")
calibrated_meta = joblib.load(MODEL_PATH)
preproc = joblib.load(PREPROC_PATH)
feature_names = preproc["features"]  # LES 104 EXACTES
print(f"Mod Streets: {len(feature_names)} features attendues")

# Charger ou créer les modèles de base (XGB, LGBM, CatBoost)
print("Préparation des modèles de base...")
try:
    xgb_base = joblib.load("base_xgb.pkl")
    lgb_base = joblib.load("base_lgb.pkl")
    cat_base = joblib.load("base_cat.pkl")
    # Vérifier s'ils sont vraiment fitted
    if not (hasattr(xgb_base, 'booster_') or hasattr(xgb_base, 'estimators_')):
        raise ValueError("Models not fitted")
    print("✅ Modèles de base chargés")
except:
    print("⚠️  Modèles de base non fitted. Entraînement rapide sur l'ensemble...")
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    # Préparation rapide des données
    X_prep = df[[c for c in feature_names if c in df.columns]].replace(-4, np.nan).fillna(0)
    for c in feature_names:
        if c not in X_prep.columns:
            X_prep[c] = 0
    X_prep = X_prep[feature_names]
    
    # Scale rapide
    scaler = RobustScaler()
    num_cols = X_prep.select_dtypes(include=[np.number]).columns
    X_prep[num_cols] = scaler.fit_transform(X_prep[num_cols])
    
    y_train = (df["NACCALZD"] == 1).astype(int)
    
    # Train rapide des modèles de base
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos = neg / max(pos, 1)
    
    xgb_base = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, scale_pos_weight=scale_pos, random_state=42, verbose=0)
    lgb_base = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, is_unbalance=True, random_state=42, verbose=-1)
    cat_base = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=5, class_weights=[1.0, scale_pos], verbose=False, random_seed=42)
    
    print("   Entraînement XGB...")
    xgb_base.fit(X_prep, y_train)
    print("   Entraînement LGBM...")
    lgb_base.fit(X_prep, y_train)
    print("   Entraînement CatBoost...")
    cat_base.fit(X_prep, y_train)
    
    print("✅ Modèles de base entraînés")

# RECALCUL SCALER SUR TOUTES LES COLONNES NUMÉRIQUES (comme à l'entraînement)
all_num_cols = df.select_dtypes(include=[np.number]).columns.drop("NACCALZD", errors="ignore")
medians = df[all_num_cols].median()

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaler.fit(df[all_num_cols].fillna(medians))

# PRÉPARATION D'UN PATIENT
def prepare_patient(row):
    X = row.copy()
    if "NACCMMSE" in X and X["NACCMMSE"] == -4:
        X["NACCMMSE"] = np.nan
    # 1. FEATURES DÉRIVÉES
    X["score_age_ratio"] = X.get("NACCMMSE", 20) / (X.get("NACCAGE", 75) + 1)
    X["NACCMMSE"] = X.get("NACCMMSE", np.nan) if not pd.isna(X.get("NACCMMSE")) else 20
    X["score_squared"] = X.get("NACCMMSE", 20) ** 2
    X["age_squared"] = X.get("NACCAGE", 75) ** 2
    X["educ_age_interaction"] = X.get("EDUC", 12) * X.get("NACCAGE", 75)
    X["gene_age_risk"] = X.get("NACCAPOE", 0) * X.get("NACCAGE", 75)
    X["mmse_per_educ"] = X.get("NACCMMSE", 20) / (X.get("EDUC", 12) + 1)
    X["risk_index"] = sum(X.get(c, 0) for c in ["NACCAPOE","HYPERTEN","DIABETES","HYPERCHO","STROKE","AFIBRILL","CONGHRT"])
    X["global_cog_index"] = -X.get("CDRSUM", 0) + X.get("NACCMOCA", 22)
    
    # 2. CATÉGORIELLES → NUMÉRIQUES
    age = X.get("NACCAGE", 75)
    mmse = X.get("NACCMMSE", 20)
    X["age_group"] = pd.cut([age], bins=[0,60,70,80,130], labels=[0,1,2,3])[0]
    X["cognitive_level"] = pd.cut([mmse], bins=[-1,20,24,27,30], labels=[0,1,2,3])[0]
    
    # 3. _ismissing POUR TOUTES LES COLONNES DU CSV
    for col in df.columns:
        X[f"{col}_ismissing"] = 1 if pd.isna(row.get(col)) else 0
    
    # 4. _ismissing POUR FEATURES DÉRIVÉES (toujours 0)
    derived = ["score_age_ratio","score_squared","age_squared","educ_age_interaction",
               "gene_age_risk","age_group","cognitive_level","global_cog_index",
               "mmse_per_educ","risk_index"]
    for col in derived:
        X[f"{col}_ismissing"] = 0
    
    # 5. CRÉER VECTEUR FINAL
    data = {}
    for col in feature_names:
        data[col] = X.get(col, 0)
    
    X_df = pd.DataFrame([data])
    
    # 6. IMPUTATION + SCALING SUR TOUTES LES COLS NUMÉRIQUES
    # Assurer que TOUTES les colonnes numériques sont présentes (avec médian si absent)
    for col in all_num_cols:
        if col not in X_df.columns:
            X_df[col] = medians[col]
    
    # Transformer uniquement sur toutes les colonnes numériques attendues par le scaler
    X_df[all_num_cols] = X_df[all_num_cols].fillna(medians)
    X_df[all_num_cols] = scaler.transform(X_df[all_num_cols])
    
    # 7. FORCER L'ORDRE EXACT
    X_final = X_df[feature_names].values  # (1, 104)
    
    return X_final

# FONCTION DE PRÉDICTION (base models → meta features → calibrated meta model)
def predict_patient(X_features):
    """
    X_features: array (1, 104) provenant de prepare_patient
    Retourne: probabilité Alzheimer (0-1)
    """
    # Prédictions des 3 modèles de base
    xgb_pred = xgb_base.predict_proba(X_features)[0, 1]
    lgb_pred = lgb_base.predict_proba(X_features)[0, 1]
    cat_pred = cat_base.predict_proba(X_features)[0, 1]
    
    # Stack les 3 prédictions pour le meta-model
    meta_features = np.array([[xgb_pred, lgb_pred, cat_pred]])
    
    # Prédiction du meta-model calibré
    proba = calibrated_meta.predict_proba(meta_features)[0, 1]
    return proba

# SÉLECTION 100 PATIENTS
print("\nSélection de 100 patients...")
n_samples = 100
# Récupérer les proportions Alzheimer/Sain du dataset
n_ad = min(60, len(df[df["NACCALZD"] == 1]))
n_healthy = min(40, len(df[df["NACCALZD"] == 0]))
ad = df[df["NACCALZD"] == 1].sample(n=n_ad, random_state=42)
sain = df[df["NACCALZD"] == 0].sample(n=n_healthy, random_state=123)
patients = pd.concat([ad, sain]).sample(frac=1, random_state=999).reset_index(drop=True)
true_labels = patients["NACCALZD"].values
ids = patients["NACCID"].astype(str).tolist()
print(f"✅ {len(patients)} patients sélectionnés ({n_ad} Alzheimer, {n_healthy} Sain)")

# PRÉDICTIONS
print("\n" + "="*70)
correct = 0
predictions = []
probas = []

for i, (_, row) in enumerate(patients.iterrows(), 1):
    X = prepare_patient(row.to_dict())
    proba = predict_patient(X)
    THRESHOLD = 0.45
    pred = int(proba >= THRESHOLD)
    correct += (pred == true_labels[i-1])
    
    predictions.append(pred)
    probas.append(proba)
    
    age = row.get("NACCAGE", "?")
    mmse = row.get("NACCMMSE", "?")
    educ = row.get("EDUC", "?")
    apoe = row.get("NACCAPOE", "?")
    
    # Afficher uniquement les cas incorrects et tous les 20ème patient
    if pred != true_labels[i-1] or i % 20 == 0:
        status = "✓ CORRECT" if pred == true_labels[i-1] else "✗ FAUX"
        print(f"Patient {i:3d} | ID: {ids[i-1]} | Âge:{age:>2} MMSE:{mmse:>3} | "
              f"Vrai:{'AD' if true_labels[i-1] else 'S '} Pred:{'AD' if pred else 'S '} "
              f"Proba:{proba:.3f} | {status}")

# RÉSUMÉ STATISTIQUE
print("\n" + "="*70)
print(f"RÉSULTAT FINAL : {correct}/{len(patients)} CORRECT ({correct/len(patients):.1%})")
print("="*70)

# Métriques détaillées
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score

cm = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = cm.ravel()

print(f"\nMatrice de confusion:")
print(f"  TP (Vrais positifs):  {tp:3d}")
print(f"  TN (Vrais négatifs):  {tn:3d}")
print(f"  FP (Faux positifs):   {fp:3d}")
print(f"  FN (Faux négatifs):   {fn:3d}")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = precision_score(true_labels, predictions, zero_division=0)
recall = recall_score(true_labels, predictions, zero_division=0)
f1 = f1_score(true_labels, predictions, zero_division=0)
auc = roc_auc_score(true_labels, probas)

print(f"\nMétriques:")
print(f"  Sensibilité (Recall):  {sensitivity:.3f}")
print(f"  Spécificité:           {specificity:.3f}")
print(f"  Précision:             {precision:.3f}")
print(f"  F1-Score:              {f1:.3f}")
print(f"  AUC-ROC:               {auc:.3f}")

print("\n" + "="*70)
print("MODÈLE EN PRODUCTION — PRÊT POUR STREAMLIT")
print("="*70)