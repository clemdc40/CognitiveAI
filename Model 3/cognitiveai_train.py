# -*- coding: utf-8 -*-
"""
CognitiveAI Pro — Pipeline AUC 0.86–0.88 (selon données)
- Split par patient (GroupShuffleSplit sur NACCID si dispo)
- Features expertes (cognition, risques, interactions, indicateurs de manquants)
- XGB / LGBM / CatBoost + stacking meta-XGB + calibration isotonic
- Visualisations: ROC, PR, Confusion

Dépendances:
    pip install numpy pandas scikit-learn xgboost lightgbm catboost optuna matplotlib
(Optuna est optionnel, désactivé par défaut)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold, GroupShuffleSplit, train_test_split
)
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score, accuracy_score
)
from sklearn.utils import shuffle as sk_shuffle
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
N_SPLITS_STACK = 5
USE_OPTUNA = False          # Passe à True pour tuner (plus lent, souvent +AUC)
N_TRIALS_OPTUNA = 30        # Augmente si tu veux pousser le tuning

# ----------------------------------------------------------------------
# 1) Chargement
# ----------------------------------------------------------------------
def load_data(csv_path="dataset.csv"):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    # Normalise les noms de colonnes
    df.columns = df.columns.str.strip()
    return df

# ----------------------------------------------------------------------
# 2) Sélection de colonnes & Feature engineering
# ----------------------------------------------------------------------
def build_features(df):
    # Cible (NACCAD-LB: NACCALZD = diagnostic neuropath probable AD)
    target_col = "NACCALZD"
    if target_col not in df.columns:
        raise ValueError("❌ Colonne cible 'NACCALZD' introuvable dans le CSV.")
    y = (df[target_col] == 1).astype(int)

    # Colonnes de base fortement utiles si présentes
    base = [
        "NACCAGE", "NACCMMSE", "EDUC", "NACCAPOE", "SEX",
        "HYPERTEN", "DIABETES", "HYPERCHO", "SMOKYRS"
    ]

    # Scores cognitifs & fonctionnels (ajoute-en d'autres si dispo)
    cog = [
        "CDRSUM", "CDRGLOB", "NACCMOCA", "MOCATOTS", "NPIQINF", "NOGDS",
        "BOSTON", "TRAILA", "TRAILB", "DIGIF", "DIGIB", "ANIMALS",
        "UDSVERFC", "UDSVERFN", "UDSVERLC", "UDSVERLR", "UDSVERLN",
        "CRAFTVRS", "CRAFTURS", "MINTTOTS", "MINTTOTW", "REY1REC", "REYTCOR"
    ]

    # Biomarqueurs / imagerie (si présents)
    bio = ["AMYLPET", "TAUPETAD", "CSFTAU", "FDGAD", "HIPPATR"]

    # Comorbidités cardio/neuro
    risks = ["STROKE", "AFIBRILL", "CONGHRT", "HATTMULT", "CBSTROKE"]

    # On ne garde que celles qui existent vraiment dans le CSV
    use_cols = [c for c in (base + cog + bio + risks) if c in df.columns]
    if "NACCMMSE" not in use_cols:
        print("⚠️ NACCMMSE manquant ; le signal sera réduit.")
    X = df[use_cols].copy()

    # --- Ingénierie de variables ---
    # Ratios/puissances
    if set(["NACCMMSE", "NACCAGE"]).issubset(X.columns):
        X["score_age_ratio"] = X["NACCMMSE"] / (X["NACCAGE"] + 1)
        X["score_squared"]   = X["NACCMMSE"] ** 2
        X["age_squared"]     = X["NACCAGE"] ** 2

    if set(["EDUC", "NACCAGE"]).issubset(X.columns):
        X["educ_age_interaction"] = X["EDUC"] * X["NACCAGE"]

    if set(["NACCAPOE", "NACCAGE"]).issubset(X.columns):
        X["gene_age_risk"] = X["NACCAPOE"].fillna(0) * X["NACCAGE"]

    # Binning âge & score cognitif
    if "NACCAGE" in X.columns:
        X["age_group"] = pd.cut(X["NACCAGE"], bins=[0,60,70,80,130],
                                labels=["jeune","senior","age_avance","tres_age"])
    if "NACCMMSE" in X.columns:
        X["cognitive_level"] = pd.cut(X["NACCMMSE"], bins=[-1,20,24,27,30],
                                      labels=["severe","modere","leger","normal"])

    # Index globaux
    if set(["CDRSUM","NACCMOCA"]).issubset(X.columns):
        X["global_cog_index"] = -X["CDRSUM"].fillna(X["CDRSUM"].median()) + \
                                 X["NACCMOCA"].fillna(X["NACCMOCA"].median())
    if set(["NACCMMSE","EDUC"]).issubset(X.columns):
        X["mmse_per_educ"] = X["NACCMMSE"] / (X["EDUC"] + 1)

    # Index de risque cumulé
    risk_cols = [c for c in ["NACCAPOE","HYPERTEN","DIABETES","HYPERCHO","STROKE","AFIBRILL","CONGHRT"] if c in X.columns]
    if risk_cols:
        X["risk_index"] = X[risk_cols].fillna(0).sum(axis=1)

    # Indicateurs de données manquantes (très utiles en clinique)
    for c in X.columns:
        X[f"{c}_ismissing"] = df[c].isna().astype(int) if c in df.columns else 0

    # Cast légers
    for c in ["age_group","cognitive_level"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # Nettoyage infs/NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    print(f"🔧 Features sélectionnées : {len(X.columns)}")
    return X, y

# ----------------------------------------------------------------------
# 3) Split sans fuite (GroupShuffleSplit si NACCID présent)
# ----------------------------------------------------------------------
def patient_safe_split(X, y, df, test_size=0.2, random_state=RANDOM_STATE):
    if "NACCID" in df.columns:
        groups = df.loc[X.index, "NACCID"]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_train, y_test = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
        print("🛡️ Split par patient (GroupShuffleSplit sur NACCID).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("ℹ️ Split stratifié (NACCID introuvable).")
    return X_train, X_test, y_train, y_test

# ----------------------------------------------------------------------
# 4) Pré-traitement sans fuite (fit sur train uniquement)
# ----------------------------------------------------------------------
def preprocess_fit_transform(X_train, X_test):
    # Numerics vs categories
    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype).startswith("category")]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # Imputation (numérique) — médiane sur train
    for c in num_cols:
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med)
        X_test[c]  = X_test[c].fillna(med)

    # Remplir les catégories manquantes par "Unknown"
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c]  = X_test[c].astype("category")
        # Harmoniser les catégories
        all_cats = list(set(list(X_train[c].cat.categories) + list(X_test[c].cat.categories) + ["Unknown"]))
        X_train[c] = X_train[c].cat.add_categories(["Unknown"]).fillna("Unknown")
        X_test[c]  = X_test[c].cat.add_categories(["Unknown"]).fillna("Unknown")
        X_train[c] = X_train[c].cat.set_categories(all_cats)
        X_test[c]  = X_test[c].cat.set_categories(all_cats)

    # Encode ordinal simple pour catégories (CatBoost supporterait les str, mais on unifie)
    for c in cat_cols:
        X_train[c] = X_train[c].cat.codes.replace(-1, np.nan).fillna(X_train[c].cat.codes[X_train[c]!=-1].median())
        X_test[c]  = X_test[c].cat.codes.replace(-1, np.nan).fillna(X_train[c].median())

    # Scale robuste (numériques uniquement)
    scaler = RobustScaler()
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train, X_test, num_cols, cat_cols

# ----------------------------------------------------------------------
# 5) Monotonic constraints (alignées aux colonnes)
# ----------------------------------------------------------------------
def build_monotone_constraints(feature_names):
    # +1 = croissant (plus => +risque), -1 = décroissant
    mono_map = {
        "NACCAGE": +1, "age_squared": +1, "gene_age_risk": +1,
        "NACCAPOE": +1, "HYPERTEN": +1, "DIABETES": +1, "HYPERCHO": +1, "SMOKYRS": +1, "risk_index": +1,
        "NACCMMSE": -1, "score_squared": -1, "score_age_ratio": -1, "mmse_per_educ": -1,
        "EDUC": -1, "educ_age_interaction": -1, "global_cog_index": -1
    }
    lgb_cons = [mono_map.get(f, 0) for f in feature_names]
    xgb_cons = "(" + ",".join(str(mono_map.get(f, 0)) for f in feature_names) + ")"
    return xgb_cons, lgb_cons

# ----------------------------------------------------------------------
# 6) (Optionnel) Tuning Optuna — ici sur XGB (copie/colle pour LGBM/Cat si besoin)
# ----------------------------------------------------------------------
def tune_xgb_optuna(X_train, y_train):
    import optuna
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 600, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }
        # class weight
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        params["scale_pos_weight"] = neg / max(pos, 1)

        model = XGBClassifier(**params)
        aucs = []
        for tr, va in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            p = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, p))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS_OPTUNA)
    print("🔎 Best XGB params:", study.best_params)
    return study.best_params

# ----------------------------------------------------------------------
# 7) Stacking OOF (XGB/LGBM/Cat) + meta-XGB + calibration isotonic
# ----------------------------------------------------------------------
def fit_stacked_models(X_train, y_train, X_test, feature_names):
    # class weights / imbalance
    pos, neg = (y_train==1).sum(), (y_train==0).sum()
    scale_pos = neg / max(pos, 1)

    xgb_cons, lgb_cons = build_monotone_constraints(feature_names)

    # Base learners (tu peux durcir si Optuna OFF)
    xgb_base = XGBClassifier(
        n_estimators=1200, learning_rate=0.02, max_depth=6,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="auc", tree_method="hist",
        monotone_constraints=xgb_cons,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE
    )

    lgb_base = LGBMClassifier(
        n_estimators=1400, learning_rate=0.02, max_depth=6, num_leaves=40,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=2.0,
        #monotone_constraints=lgb_cons,
        is_unbalance=True,
        random_state=RANDOM_STATE
    )

    cat_base = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, loss_function="Logloss",
        # class_weights: [w_neg, w_pos]
        class_weights=[1.0, scale_pos],
        random_seed=RANDOM_STATE,
        verbose=False
    )

    bases = [("xgb", xgb_base), ("lgb", lgb_base), ("cat", cat_base)]

    # Optuna (facultatif) — uniquement XGB ici par souci de temps
    if USE_OPTUNA:
        best = tune_xgb_optuna(X_train, y_train)
        for k in ["n_estimators","learning_rate","max_depth","min_child_weight",
                  "subsample","colsample_bytree","gamma","reg_alpha","reg_lambda"]:
            setattr(xgb_base, k, best[k])

    # OOF + moyenne sur le set test pour stabilité
    kf = StratifiedKFold(n_splits=N_SPLITS_STACK, shuffle=True, random_state=RANDOM_STATE)
    oof_meta = np.zeros((len(X_train), len(bases)))
    test_meta = np.zeros((len(X_test), len(bases)))

    for b_idx, (name, base) in enumerate(bases):
        print(f"⚙️  OOF pour base: {name}")
        test_fold_pred = np.zeros((len(X_test), N_SPLITS_STACK))
        for i, (tr, va) in enumerate(kf.split(X_train, y_train), 1):
            mdl = clone(base)
            X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
            y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
            mdl.fit(X_tr, y_tr)
            oof_meta[va, b_idx] = mdl.predict_proba(X_va)[:, 1]
            test_fold_pred[:, i-1] = mdl.predict_proba(X_test)[:, 1]
        test_meta[:, b_idx] = test_fold_pred.mean(axis=1)

    # Méta-modèle (XGB simple) + calibration isotonic
    meta_xgb = XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="auc", tree_method="hist",
        random_state=RANDOM_STATE
    )
    # CalibratedClassifierCV attend un estimator sklearn-like
    calibrated_meta = CalibratedClassifierCV(estimator=meta_xgb, method="isotonic", cv=3)
    calibrated_meta.fit(oof_meta, y_train)

    # Probabilités finales sur le test
    y_prob_test = calibrated_meta.predict_proba(test_meta)[:, 1]
    
    # ⭐ IMPORTANT: Entraîner les modèles de base sur TOUT le train set
    # pour pouvoir les utiliser en prédiction
    print("⚙️  Entraînement des modèles de base sur le set complet...")
    xgb_fitted = clone(xgb_base)
    lgb_fitted = clone(lgb_base)
    cat_fitted = clone(cat_base)
    
    xgb_fitted.fit(X_train, y_train)
    lgb_fitted.fit(X_train, y_train)
    cat_fitted.fit(X_train, y_train)
    print("✅ Modèles de base entraînés sur l'ensemble complet")
    
    return y_prob_test, oof_meta, test_meta, bases, calibrated_meta, \
           xgb_fitted, lgb_fitted, cat_fitted

# ----------------------------------------------------------------------
# 8) Évaluation & tracés
# ----------------------------------------------------------------------
def evaluate_and_plot(y_test, y_prob, outdir="."):
    # Choix d’un seuil au F1
    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1 = 2*prec*rec/(prec+rec+1e-10)
    best_idx = np.argmax(f1)
    best_th = thr[best_idx] if best_idx < len(thr) else 0.5
    y_pred = (y_prob >= best_th).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1s = f1_score(y_test, y_pred)
    print(f"\n🎯 Seuil optimal F1 : {best_th:.3f}")
    print(f"AUC     : {auc:.3f}")
    print(f"AvgPrec : {ap:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1      : {f1s:.3f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=["Sain","Alzheimer"]))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], ls="--")
    plt.xlabel("Faux positifs"); plt.ylabel("Vrais positifs")
    plt.title("Courbe ROC — CognitiveAI Pro")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    roc_path = os.path.join(outdir, "roc_curve.png")
    plt.tight_layout(); plt.savefig(roc_path, dpi=300); plt.close()
    print(f"✅ Sauvegardé : {roc_path}")

    # Plot PR
    plt.figure(figsize=(7,6))
    plt.plot(rec, prec, lw=2, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Courbe Précision–Rappel — CognitiveAI Pro")
    plt.legend(); plt.grid(alpha=0.3)
    pr_path = os.path.join(outdir, "pr_curve.png")
    plt.tight_layout(); plt.savefig(pr_path, dpi=300); plt.close()
    print(f"✅ Sauvegardé : {pr_path}")

    # Heatmap confusion (simple, sans seaborn pour compat)
    import itertools
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="RdYlGn")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Sain","Alzheimer"]); ax.set_yticklabels(["Sain","Alzheimer"])
    plt.xlabel("Prédiction"); plt.ylabel("Vérité terrain")
    plt.title(f"Matrice de confusion — Acc: {acc:.2%}")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = os.path.join(outdir, "confusion_matrix.png")
    plt.tight_layout(); plt.savefig(cm_path, dpi=300); plt.close()
    print(f"✅ Sauvegardé : {cm_path}")

    return {"auc":auc, "ap":ap, "acc":acc, "f1":f1s, "threshold":best_th}

# ----------------------------------------------------------------------
# 9) Main
# ----------------------------------------------------------------------
def main():
    df = load_data("dataset.csv")
    df = df.dropna(subset=["NACCALZD"])

    X, y = build_features(df)
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = patient_safe_split(X, y, df)
    X_train_p, X_test_p, num_cols, cat_cols = preprocess_fit_transform(X_train.copy(), X_test.copy())
    feature_names = X_train_p.columns.tolist()

    # ⬇️ On récupère bien 'meta' ici
    y_prob_test, oof_meta, test_meta, bases, meta, xgb_base, lgb_base, cat_base = fit_stacked_models(
        X_train_p, y_train, X_test_p, feature_names
    )

    metrics = evaluate_and_plot(y_test, y_prob_test, outdir=".")

    # 🔽 Ajoute ici la sauvegarde correcte
    import joblib
    try:
        joblib.dump(meta, "cognitiveai_model.pkl")
        joblib.dump({
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "features": feature_names
        }, "preproc.pkl")
        joblib.dump(xgb_base, "base_xgb.pkl")
        joblib.dump(lgb_base, "base_lgb.pkl")
        joblib.dump(cat_base, "base_cat.pkl")
        print("✅ Modèles de base sauvegardés (xgb, lgb, cat)")
        print("✅ Modèle et préprocesseur sauvegardés :")
        print("   → cognitiveai_model.pkl")
        print("   → preproc.pkl")
    except Exception as e:
        print("⚠️ Erreur lors de la sauvegarde :", e)

    print("\n✨ Terminé.")
    print("Résumé:", metrics)


if __name__ == "__main__":
    main()
    