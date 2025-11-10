# -*- coding: utf-8 -*-
"""
CognitiveAI Pro — FT-Transformer Tabular (TensorFlow)
- Split par patient (GroupShuffleSplit sur NACCID)
- Feature engineering (cognition, risques, interactions, indicateurs de manquants)
- FT-Transformer "par colonnes": num -> projection dense (tokens), cat -> embeddings (tokens)
- Entraînement Keras avec EarlyStopping/ModelCheckpoint
- Évaluation: ROC AUC, AP, F1 (seuil optimal), matrice de confusion + courbes
- Sauvegardes: modèle .keras, préproc .pkl, seuil optimal .txt

Dépendances:
    pip install numpy pandas scikit-learn tensorflow matplotlib joblib
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import List, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, f1_score, accuracy_score
)

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

CSV_PATH = "dataset.csv"
OUTDIR   = "."
MODEL_OUT = os.path.join(OUTDIR, "ft_transformer_model.keras")
PREPROC_OUT = os.path.join(OUTDIR, "tf_preproc.pkl")
THRESH_OUT  = os.path.join(OUTDIR, "best_threshold.txt")

# FT-Transformer hyperparams
D_MODEL    = 64          # taille des tokens
N_HEADS    = 4
DEPTH      = 3           # nb de blocks Transformer
DROPOUT    = 0.2
FF_MULT    = 4           # expand factor du FFN (d_ff = FF_MULT * D_MODEL)
BATCH_SIZE = 128
EPOCHS     = 50
VAL_SIZE   = 0.2

# ----------------------------------------------------------------------
# 1) Chargement & Features
# ----------------------------------------------------------------------
def load_data(csv_path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()
    if "NACCALZD" not in df.columns:
        raise ValueError("❌ Colonne cible 'NACCALZD' introuvable.")
    # MMSE = -4 → valeurs manquantes (code NACC)
    if "NACCMMSE" in df.columns:
        df.loc[df["NACCMMSE"] == -4, "NACCMMSE"] = np.nan
    print(f"✅ Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = "NACCALZD"
    y = (df[target_col] == 1).astype(int)

    base = [
        "NACCAGE", "NACCMMSE", "EDUC", "NACCAPOE", "SEX",
        "HYPERTEN", "DIABETES", "HYPERCHO", "SMOKYRS"
    ]
    cog = [
        "CDRSUM", "CDRGLOB", "NACCMOCA", "MOCATOTS", "NPIQINF", "NOGDS",
        "BOSTON", "TRAILA", "TRAILB", "DIGIF", "DIGIB", "ANIMALS",
        "UDSVERFC", "UDSVERFN", "UDSVERLC", "UDSVERLR", "UDSVERLN",
        "CRAFTVRS", "CRAFTURS", "MINTTOTS", "MINTTOTW", "REY1REC", "REYTCOR"
    ]
    bio = ["AMYLPET", "TAUPETAD", "CSFTAU", "FDGAD", "HIPPATR"]
    risks = ["STROKE", "AFIBRILL", "CONGHRT", "HATTMULT", "CBSTROKE"]

    use_cols = [c for c in (base + cog + bio + risks) if c in df.columns]
    X = df[use_cols].copy()

    # dérivées
    if set(["NACCMMSE","NACCAGE"]).issubset(X.columns):
        X["score_age_ratio"] = X["NACCMMSE"] / (X["NACCAGE"] + 1)
        X["score_squared"]   = X["NACCMMSE"] ** 2
        X["age_squared"]     = X["NACCAGE"] ** 2
    if set(["EDUC","NACCAGE"]).issubset(X.columns):
        X["educ_age_interaction"] = X["EDUC"] * X["NACCAGE"]
    if set(["NACCAPOE","NACCAGE"]).issubset(X.columns):
        X["gene_age_risk"] = X["NACCAPOE"].fillna(0) * X["NACCAGE"]
    if "NACCAGE" in X.columns:
        X["age_group"] = pd.cut(X["NACCAGE"], bins=[0,60,70,80,130],
                                labels=["jeune","senior","age_avance","tres_age"])
    if "NACCMMSE" in X.columns:
        X["cognitive_level"] = pd.cut(X["NACCMMSE"], bins=[-1,20,24,27,30],
                                      labels=["severe","modere","leger","normal"])
    if set(["CDRSUM","NACCMOCA"]).issubset(X.columns):
        X["global_cog_index"] = -X["CDRSUM"].fillna(X["CDRSUM"].median()) + \
                                 X["NACCMOCA"].fillna(X["NACCMOCA"].median())
    if set(["NACCMMSE","EDUC"]).issubset(X.columns):
        X["mmse_per_educ"] = X["NACCMMSE"] / (X["EDUC"] + 1)

    risk_cols = [c for c in ["NACCAPOE","HYPERTEN","DIABETES","HYPERCHO","STROKE","AFIBRILL","CONGHRT"] if c in X.columns]
    if risk_cols:
        X["risk_index"] = X[risk_cols].fillna(0).sum(axis=1)

    # indicateurs de manquants
    for c in X.columns:
        X[f"{c}_ismissing"] = df[c].isna().astype(int) if c in df.columns else 0

    # cast catégories
    for c in ["age_group","cognitive_level"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    # nettoyage
    X = X.replace([np.inf, -np.inf], np.nan)
    # filtrage patients trop vides
    non_nan = X.notna().sum(axis=1)
    keep = non_nan > (0.3 * X.shape[1])
    X, y = X.loc[keep], y.loc[keep]

    print(f"🔧 Features sélectionnées : {X.shape[1]} (après filtrage patients: {len(X)})")
    return X, y

# ----------------------------------------------------------------------
# 2) Split par patient (GroupShuffleSplit)
# ----------------------------------------------------------------------
def patient_safe_split(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame,
                       test_size=VAL_SIZE, random_state=RANDOM_STATE):
    if "NACCID" not in df_raw.columns:
        raise ValueError("❌ 'NACCID' introuvable pour split par patient.")
    groups = df_raw.loc[X.index, "NACCID"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    X_train, X_val = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_train, y_val = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    print("🛡️ Split par patient (NACCID).")
    return X_train, X_val, y_train, y_val

# ----------------------------------------------------------------------
# 3) Préprocessing -> Num / Cat, imputations, codes, normalisation
# ----------------------------------------------------------------------
def preprocess_fit(X_train: pd.DataFrame, X_val: pd.DataFrame):
    # colonnes catégorielles (déduites) + forcer certaines
    inferred_cats = [c for c in X_train.columns if str(X_train[c].dtype).startswith("category")]
    forced_cats = [c for c in ["age_group","cognitive_level","SEX"] if c in X_train.columns]
    cat_cols = list(sorted(set(inferred_cats + forced_cats)))

    # harmoniser catégories sur train/val
    cat_vocabs: Dict[str, List] = {}
    for c in cat_cols:
        tr = X_train[c].astype("category")
        # ajoute 'Unknown'
        tr = tr.cat.add_categories(["Unknown"])
        X_train[c] = tr.fillna("Unknown")
        # val
        vv = X_val[c].astype("category")
        # union des cats + Unknown
        union_cats = list(set(list(tr.cat.categories) + list(vv.cat.categories) + ["Unknown"]))
        X_train[c] = X_train[c].astype("category").cat.set_categories(union_cats)
        X_val[c]   = vv.cat.add_categories(["Unknown"]).fillna("Unknown").cat.set_categories(union_cats)
        cat_vocabs[c] = union_cats

    # convertir catégories en codes
    for c in cat_cols:
        X_train[c] = X_train[c].cat.codes
        X_val[c]   = X_val[c].cat.codes

    # colonnes numériques = tout le reste
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # imputations numériques (médianes sur train)
    medians = X_train[num_cols].median()
    X_train[num_cols] = X_train[num_cols].fillna(medians)
    X_val[num_cols]   = X_val[num_cols].fillna(medians)

    # Normalization layer (Keras) sur les num
    norm_layer = layers.Normalization(axis=-1)
    norm_layer.adapt(X_train[num_cols].astype("float32").values)

    preproc = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "medians": medians.to_dict(),
        "cat_vocabs": cat_vocabs,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "depth": DEPTH,
        "dropout": DROPOUT,
        "ff_mult": FF_MULT
    }
    return X_train, X_val, preproc, norm_layer

def to_tf_inputs(X: pd.DataFrame, preproc: dict):
    num_cols = preproc["num_cols"]
    cat_cols = preproc["cat_cols"]

    X_num = X[num_cols].astype("float32").values  # (N, n_num)
    X_cats = [X[c].astype("int32").values for c in cat_cols]  # list of (N,)
    return X_num, X_cats

# ----------------------------------------------------------------------
# 4) Modèle FT-Transformer tabulaire
# ----------------------------------------------------------------------
def transformer_block(x, d_model, n_heads, dropout, ff_mult, name):
    # x: (batch, T, d_model)
    ln1 = layers.LayerNormalization(name=f"{name}_ln1")(x)
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads,
                                     dropout=dropout, name=f"{name}_mha")(ln1, ln1)
    x = layers.Add(name=f"{name}_res1")([x, attn])
    ln2 = layers.LayerNormalization(name=f"{name}_ln2")(x)
    ff  = layers.Dense(ff_mult * d_model, activation="gelu", name=f"{name}_ff1")(ln2)
    ff  = layers.Dropout(dropout, name=f"{name}_drop1")(ff)
    ff  = layers.Dense(d_model, name=f"{name}_ff2")(ff)
    x   = layers.Add(name=f"{name}_res2")([x, ff])
    return x
# ----------------------------------------------------------------------
# Layer [CLS] sérialisable
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Layer [CLS] sérialisable
# ----------------------------------------------------------------------
class CLSToken(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.cls = self.add_weight(
            shape=(1, 1, dim),
            initializer="random_normal",
            trainable=True,
            name="cls_token"
        )

    def call(self, x):
        return tf.tile(self.cls, [tf.shape(x)[0], 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


def build_ft_transformer(num_numeric: int,
                         cat_cardinalities: List[int],
                         d_model=D_MODEL, n_heads=N_HEADS,
                         depth=DEPTH, dropout=DROPOUT, ff_mult=FF_MULT,
                         norm_layer: layers.Normalization = None) -> Model:
    # Inputs
    num_in = layers.Input(shape=(num_numeric,), name="numeric")
    cat_ins = [layers.Input(shape=(1,), dtype="int32", name=f"cat_{i}") for i in range(len(cat_cardinalities))]

    # Normalisation num
    x_num = num_in
    if norm_layer is not None:
        x_num = norm_layer(x_num)  # (B, n_num)

    # Numeric tokens: (B, n_num, 1) -> TimeDistributed(Dense(d_model))
    x_num_tok = layers.Reshape((num_numeric, 1), name="num_reshape")(x_num)
    x_num_tok = layers.TimeDistributed(layers.Dense(d_model, activation=None), name="num_proj")(x_num_tok)  # (B, n_num, d_model)

    # Cat tokens: pour chaque cat → Embedding(vocab, d_model)
    cat_tok_list = []
    for i, card in enumerate(cat_cardinalities):
        emb = layers.Embedding(input_dim=card, output_dim=d_model, name=f"cat_emb_{i}")(cat_ins[i])
        emb = layers.Reshape((1, d_model), name=f"cat_emb_rs_{i}")(emb)  # (B, 1, d_model)
        cat_tok_list.append(emb)

    # Concat tokens: (B, T, d_model)
    if len(cat_tok_list) > 0:
        x_cat_tok = layers.Concatenate(axis=1, name="cat_concat")(cat_tok_list)  # (B, n_cat, d_model)
        tokens = layers.Concatenate(axis=1, name="all_tokens")([x_num_tok, x_cat_tok])
    else:
        tokens = x_num_tok

    # Ajouter un token [CLS] appris
    # crée un vecteur entraînable (1, 1, d_model), puis tuile sur batch
    # Ajouter un token [CLS] appris (couche sérialisable)
    cls_tok = CLSToken(d_model)(tokens)

    tokens = layers.Concatenate(axis=1, name="prepend_cls")([cls_tok, tokens])  # (B, 1+T, d_model)

    # Stacks Transformer
    x = tokens
    for d in range(depth):
        x = transformer_block(x, d_model, n_heads, dropout, ff_mult, name=f"trf{d+1}")

    # On récupère le token [CLS] (position 0)
    cls_out = layers.Lambda(lambda t: t[:, 0, :], name="cls_extract")(x)

    # Head classification
    out = layers.LayerNormalization(name="head_ln")(cls_out)
    out = layers.Dropout(dropout, name="head_drop")(out)
    out = layers.Dense(1, activation="sigmoid", name="pred")(out)

    model = Model(inputs=[num_in] + cat_ins, outputs=out, name="FTTransformerTabular")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model

# ----------------------------------------------------------------------
# 5) Datasets TF
# ----------------------------------------------------------------------
def make_tf_dataset(X_num, X_cats, y=None, batch_size=BATCH_SIZE, shuffle=False):
    inputs = {"numeric": X_num}
    for i, arr in enumerate(X_cats):
        inputs[f"cat_{i}"] = arr.reshape(-1, 1)

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(inputs)
    else:
        y = y.astype("float32").values if isinstance(y, pd.Series) else y.astype("float32")
        ds = tf.data.Dataset.from_tensor_slices((inputs, y))
    if shuffle:
        ds = ds.shuffle(65536, seed=RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------------------------------------------------
# 6) Entraînement + Évaluation
# ----------------------------------------------------------------------
def plot_and_save_curves(y_true, y_prob, outdir=OUTDIR, prefix="fttf"):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2*prec*rec/(prec+rec+1e-10)
    best_idx = np.argmax(f1)
    best_th = thr[best_idx] if best_idx < len(thr) else 0.5

    # ROC
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — FT-Transformer")
    plt.legend(); plt.grid(alpha=0.3)
    path_roc = os.path.join(outdir, f"{prefix}_roc.png")
    plt.tight_layout(); plt.savefig(path_roc, dpi=300); plt.close()

    # PR
    plt.figure(figsize=(7,6))
    plt.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — FT-Transformer")
    plt.legend(); plt.grid(alpha=0.3)
    path_pr = os.path.join(outdir, f"{prefix}_pr.png")
    plt.tight_layout(); plt.savefig(path_pr, dpi=300); plt.close()

    print(f"✅ Courbes sauvegardées : {path_roc} | {path_pr}")
    return best_th, {"auc":auc, "ap":ap}

def main():
    print("📥 Chargement...")
    df = load_data(CSV_PATH)
    X, y = build_features(df)

    # split par patient
    X_train, X_val, y_train, y_val = patient_safe_split(X, y, df, test_size=VAL_SIZE)

    # préproc
    X_train_p, X_val_p, preproc, norm_layer = preprocess_fit(X_train.copy(), X_val.copy())
    num_cols = preproc["num_cols"]
    cat_cols = preproc["cat_cols"]

    # cardinalités pour embeddings ( +1 si codes commencent à 0 )
    cat_card = []
    for c in cat_cols:
        max_code = max(int(X_train_p[c].max()), int(X_val_p[c].max())) if len(X_train_p) else int(X_val_p[c].max())
        cat_card.append(max_code + 1)

    # tensors
    Xtr_num, Xtr_cats = to_tf_inputs(X_train_p, preproc)
    Xva_num, Xva_cats = to_tf_inputs(X_val_p, preproc)

    # datasets
    ds_tr = make_tf_dataset(Xtr_num, Xtr_cats, y_train, shuffle=True)
    ds_va = make_tf_dataset(Xva_num, Xva_cats, y_val, shuffle=False)

    # modèle
    print("🧪 Construction du modèle FT-Transformer...")
    model = build_ft_transformer(
        num_numeric=len(num_cols),
        cat_cardinalities=cat_card,
        d_model=D_MODEL, n_heads=N_HEADS,
        depth=DEPTH, dropout=DROPOUT, ff_mult=FF_MULT,
        norm_layer=norm_layer
    )
    model.summary()

    # callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_auc", mode="max",
                                              save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                          patience=6, restore_best_weights=True, verbose=1)
    lrplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                     factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    print("🚀 Entraînement...")
    hist = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=EPOCHS,
        callbacks=[ckpt, es, lrplateau],
        verbose=1
    )

    print("🔎 Évaluation sur le set de validation...")
    # recharger meilleur modèle si besoin
    if os.path.exists(MODEL_OUT):
        model = tf.keras.models.load_model(MODEL_OUT)

    # prédictions
    y_prob = model.predict(ds_va, verbose=0).ravel()
    best_th, curve_metrics = plot_and_save_curves(y_val.values, y_prob, OUTDIR)

    y_pred = (y_prob >= best_th).astype(int)

    auc = roc_auc_score(y_val, y_prob)
    ap  = average_precision_score(y_val, y_prob)
    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred)
    cm  = confusion_matrix(y_val, y_pred)

    print(f"\n🎯 Seuil optimal F1 : {best_th:.3f}")
    print(f"AUC     : {auc:.3f}")
    print(f"AvgPrec : {ap:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1      : {f1:.3f}\n")
    print("Rapport de classification :")
    print(classification_report(y_val, y_pred, target_names=["Sain","Alzheimer"]))
    print("Matrice de confusion :\n", cm)

    # sauvegardes préproc + seuil
    joblib.dump({
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "medians": preproc["medians"],
        "cat_vocabs": preproc["cat_vocabs"],
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "depth": DEPTH,
        "dropout": DROPOUT,
        "ff_mult": FF_MULT
    }, PREPROC_OUT)
    with open(THRESH_OUT, "w") as f:
        f.write(str(float(best_th)))

    # Confusion heatmap
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
    cm_path = os.path.join(OUTDIR, "fttf_confusion.png")
    plt.tight_layout(); plt.savefig(cm_path, dpi=300); plt.close()
    print(f"✅ Sauvegardé : {cm_path}")

    print("\n✅ Modèle sauvegardé :", MODEL_OUT)
    print("✅ Préprocessing     :", PREPROC_OUT)
    print("✅ Seuil optimal     :", THRESH_OUT)
    print("✨ Terminé.")


if __name__ == "__main__":
    main()
