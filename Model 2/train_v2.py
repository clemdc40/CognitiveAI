import os
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
)

# =========================================
# Config
# =========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"GPU utilisé : {torch.cuda.get_device_name(0)}")
else:
    print("Mode CPU")

DATA_PATH = "alzheimers_prediction_dataset.csv"
BATCH_SIZE = 512
EPOCHS = 600
EARLY_STOP_PATIENCE = 50          # patience sur l'AUC
LR = 1e-3
WD = 1e-4                         # L2
VAL_METRIC = "auc"                # suivi principal
RECALL_TARGET = None              # ex: 0.85 pour forcer un recall min (sinon None)

# =========================================
# Données et prétraitement
# =========================================
data = pd.read_csv(DATA_PATH)

suivi_features = [
    "Age",
    "Gender",
    "BMI",
    "Physical Activity Level",
    "Smoking Status",
    "Alcohol Consumption",
    "Diabetes",
    "Hypertension",
    "Cholesterol Level",
    "Cognitive Test Score",
    "Depression Level",
    "Sleep Quality",
    "Social Engagement Level",
    "Stress Levels"
]
label_col = "Alzheimer’s Diagnosis"

X_raw = data[suivi_features].copy()
y_raw = pd.Categorical(data[label_col]).codes
classes = list(pd.Categorical(data[label_col]).categories)
print("Classes :", classes)

cat_cols = [c for c in X_raw.columns if X_raw[c].dtype == 'object' or str(X_raw[c].dtype).startswith("category")]
num_cols = [c for c in X_raw.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X = preprocessor.fit_transform(X_raw)
joblib.dump(preprocessor, "alz_preprocessor_suivi.pkl")

X_train, X_val, y_train_np, y_val_np = train_test_split(
    X, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)

x_train = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train,
                       dtype=torch.float32, device=DEVICE)
x_val   = torch.tensor(X_val.toarray()   if hasattr(X_val, "toarray")   else X_val,
                       dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(y_train_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
y_val   = torch.tensor(y_val_np,   dtype=torch.float32, device=DEVICE).unsqueeze(1)

# =========================================
# Modèle (sortie = logits, pas de Sigmoid)
# =========================================
class AlzheimerRiskModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # logits
        )
    def forward(self, x):
        return self.net(x)

model = AlzheimerRiskModel(x_train.shape[1]).to(DEVICE)

# =========================================
# Perte, poids de classes, optim, scheduler
# =========================================
# pos_weight = N_neg / N_pos
pos = (y_train_np == 1).sum()
neg = (y_train_np == 0).sum()
pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=DEVICE)
print(f"pos_weight (N_neg/N_pos) = {pos_weight.item():.3f}")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=20, verbose=True
)

# =========================================
# Boucle d'entraînement + Early stopping
# =========================================
best_auc = -np.inf
best_epoch = -1
best_state = None
t0 = time.time()

def eval_metrics(logits, y_true):
    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_true = y_true.detach().cpu().numpy().ravel()
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    f1  = f1_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    return {"acc": acc, "auc": auc, "f1": f1, "recall": rec, "precision": pre, "probs": probs}

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(x_train.size(0), device=DEVICE)
    epoch_loss = 0.0

    for i in range(0, x_train.size(0), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        bx, by = x_train[idx], y_train[idx]
        optimizer.zero_grad()
        logits = model(bx)
        loss = loss_fn(logits, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * bx.size(0)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        metrics = eval_metrics(val_logits, y_val)

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}] | Loss: {epoch_loss/x_train.size(0):.4f} "
              f"| Acc: {metrics['acc']:.3f} | AUC: {metrics['auc']:.3f} "
              f"| F1: {metrics['f1']:.3f} | Recall: {metrics['recall']:.3f}")

    # Scheduler (sur AUC)
    scheduler.step(metrics["auc"])

    # Early stopping sur AUC
    if metrics["auc"] > best_auc:
        best_auc = metrics["auc"]
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch - best_epoch >= EARLY_STOP_PATIENCE:
        print(f"Early stopping à l'epoch {epoch} (meilleure AUC={best_auc:.3f} à l'epoch {best_epoch})")
        break

print(f"Durée totale: {(time.time()-t0)/60:.2f} min")
model.load_state_dict(best_state)

# =========================================
# Calibration par Temperature Scaling (sur validation)
# =========================================
# On optimise T pour minimiser la NLL (BCEWithLogits) sur la val
with torch.no_grad():
    logits_val = model(x_val).detach()

T = torch.tensor([1.0], device=DEVICE, requires_grad=True)
opt_T = optim.LBFGS([T], lr=0.1, max_iter=50)

def _nll_with_T():
    def closure():
        opt_T.zero_grad()
        scaled = logits_val / T
        loss = nn.BCEWithLogitsLoss()(scaled, y_val)
        loss.backward()
        return loss
    return closure

opt_T.step(_nll_with_T())

T_clamped = T.detach().clamp(1e-3, 100.0)  # sécurité
print(f"Température calibrée T = {T_clamped.item():.4f}")

def sigmoid_with_T(logits, Tval):
    return torch.sigmoid(logits / Tval)

# =========================================
# Choix du seuil optimal (par défaut: max F1)
# Optionnel: contrainte de recall min (RECALL_TARGET)
# =========================================
with torch.no_grad():
    logits_val = model(x_val)
    probs_val = sigmoid_with_T(logits_val, T_clamped).cpu().numpy().ravel()
y_true_val = y_val.cpu().numpy().ravel()

thresholds = np.linspace(0.05, 0.95, 181)
best_thr = 0.5
best_f1 = -1.0

if RECALL_TARGET is None:
    for th in thresholds:
        yp = (probs_val >= th).astype(int)
        f1 = f1_score(y_true_val, yp)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = th
else:
    # on maximise F1 sous contrainte recall >= RECALL_TARGET
    for th in thresholds:
        yp = (probs_val >= th).astype(int)
        rec = recall_score(y_true_val, yp)
        if rec >= RECALL_TARGET:
            f1 = f1_score(y_true_val, yp)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = th

print(f"Seuil choisi = {best_thr:.3f} (F1 sur val = {best_f1:.3f}, "
      f"AUC sur val = {roc_auc_score(y_true_val, probs_val):.3f})")

# =========================================
# Sauvegarde bundle complet
# =========================================
bundle = {
    "model_state_dict": model.state_dict(),
    "input_dim": x_train.shape[1],
    "temperature": float(T_clamped.item()),
    "threshold": float(best_thr),
    "classes": classes,
}
torch.save(bundle, "alz_suivi_bundle.pth")
print("Sauvegardé: alz_suivi_bundle.pth + alz_preprocessor_suivi.pkl")

# =========================================
# Inference: score de risque + rapport
# =========================================
def load_bundle(model_path="alz_suivi_bundle.pth", preproc_path="alz_preprocessor_suivi.pkl"):
    b = torch.load(model_path, map_location=DEVICE)
    prep = joblib.load(preproc_path)
    m = AlzheimerRiskModel(b["input_dim"]).to(DEVICE)
    m.load_state_dict(b["model_state_dict"])
    m.eval()
    return m, prep, b["temperature"], b["threshold"], b.get("classes", ["No", "Yes"])

def predict_risk(patient_dict, model_obj=None, preproc_obj=None, Tval=1.0):
    need_close = False
    if model_obj is None or preproc_obj is None:
        model_obj, preproc_obj, Tval, _, _ = load_bundle()
        need_close = True
    df = pd.DataFrame([patient_dict])
    Xp = preproc_obj.transform(df)
    xt = torch.tensor(Xp.toarray() if hasattr(Xp, "toarray") else Xp,
                      dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits = model_obj(xt)
        prob = sigmoid_with_T(logits, torch.tensor(Tval, device=DEVICE)).item()
    score = int(round(prob * 100))
    if need_close:
        pass
    return score, prob

def rapport_patient(patient_dict, threshold=None):
    model_obj, preproc_obj, Tval, thr, classes_ = load_bundle()
    if threshold is not None:
        thr_use = threshold
    else:
        thr_use = thr

    score, prob = predict_risk(patient_dict, model_obj, preproc_obj, Tval)
    label = "Élevé" if prob >= thr_use else ("Modéré" if prob >= 0.3 else "Faible")

    print("\n--- Rapport de Suivi Alzheimer ---")
    print(f"Probabilité calibrée : {prob*100:.1f}%  | Score : {score}/100")
    print(f"Seuil décisionnel : {thr_use:.2f}  → Catégorie : {label}")
    if label == "Faible":
        print("🟢 Risque faible : pas d’indicateur alarmant pour le moment.")
    elif label == "Modéré":
        print("🟡 Risque modéré : certains facteurs de vigilance présents.")
    else:
        print("🔴 Risque élevé : suivi médical conseillé.")
    print("\nFacteurs clés observés :")
    for k, v in patient_dict.items():
        print(f" - {k}: {v}")
    print("-----------------------------")

# =========================================
# Exemple d’utilisation
# =========================================
if __name__ == "__main__":
    patient_exemple = {
        "Age": 74,
        "Gender": "Female",
        "BMI": 26.5,
        "Physical Activity Level": "Low",
        "Smoking Status": "Current",
        "Alcohol Consumption": "Occasionally",
        "Diabetes": "Yes",
        "Hypertension": "Yes",
        "Cholesterol Level": "High",
        "Cognitive Test Score": 45,
        "Depression Level": "High",
        "Sleep Quality": "Poor",
        "Social Engagement Level": "Low",
        "Stress Levels": "High"
    }
    rapport_patient(patient_exemple)
