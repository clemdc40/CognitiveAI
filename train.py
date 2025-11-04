# train.py
# Entraîne le MLP, imprime l'accuracy, et sauvegarde :
# - Cognitive_1.pth (poids du modèle)
# - scaler.pkl (StandardScaler)
# - features.json (ordre exact des features)

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

DATA_PATH = "alzheimers_disease_data.csv"
MODEL_PATH = "Cognitive_1.pth"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "features.json"

# 1) Chargement
data = pd.read_csv(DATA_PATH)

# 2) Préparation
features = ["Age","Gender","BMI","Smoking","AlcoholConsumption","PhysicalActivity",
            "SleepQuality","FamilyHistoryAlzheimers","MMSE","FunctionalAssessment","ADL"]
X = data[features].values.astype(np.float32)
y = data["Diagnosis"].values.astype(np.int64)

# 3) Normalisation (fit sur train uniquement)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_te = scaler.transform(X_te).astype(np.float32)

# 4) Conversion en tenseurs
x_train = torch.tensor(X_tr, dtype=torch.float32)
y_train = torch.tensor(y_tr, dtype=torch.long)
x_test  = torch.tensor(X_te, dtype=torch.float32)
y_test  = torch.tensor(y_te, dtype=torch.long)

# 5) Modèle
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

model = MLP(input_dim=x_train.shape[1], num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# 6) Entraînement
epochs = 1200
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - loss={loss.item():.4f}")

# 7) Évaluation
model.eval()
with torch.no_grad():
    logits = model(x_test)
    pred = logits.argmax(dim=1)
    acc = (pred == y_test).float().mean().item()
print(f"Test accuracy: {acc:.4f}")

# 8) Sauvegardes
torch.save(model.state_dict(), MODEL_PATH)
dump(scaler, SCALER_PATH)
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump({"features": features}, f, ensure_ascii=False, indent=2)

print(f"Artifacts saved: {MODEL_PATH}, {SCALER_PATH}, {FEATURES_PATH}")
