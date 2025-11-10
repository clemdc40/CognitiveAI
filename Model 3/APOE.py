import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement
df = pd.read_csv("investigator_nacc71.csv", low_memory=False)

# --- Nettoyage des valeurs manquantes / codes spéciaux ---
df.replace([-4, -9], np.nan, inplace=True)

# --- Sélection des colonnes d'intérêt ---
cols = ['NACCID', 'VISITYR', 'VISITMO', 'PACKET', 'NACCAPOE', 'NACCMMSE']
df = df[cols].dropna(subset=['NACCAPOE', 'NACCMMSE'])

# --- Extraction du génotype APOE ---
# Porteurs de ε4 si le génotype contient un 4 (ex: 24, 34, 44)
df['APOE_e4_carrier'] = df['NACCAPOE'].astype(str).str.contains('4').astype(int)

# --- Création d’une variable temporelle (année de visite) ---
df['VISITYEAR'] = df['VISITYR']

# --- Calcul du MMSE moyen par année et groupe APOE ---
summary = df.groupby(['VISITYEAR', 'APOE_e4_carrier'])['NACCMMSE'].mean().reset_index()

# --- Visualisation ---
plt.figure(figsize=(8,5))
for carrier, sub in summary.groupby('APOE_e4_carrier'):
    label = "ε4 carriers" if carrier == 1 else "Non-carriers"
    plt.plot(sub['VISITYEAR'], sub['NACCMMSE'], marker='o', label=label)

plt.title("Déclin cognitif (MMSE) selon le statut APOE ε4")
plt.xlabel("Année de visite")
plt.ylabel("Score MMSE moyen")
plt.legend()
plt.grid(True)
plt.show()

# --- Corrélation statistique globale ---
corr = df[['APOE_e4_carrier', 'NACCMMSE']].corr().iloc[0,1]
print(f"Corrélation APOE ε4 ↔ MMSE : {corr:.3f}")
