import pandas as pd

# === PARAMÈTRES ===
CSV_FILE = "investigator_nacc71.csv"

# === CHARGEMENT DU FICHIER ===
print(f"[+] Chargement du fichier {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE, low_memory=False)
print(f"[✓] Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# === FONCTIONS UTILES ===
def search_columns(keywords):
    """Retourne les colonnes contenant un des mots-clés."""
    found = [c for c in df.columns if any(k in c.lower() for k in keywords)]
    return found

# === ANALYSE PAR DOMAINES ===
domains = {
    "Identifiants / visites": ["naccid", "visit", "packet", "formver"],
    "Démographie": ["age", "sex", "educ", "ethnic", "marist"],
    "Cognitif": ["mmse", "moca", "neuro", "memory", "lang", "visu", "exec"],
    "Génétique": ["apoe", "genotype", "gene"],
    "Cliniques / risques": ["diab", "hyper", "stroke", "smok", "cardio", "cvd", "bmi"],
    "Diagnostic": ["alzd", "uds", "diag", "dement", "mci"],
    "Neuropathologie": ["braak", "cerad", "tau", "amyloid", "ab42", "path"],
}

print("\n=== 🔍 Colonnes détectées par catégorie ===\n")
selected = {}

for domain, keywords in domains.items():
    cols = search_columns(keywords)
    if cols:
        selected[domain] = cols
        print(f"\n[{domain}] ({len(cols)} trouvées)")
        for c in cols:
            print("  -", c)
    else:
        print(f"\n[{domain}] : aucune colonne trouvée")

# === SYNTHÈSE DES VARIABLES PERTINENTES ===
print("\n=== 🧩 VARIABLES RECOMMANDÉES POUR L'IA ===")
recommended = []

# Variable cible
target_candidates = search_columns(["mmse", "alzd", "uds", "diag"])
if target_candidates:
    print(f"\n🎯 Variable(s) cible(s) candidate(s) : {target_candidates}")
    recommended += target_candidates[:1]  # la première par défaut

# Variables explicatives pertinentes
for cat in ["Génétique", "Démographie", "Cognitif", "Cliniques / risques", "Neuropathologie"]:
    if cat in selected:
        recommended += selected[cat]

# Retirer doublons
recommended = list(dict.fromkeys(recommended))

print("\n✅ Variables conseillées à conserver :")
for c in recommended:
    print("  -", c)

print(f"\nTotal : {len(recommended)} variables pertinentes sur {df.shape[1]} disponibles.")

# === SAUVEGARDE OPTIONNELLE ===
out_file = "selected_variables.txt"
with open(out_file, "w") as f:
    for c in recommended:
        f.write(c + "\n")
print(f"\n💾 Liste sauvegardée dans : {out_file}")
