# CognitiveAI Pro

**Modèle IA Alzheimer optimisé – Recherche expérimentale**

Ce projet n’est pas destiné à la production.
Il s’agit d’un prototype de recherche visant à explorer la faisabilité d’un système d’analyse prédictive pour le suivi des patients atteints de la maladie d’Alzheimer.

---

## Objectif du projet

Le but de **CognitiveAI Pro** est de :

* Concevoir une IA d’évaluation préliminaire du risque d’Alzheimer, basée sur des données comportementales, médicales et cognitives.
* Tester différentes architectures de réseaux de neurones (TensorFlow, PyTorch) et modèles tabulaires hybrides (XGBoost, LightGBM, CatBoost).
* Fournir un prototype d’interface web permettant la saisie des données d’un patient et la visualisation de son score de risque estimé.

---

## Partie Web

Le site est généré automatiquement via intelligence artificielle grâce à **Wix ADI**, afin de se concentrer sur :

* la structure UX/UI du tableau de bord,
* la visualisation des résultats IA (courbes ROC, matrices de confusion, rapport de suivi patient).

L’objectif n’est pas de coder le front-end manuellement, mais de valider le concept d’intégration IA + Web automatisé.

---

## Partie Intelligence Artificielle

L’IA est développée en **Python**, avec un environnement isolé conda nommé `cognitiveai`.
Elle utilise deux approches parallèles :

1. **Réseaux de neurones légers** (TensorFlow / PyTorch) pour les tests de modèles denses.
2. **Ensembles de modèles** (XGBoost + LightGBM + CatBoost) pour les grands jeux de données tabulaires (NACC).

---

### Environnement d’installation

```bash
conda create -n cognitiveai python=3.11
conda activate cognitiveai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn pandas seaborn xgboost lightgbm catboost imbalanced-learn matplotlib tensorflow
```

---

## Exécutions et Résultats

### Exécution 1 – Réseau simple (PyTorch)

```python
Sequential(
  (0): Linear(in_features=11, out_features=64, bias=True)
  (1): ReLU()
  (3): ReLU()
  (4): Linear(in_features=128, out_features=2, bias=True)
)
Epoch [1000/1000], Loss: 0.0019
Accuracy: 0.7279
```

Le modèle atteint **72 % d’exactitude**, mais présente un **overfitting** prononcé (loss quasi nul).

---

### Exécution 2 – AdamW + Dropout

```python
Sequential(
  (0): Linear(in_features=11, out_features=64, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.3)
  (3): Linear(in_features=64, out_features=128, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.3)
  (6): Linear(in_features=128, out_features=2, bias=True)
)
Epoch [3000/3000], Loss: 0.1939
Accuracy: 0.7860
```

Amélioration notable : **78 % d’efficacité**, avec une meilleure généralisation.

---

## Deuxième génération – Dataset étendu (~70 000 patients)

### Exécution 1

```python
GPU utilisé : NVIDIA GeForce RTX 3060
Classes : ['No', 'Yes']
Epoch [300/300] | Loss: 0.4216 | Val Acc: 0.714 | AUC: 0.748
```

Risque estimé (exemple patient) :

```
Score global : 41/100
Risque modéré – certains facteurs de vigilance présents
```

---

### Exécution 2 – Optimisation

```python
Epoch [300/300] | Loss: 0.7076 | Val Acc: 0.715 | AUC: 0.746
```

**Résultats :**

* Accuracy : 71 %
* AUC : 0.75
* F1 : ~0.65

Le modèle se trompe environ **3 fois sur 10**, ce qui reste raisonnable pour un diagnostic préliminaire.

---

## Version TensorFlow – FT-Transformer (NACC Dataset)

Après obtention du dataset **NACC (National Alzheimer’s Coordinating Center)**,
l’entraînement se fait désormais sur **205 000 patients** et **plus de 1000 variables**.

Ce jeu de données est confidentiel et non redistribuable.

**Derniers résultats (TensorFlow FT-Transformer) :**

```
Accuracy : 0.729 (72.9%)
F1-score : 0.645
AUC      : 0.806
Recall Alzheimer : 0.72
Precision Alzheimer : 0.59
```

Ces performances placent **CognitiveAI Pro** au-dessus des modèles tabulaires de base publiés pour l’Alzheimer non clinique.

---

## Objectif du Prototype

Le prototype (Streamlit/Wix) a pour but de test et démonstration :

1. **Tester l’ergonomie** : facilité de saisie des données patient.
2. **Valider les performances IA** : comportement du modèle sur des profils variés.
3. **Illustrer une application clinique conceptuelle** : montrer comment une IA pourrait aider à détecter précocement des signaux faibles.

---

###  Avertissement important

* Cette IA est **expérimentale** et **non validée médicalement**.
* Les prédictions **ne constituent pas un diagnostic**.
* Les données utilisées sont **synthétiques ou pseudonymisées**.
* Ce travail a une **finalité scientifique et démonstrative**, non commerciale.
* L’objectif est de poser les bases d’un **futur modèle validé cliniquement et éthiquement**.

---

## Structure du projet

```
CognitiveAI
 ├ cognitiveai_train.py      # Entraînement complet (TensorFlow)
 ├ cognitiveai_infer.py      # Prédiction et visualisation
 ├ streamlit_app.py          # Interface web de test
 ├ requirements.txt          # Dépendances
 ├ README.md                 # Documentation (ce fichier)
 ┗ results/                  # Courbes ROC, matrices de confusion, rapports
```

---

## Avancement – Phase NACC

Ayant obtenu un accès restreint au dataset NACC,
le travail actuel se concentre sur un modèle **plus robuste et interprétable**, avec :

* Contraintes **monotones** (p. ex. âge, score cognitif, génétique).
* Calibration **isotone** pour corriger les probabilités.
* Interprétabilité **SHAP** et visualisation des variables importantes.
* Visualisation dynamique du risque patient via dashboard interactif.

Les données NACC restent **confidentielles** conformément à leurs conditions d’accès.

---

## Perspectives

* Intégration complète de **SHAP / LIME** pour expliquer chaque prédiction.
* Déploiement d’une **API Flask / FastAPI** pour servir le modèle.
* Ajout d’un mode **suivi longitudinal** (analyse temporelle du déclin cognitif).
* Passage à un modèle hybride **Deep TabNet + Gradient Boosting**.
* Création d’un **tableau de bord médical interactif** (Streamlit + Plotly).
* Optimisation du modèle pour **GPU Tensor Cores** avec TensorFlow 2.15 / CUDA 12.1.

---

###  En résumé

**CognitiveAI Pro** explore la frontière entre recherche médicale et intelligence artificielle,
en testant des architectures de pointe (FT-Transformer, XGBoost, TabNet) sur des données complexes,
dans un but clair : **aider à détecter précocement la maladie d’Alzheimer** par l’analyse intelligente de signaux faibles.

---
