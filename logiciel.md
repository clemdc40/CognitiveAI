# CognitiveAI
Ce projet n'est pas destiné à la production ! Ce n'est qu'une recherche pour un futur projet plus approfondie sur le suivi de patient atteint par Alzheimer.

## Projet
#### Web
le site web sera réalisé à l'aide d'intelligence artifiel grâce à l'outil wix, je préfère me concentrer sur l'efficacité de l'intelligence artificielle.

#### IA
L'intelligence artificielle sera entraînée grâce à PyTorch et au language python.
Nom de l'environnement conda : cognitiveai

## Installation
```bash
conda create -n cognitiveai python=3.11
conda activate cognitiveai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Exécutions 
#### Exécution 1 
```python
Sequential(
  (0): Linear(in_features=11, out_features=64, bias=True)
  (1): ReLU()
  (3): ReLU()
  (4): Linear(in_features=128, out_features=2, bias=True)
)
Epoch [100/1000], Loss: 0.4014
Epoch [200/1000], Loss: 0.2790
Epoch [300/1000], Loss: 0.1284
Epoch [400/1000], Loss: 0.0470
Epoch [500/1000], Loss: 0.0189
Epoch [600/1000], Loss: 0.0094
Epoch [700/1000], Loss: 0.0055
Epoch [800/1000], Loss: 0.0036
Epoch [900/1000], Loss: 0.0025
Epoch [1000/1000], Loss: 0.0019
Accuracy: 0.7279
```

Nous obtenons une efficacité de 72% mais l'IA à quasiement appris le dataset par coeur comme nous montre la faible valeur du loss.

#### Exécution 2
Changement d'optimizer : utilisation de AdamW plutot que Adam. Ajoute de 2 couches de dropout.
```python
Sequential(
  (0): Linear(in_features=11, out_features=64, bias=True)
  (1): ReLU()
  (2): Dropout(p=0.3, inplace=False)
  (3): Linear(in_features=64, out_features=128, bias=True)
  (4): ReLU()
  (5): Dropout(p=0.3, inplace=False)
  (6): Linear(in_features=128, out_features=2, bias=True)
)
Epoch [100/3000], Loss: 0.4598
Epoch [200/3000], Loss: 0.4282
Epoch [300/3000], Loss: 0.3876
Epoch [400/3000], Loss: 0.3685
Epoch [500/3000], Loss: 0.3564
Epoch [600/3000], Loss: 0.3304
Epoch [700/3000], Loss: 0.3191
Epoch [800/3000], Loss: 0.3103
Epoch [900/3000], Loss: 0.3012
Epoch [1000/3000], Loss: 0.2937
Epoch [1100/3000], Loss: 0.2751
Epoch [1200/3000], Loss: 0.2653
Epoch [1300/3000], Loss: 0.2572
Epoch [1400/3000], Loss: 0.2481
Epoch [1500/3000], Loss: 0.2574
Epoch [1600/3000], Loss: 0.2314
Epoch [1700/3000], Loss: 0.2386
Epoch [1800/3000], Loss: 0.2235
Epoch [1900/3000], Loss: 0.2187
Epoch [2000/3000], Loss: 0.2157
Epoch [2100/3000], Loss: 0.2056
Epoch [2200/3000], Loss: 0.2083
Epoch [2300/3000], Loss: 0.2188
Epoch [2400/3000], Loss: 0.2029
Epoch [2500/3000], Loss: 0.1911
Epoch [2600/3000], Loss: 0.2148
Epoch [2700/3000], Loss: 0.2006
Epoch [2800/3000], Loss: 0.1933
Epoch [2900/3000], Loss: 0.2134
Epoch [3000/3000], Loss: 0.1939
Accuracy: 0.7860
Predicted class index for sample: 0
```

Evolution de l'efficacité en passant de 72% à 78%.

## Objectif de test
Ce site web avec Streamlit a été développé dans un but de test et de démonstration uniquement. Il permet de :

1. **Tester l'interface utilisateur** : Évaluer l'ergonomie et la facilité d'utilisation de l'interface pour la saisie des données patient.
2. **Valider le modèle** : Vérifier le bon fonctionnement du modèle d'IA en conditions réelles avec différentes entrées.
3. **Démonstration conceptuelle** : Illustrer comment une IA pourrait potentiellement assister dans l'évaluation préliminaire des risques d'Alzheimer.

 **Avertissement important** :
- Cette application est strictement à des fins de test et de démonstration
- Les prédictions ne doivent en aucun cas être utilisées pour des diagnostics médicaux
- Les données utilisées pour l'entraînement sont synthétiques et ne reflètent pas des cas réels
- Ce prototype servira de base pour développer une solution plus robuste et médicalement validée dans le futur
