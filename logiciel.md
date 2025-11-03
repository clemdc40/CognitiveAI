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