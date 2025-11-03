# Projet : CognitiveTrack – Outil IA de suivi cognitif léger
## Objectif

Créer un outil web open-source permettant de suivre la cognition et la routine d’une personne atteinte d’Alzheimer, sans établir de diagnostic médical.

## Fonctionnalités minimales (MVP)
1. Interface utilisateur (Streamlit)

Page Suivi : mini-tests cognitifs simples (mémoire d’images, suites logiques, reconnaissance).

Page Journal : saisie manuelle de l’humeur, du sommeil, de l’appétit et des interactions.

2. Base de données locale (pandas / CSV)

Stockage des résultats en local, sans serveur externe.

3. Analyse IA légère

Calcul de moyennes mobiles et de z-scores pour détecter les baisses de performance ou de moral.

Affichage de tendances visuelles.

4. Tableau de bord visuel

Graphiques de progression (matplotlib ou Plotly).

Messages d’interprétation simples, par exemple :
« L’attention semble légèrement plus faible cette semaine. »

## Stack technique

Langage : Python 3

Librairies : streamlit, pandas, scikit-learn, matplotlib

Hébergement : Hugging Face Spaces ou Streamlit Cloud

Exécution : 100 % sur CPU, sans GPU

Poids cible : < 100 Mo

Modules IA
Module “Routine”

Détection de changement de rythme à l’aide d’un modèle d’isolation (IsolationForest).

## Module “Cognition”

Évaluation de la performance cognitive : temps de réaction, erreurs, cohérence.

## Module “Visualisation”

Calcul de moyennes glissantes et affichage d’indicateurs colorés selon les variations.

## Aspects éthiques

Données conservées uniquement sur l’appareil local.

Aucune décision médicale, uniquement de la visualisation de tendances.

Mention claire : “outil expérimental, non médical”.