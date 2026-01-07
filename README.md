# Optimisation de Circuits Quantiques avec Algorithme Génétique

## Description
Ce projet implémente un algorithme génétique pour l'optimisation et l'approximation de circuits quantiques. L'objectif principal est d'utiliser des techniques évolutives pour générer des circuits quantiques qui s'approchent au mieux d'une matrice unitaire cible ou pour optimiser les paramètres de circuits existants.

Le projet utilise **Qiskit** pour la manipulation et la simulation des circuits quantiques, ainsi que des outils de calcul scientifique comme **NumPy** et **SciPy**.

## Installation

Pour installer les dépendances nécessaires, assurez-vous d'avoir Python installé, puis exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## Utilisation

Le projet est principalement structuré autour de notebooks Jupyter. Pour explorer et exécuter le code :

1. Lancez Jupyter Notebook ou JupyterLab :
   ```bash
   jupyter notebook
   ```
2. Ouvrez l'un des notebooks principaux, par exemple :
   - `AG_mono/code_ag.ipynb` : Algorithme génétique mono-objectif.
   - `NSGA-II/AG_multi_objectifs_VF.ipynb` : Algorithme génétique multi-objectifs (NSGA-II).
   - `final_test_AG/code_travaille copy 3.ipynb` : Tests finaux de l'algorithme d'optimisation.

## Structure du Projet
- `AG_mono/` : Implémentations mono-objectif.
- `NSGA-II/` : Implémentations multi-objectifs.
- `Final_test/` & `final_test_AG/` : Scripts de validation et tests de performance.
- `m1*/`, `m2*/` : Modules de test pour différents types de blocs de circuits.