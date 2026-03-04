# 📘 HOW TO RUN — Dossier `M1_finale`

Guide complet pour installer l'environnement et exécuter tous les scripts du dossier.

---

## 🗂️ Structure du dossier

```
M1_finale/
├── final_m1.ipynb          ← Notebook principal (pipeline complet)
├── final_m1_script.py      ← Version script du notebook (sans Jupyter)
├── run_m1_test.py          ← Script de test (circuit 30 qubits)
├── circuit_64qubits.py     ← ⭐ Nouveau : circuit faiblement connecté 64 qubits
├── setup_env.ps1           ← Script PowerShell de création de l'environnement
├── requirements_m1.txt     ← Liste des dépendances Python
├── HOW_TO_RUN.md           ← Ce fichier
└── out_figs/               ← Dossier de sortie des figures générées
```

---

## ⚙️ 1. Prérequis

| Outil | Version recommandée |
|-------|---------------------|
| Python | **≥ 3.10** |
| pip | dernière version |

---

## 🚀 2. Installation de l'environnement (une seule fois)

Ouvrez **PowerShell** dans le dossier `M1_finale`, puis exécutez :

```powershell
# Autoriser l'exécution de scripts (si nécessaire)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Lancer le script d'installation
.\setup_env.ps1
```

Ce script :
1. Crée un environnement virtuel `env_m1/`
2. Met à jour `pip`
3. Installe toutes les dépendances de `requirements_m1.txt`

---

## ▶️ 3. Activation de l'environnement (à chaque session)

```powershell
.\env_m1\Scripts\Activate.ps1
```

Votre terminal affichera `(env_m1)` pour confirmer l'activation.

---

## 🔬 4. Exécuter les scripts

### 4.1 — Notebook complet (Jupyter)
Lance le pipeline d'optimisation complet avec visualisations interactives.

```powershell
jupyter notebook final_m1.ipynb
```

### 4.2 — Script principal (sans Jupyter)
Version Python pure du notebook, identique en logique.

```powershell
python final_m1_script.py
```

### 4.3 — Test rapide (circuit 30 qubits)

```powershell
python run_m1_test.py
```

### 4.4 — ⭐ Nouveau : Circuit faiblement connecté 64 qubits

```powershell
python circuit_64qubits.py
```

**Ce que fait ce script :**
- Génère un circuit quantique **aléatoire à 64 qubits** avec une **très faible connectivité** :
  - Profondeur : 24 couches
  - Portes à 2 qubits : seulement 12 au total
  - Arêtes de connectivité : seulement 8 (sur 2016 possibles)
- Affiche les statistiques du circuit (taille, profondeur, portes 2-qubits)
- Sauvegarde le dessin du circuit → `out_figs/circuit_64q_original.png`
- Tente de lancer le pipeline d'optimisation via `final_m1_script.py`

**Paramètres modifiables** (en haut de `circuit_64qubits.py`) :

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `n_qubits` | 64 | Nombre de qubits |
| `depth` | 24 | Profondeur du circuit |
| `twoq_gates_total` | 12 | Nombre de portes 2-qubits |
| `connectivity_edges` | 8 | Nombre d'arêtes de topologie |
| `use_cz` | `True` | `True` = CZ, `False` = CNOT |
| `seed` | 2024 | Graine aléatoire (reproductibilité) |

---

## 🖼️ 5. Résultats et figures

Toutes les figures générées sont sauvegardées dans le dossier `out_figs/` :

| Fichier | Généré par |
|--------|------------|
| `circuit_original.png` | `final_m1_script.py` |
| `circuit_64q_original.png` | `circuit_64qubits.py` |
| `block_X_circuit_original.png` | pipeline d'optimisation |
| `optimized_block_X_circuit.png` | pipeline d'optimisation |

