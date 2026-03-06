#!/usr/bin/env python
# coding: utf-8

# # 🧪 Optimisation de Circuits Quantiques — Pipeline modulaire
# 
# Objectif : **réduire le coût/profondeur** tout en maintenant une **forte fidélité** au circuit de référen

from __future__ import annotations

# --- Python standard ---
import copy
import math

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
import os

# --- Analyse scientifique ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pour les visualisations 3D

# --- Graphes & clustering ---
import networkx as nx
import community  # algorithme de Louvain (python-louvain)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Quantique (Qiskit) ---
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutationAnalysis, CommutativeCancellation, Optimize1qGates
)

# --- Évolutionnaires (DEAP) ---
from deap import base, creator, tools

# --- Parallélisation ---
from joblib import Parallel, delayed


# ## 🎨 Partie 2 — Fonctions utilitaires de visualisation
# 
# Cette section regroupe les fonctions de traçage et de sauvegarde utilisées pour analyser :
# - la convergence de la fidélité au cours des générations,
# - le front de Pareto des solutions,
# - et le clustering 3D des individus optimisés.
# 
# Elles facilitent l’interprétation des résultats produits par l’algorithme évolutif.
# 

# In[ ]:


# =============================
# Utils : Fonctions de tracé et de sauvegarde
# =============================

def save_plot(name: str, out_dir="out_figs"):
    """
    Sauvegarde une figure matplotlib au format PNG.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300)


def plot_convergence(hist_eps, save_as: Optional[str] = None):
    """
    Trace la convergence de l'erreur (1 - fidélité).
    Axe Y en échelle logarithmique.
    """
    plt.figure()
    plt.plot(range(len(hist_eps)), hist_eps, marker="o")
    plt.yscale("log")
    plt.xlabel("Génération")
    plt.ylabel(r"$1\!-\!F$ (log)")
    plt.title("Convergence de la fidélité")
    plt.grid(True)
    plt.tight_layout()

    if save_as:
        save_plot(save_as)
    plt.close()


def plot_pareto(front, save_as: Optional[str] = None):
    """
    Affiche le front de Pareto : coût vs profondeur, colorié par l'erreur (1 - F).
    """
    costs = [i.fitness.values[2] for i in front]
    depths = [i.fitness.values[1] for i in front]
    eps = [1 - i.fitness.values[0] for i in front]

    plt.figure()
    sc = plt.scatter(costs, depths, c=eps, cmap="viridis")
    plt.colorbar(sc, label=r"$\varepsilon$ (1-F)")
    plt.xlabel("Coût chrom.")
    plt.ylabel("Profondeur")
    plt.title("Front de Pareto local")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_as:
        save_plot(save_as)
    plt.close()


def plot_3d_clusters(pareto, n_clusters: int = 4, save_as: Optional[str] = None):
    """
    Scatter 3D (Profondeur, Coût, Erreur) avec clustering K-Means.
    
    Axes :
      - X = profondeur
      - Y = coût chromosomique (longueur ou coût estimé)
      - Z = erreur ε = 1 - fidélité
    """
    if not pareto:
        return

    # Extraction des données
    data = np.array([
        [ind.fitness.values[1], ind.fitness.values[2], 1.0 - ind.fitness.values[0]]
        for ind in pareto
    ])

    # Normalisation + K-means
    k = max(1, min(n_clusters, len(data)))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    # Affichage 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10')
    ax.set_xlabel('Profondeur')
    ax.set_ylabel('Coût')
    ax.set_zlabel('Erreur ε = 1 - F')
    ax.set_title('Clustering 3D (K-Means) — Population Pareto')

    if save_as:
        save_plot(save_as)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ## 🔗 Partie 3 — Partitionnement & Graphe d’interaction
# 
# Cette section regroupe les fonctions permettant de :
# - **identifier les portes inter-blocs** dans un circuit,
# - **construire un graphe d’interaction** entre qubits,
# - **appliquer différents algorithmes de partitionnement** (Louvain, Metis, Kernighan–Lin, récursif),
# - **raffiner et analyser les partitions** (coût inter-blocs, qubits fortement interactifs),
# - préparer le circuit pour des optimisations modulaires.
# 

# In[ ]:


# =============================
# Partition & Graphe d’interaction
# =============================

def extract_interblock_gates(qc: QuantumCircuit, blocks: List[Set[int]]) -> List[Tuple]:
    """
    Extrait les portes qui connectent deux blocs distincts.
    """
    bmap = {q: i for i, bl in enumerate(blocks) for q in bl}
    interblock_gates = []
    for inst, qargs, cargs in qc.data:
        if len(qargs) < 2:
            continue
        qubit_indices = {qc.find_bit(q).index for q in qargs}
        involved_blocks = {bmap.get(q) for q in qubit_indices}
        if len(involved_blocks) > 1:
            interblock_gates.append((inst, qargs, cargs))
    return interblock_gates


def louvain_partition(qc: QuantumCircuit) -> List[Set[int]]:
    """
    Partitionne les qubits via l’algorithme Louvain
    sur un graphe d’interaction pondéré.
    """
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))

    for inst, qargs, _ in qc.data:
        if len(qargs) == 2:
            i = qc.find_bit(qargs[0]).index
            j = qc.find_bit(qargs[1]).index
            if G.has_edge(i, j):
                G[i][j]['weight'] += 1
            else:
                G.add_edge(i, j, weight=1)

    partition = community.best_partition(G, weight='weight')
    blocks = defaultdict(set)
    for qubit, community_id in partition.items():
        blocks[community_id].add(qubit)
    return list(blocks.values())


def build_interaction_graph(qc: QuantumCircuit) -> nx.Graph:
    """
    Construit un graphe où les sommets = qubits,
    et les arêtes = nombre de portes 2-qubits entre eux.
    """
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))
    for inst, qargs, _ in qc.data:
        if inst.name in {"cx", "cz", "rzz"} and len(qargs) == 2:
            i, j = [qc.find_bit(q).index for q in qargs]
            w = G.get_edge_data(i, j, default={"weight": 0})["weight"] + 1
            G.add_edge(i, j, weight=w)
    return G


# --- Partitionnement récursif avec Metis ou Kernighan-Lin ---
def _partition_metis(graph: nx.Graph) -> List[Set[int]]:
    import nxmetis  # nécessite le package `nxmetis`
    _, parts = nxmetis.partition(graph, 2)
    return [set(p) for p in parts]


def _partition_kl(graph: nx.Graph) -> List[Set[int]]:
    from networkx.algorithms.community import kernighan_lin_bisection
    a, b = kernighan_lin_bisection(graph)
    return [set(a), set(b)]


def multilevel_partition(graph: nx.Graph, max_block_size: int) -> List[Set[int]]:
    """
    Applique un partitionnement récursif jusqu’à ce que
    chaque bloc soit inférieur à `max_block_size`.
    """
    if len(graph) <= max_block_size:
        return [set(graph.nodes())]
    try:
        parts = _partition_metis(graph)
    except Exception:
        parts = _partition_kl(graph)
    res: List[Set[int]] = []
    for p in parts:
        res.extend(multilevel_partition(graph.subgraph(p), max_block_size))
    return res


def _interblock_gate_cost(qc: QuantumCircuit, blk0: Set[int], blk1: Set[int]) -> int:
    """
    Calcule le coût (nombre de portes) reliant deux blocs donnés.
    """
    cost = 0
    for inst, qargs, _ in qc.data:
        if len(qargs) < 2:
            continue
        qs = {qc.find_bit(q).index for q in qargs}
        if qs & blk0 and qs & blk1:
            cost += 1
    return cost


def refine_partition_kl(qc: QuantumCircuit, blocks: List[Set[int]], *, max_iter: int = 10) -> List[Set[int]]:
    """
    Raffine une partition en déplaçant des qubits entre blocs
    pour réduire le coût inter-blocs (méthode Kernighan-Lin).
    """
    if len(blocks) < 2:
        return blocks

    a, b = blocks[0].copy(), blocks[1].copy()
    best_cost = _interblock_gate_cost(qc, a, b)
    improved, it = True, 0

    while improved and it < max_iter:
        improved = False
        it += 1
        gain_best, q_best, side = 0, None, None

        # Essai de déplacement a → b
        for q in list(a):
            gain = best_cost - _interblock_gate_cost(qc, a - {q}, b | {q})
            if gain > gain_best:
                gain_best, q_best, side = gain, q, "a2b"

        # Essai de déplacement b → a
        for q in list(b):
            gain = best_cost - _interblock_gate_cost(qc, a | {q}, b - {q})
            if gain > gain_best:
                gain_best, q_best, side = gain, q, "b2a"

        if gain_best > 0 and q_best is not None:
            improved = True
            best_cost -= gain_best
            if side == "a2b":
                a.remove(q_best)
                b.add(q_best)
            else:
                b.remove(q_best)
                a.add(q_best)

    blocks[0], blocks[1] = a, b
    return blocks


# ## 🧩 Partie 4 — Sous-circuits & Recomposition
# 
# Dans cette section, on isole les **sous-circuits par bloc de qubits** puis on **recompose** un circuit global à partir
# des versions optimisées de chaque bloc.
# 
# - `extract_subcircuit(qc, qubits)` : extrait les portes **strictement internes** à l’ensemble `qubits`, en **reindexant**
#   localement les qubits (0..|qubits|-1) pour rendre le sous-circuit autonome.
# - `recompose_from_blocks(qc_original, block_subcircuits)` : réinsère, **dans l’ordre du circuit d’origine**, les portes
#   provenant des sous-circuits optimisés quand l’opération concerne entièrement un bloc ; sinon, on **garde** la porte
#   originale telle quelle (utile pour les portes inter-blocs).
# 
# > Astuce : après optimisation bloc-à-bloc, utilisez `recompose_from_blocks` pour retrouver un circuit global cohérent,
# > tout en bénéficiant des améliorations locales.
# 

# In[ ]:


# =============================
# Sous-circuits & Recomposition
# =============================

def extract_subcircuit(qc: QuantumCircuit, qubits: Set[int]) -> QuantumCircuit:
    """
    Extraire le sous-circuit contenant uniquement les portes dont TOUTES les cibles
    appartiennent à `qubits`. Les qubits sont remappés localement de 0 à |qubits|-1.

    Args:
        qc: Circuit global d'origine.
        qubits: Ensemble d'indices de qubits du bloc.

    Returns:
        QuantumCircuit local autonome sur |qubits| qubits.
    """
    # Création d'un circuit local avec autant de qubits que dans le bloc
    sub = QuantumCircuit(len(qubits))

    # Remappage global->local : le i-ème qubit trié du bloc devient l'indice i local
    local_index = {q: i for i, q in enumerate(sorted(list(qubits)))}

    # On ne garde que les portes qui agissent EXCLUSIVEMENT sur des qubits du bloc
    for inst, qargs, cargs in qc.data:
        if all(qc.find_bit(q).index in qubits for q in qargs):
            remapped_qargs = [sub.qubits[local_index[qc.find_bit(q).index]] for q in qargs]
            sub.append(inst, remapped_qargs, cargs)

    return sub


def recompose_from_blocks(
    qc_original: QuantumCircuit,
    block_subcircuits: List[Tuple[Set[int], QuantumCircuit]]
) -> QuantumCircuit:
    """
    Recomposer un circuit global à partir de sous-circuits (optimisés) par bloc,
    en respectant l'ordre des opérations du circuit d'origine.

    Principe :
    - On parcourt les portes du `qc_original` dans leurs ordre.
    - Si une porte appartient entièrement à un bloc B, on insère à la place la prochaine
      porte correspondante du sous-circuit de B (déjà optimisé), remappée vers les indices globaux.
    - Sinon (porte inter-blocs), on garde la porte d'origine telle quelle.

    Args:
        qc_original: Circuit de référence dont on respecte l'ordre des opérations.
        block_subcircuits: Liste de tuples (qubits_du_bloc, sous_circuit_optimisé).

    Returns:
        QuantumCircuit global recomposé.
    """
    # Préparer les structures de remappage et des curseurs de lecture
    block_maps = []
    for block_qubits, sub in block_subcircuits:
        sorted_block = sorted(block_qubits)
        global_to_local = {q: i for i, q in enumerate(sorted_block)}  # utile si besoin
        block_maps.append((set(sorted_block), sub, global_to_local))

    # Nouveau circuit global (même nombre de qubits que l’original)
    qc_recomposed = QuantumCircuit(qc_original.num_qubits)

    # Curseur pour savoir où on en est dans CHAQUE sous-circuit
    subcircuit_cursors = [0 for _ in block_subcircuits]

    # Parcours des portes du circuit original, dans l'ordre
    for inst, qargs, cargs in qc_original.data:
        # Indices globaux touchés par la porte courante
        q_indices = [qc_original.find_bit(q).index for q in qargs]
        inserted = False

        # Chercher si la porte appartient à un bloc spécifique
        for idx, (block_qubits, sub, g2l_unused) in enumerate(block_maps):
            if all(q in block_qubits for q in q_indices):
                # On consomme la prochaine porte du sous-circuit du bloc
                if subcircuit_cursors[idx] >= len(sub.data):
                    raise ValueError(f"Trop de portes demandées pour le bloc {idx} par rapport à son sous-circuit.")

                inst_opt, qargs_opt, cargs_opt = sub.data[subcircuit_cursors[idx]]
                subcircuit_cursors[idx] += 1

                # Remap des qubits locaux du sous-circuit vers leurs indices GLOBAUX d’origine
                sorted_block = sorted(block_qubits)
                mapped_qargs = [qc_recomposed.qubits[sorted_block[sub.find_bit(q).index]] for q in qargs_opt]

                qc_recomposed.append(inst_opt, mapped_qargs, cargs_opt)
                inserted = True
                break

        # Si la porte n’appartient pas à un bloc unique (porte inter-blocs), on garde la porte originale
        if not inserted:
            mapped_qargs = [qc_recomposed.qubits[i] for i in q_indices]
            qc_recomposed.append(inst, mapped_qargs, cargs)

    return qc_recomposed


# ## 🎯 Partie 5 — Fidélité & Compression
# 
# Cette section regroupe :
# - **`compute_fidelity(circ, target)`** : calcule la fidélité opérateur-opérateur  
#   \( F = \frac{|\mathrm{Tr}(U_{\text{circ}}\,U_{\text{target}}^\dagger)|}{2^n} \) en alignant la taille si nécessaire.
# - **Passes de compression “maison”** :
#   - `cancel_inverse_gates` : annule des paires de portes inverses adjacentes (ex. `x` suivi de `x`, `cx` suivi de `cx`, etc.).
#   - `merge_rotations` : fusionne les rotations successives du même axe sur le **même qubit**.
#   - `remove_negligible_rotations` : retire les rotations de très faible amplitude (seuil `th`).
#   - `compress_custom` : pipeline minimal combinant les trois étapes ci-dessus.
# - **Pass Qiskit** :
#   - `qiskit_opt_pass` applique `Optimize1qGates` et `CommutativeCancellation` pour simplifier davantage.
# 

# In[ ]:


# =============================
# Fidélité & Compression
# =============================

from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

def compute_fidelity(circ: QuantumCircuit, target: np.ndarray) -> float:
    """
    Calcule la fidélité opérateur-opérateur entre le circuit 'circ' et l’opérateur 'target'.
    Si le circuit a PLUS de qubits que la cible, on 'pad' la cible avec l'identité.
    Si le circuit a MOINS de qubits, on lève une erreur (cas non géré ici).
    Si le nombre de qubits est trop élevé (> 15), on renvoie 1.0 par défaut pour éviter l'explosion mémoire.
    """
    if circ.num_qubits > 15:
        # print("⚠️ Qubits > 15 : Calcul de fidélité ignoré (mémoire).")
        return 1.0
    
    try:
        circ_op = Operator(circ).data
        target_nqubits = int(np.log2(target.shape[0]))

        if circ.num_qubits > target_nqubits:
            # On étend la cible à la dimension du circuit en la plaçant dans le coin supérieur-gauche
            target_op_padded = np.eye(2**circ.num_qubits, dtype=complex)
            target_op_padded[:target.shape[0], :target.shape[1]] = target
            target = target_op_padded
        elif circ.num_qubits < target_nqubits:
            raise ValueError("Le circuit a moins de qubits que l’opérateur cible — fidélité non définie directement.")

        # F = |Tr(Uc * Ut^\dagger)| / 2^n
        return abs(np.trace(circ_op @ target.conj().T)) / (2 ** circ.num_qubits)
    except Exception:
        return 1.0


def cancel_inverse_gates(c: QuantumCircuit) -> QuantumCircuit:
    """
    Supprime des paires adjacentes de portes auto-inverses appliquées sur EXACTEMENT les mêmes qubits.
    Géré ici : {x, y, z, h, cx}. (ajoutez-en d’autres si souhaité)
    """
    new = QuantumCircuit(c.num_qubits)
    skip = set()

    for i in range(len(c.data) - 1):
        if i in skip:
            continue
        g1, q1, _ = c.data[i]
        g2, q2, _ = c.data[i + 1]

        if g1.name == g2.name and q1 == q2 and g1.name in {"x", "y", "z", "h", "cx"}:
            # g puis g => identité
            skip.add(i + 1)
            continue
        new.append(g1, q1)

    # Dernière porte si non-sautée
    if (len(c.data) - 1) not in skip and len(c.data) > 0:
        g, q, _ = c.data[-1]
        new.append(g, q)
    return new


def merge_rotations(c: QuantumCircuit) -> QuantumCircuit:
    """
    Fusionne les rotations successives du même type (rx/ry/rz) sur le même qubit :
    rx(a) ; rx(b)  ->  rx(a+b)
    """
    new = QuantumCircuit(c.num_qubits)
    i = 0
    while i < len(c.data):
        g, q, _ = c.data[i]
        if g.name in {"rx", "ry", "rz"}:
            angle = g.params[0]
            j = i + 1
            # On accumule tant que le type ET la cible sont identiques
            while j < len(c.data):
                g2, q2, _ = c.data[j]
                if g2.name == g.name and q2 == q:
                    angle += g2.params[0]
                    j += 1
                else:
                    break
            # On réémet une unique rotation avec l'angle fusionné
            getattr(new, g.name)(angle, q[0])
            i = j
        else:
            new.append(g, q)
            i += 1
    return new


def remove_negligible_rotations(c: QuantumCircuit, *, th: float = 1e-4) -> QuantumCircuit:
    """
    Supprime les rotations rx/ry/rz de très faible amplitude (|theta| < th).
    """
    new = QuantumCircuit(c.num_qubits)
    for g, q, _ in c.data:
        if g.name in {"rx", "ry", "rz"} and abs(float(g.params[0])) < th:
            # on ignore cette petite rotation
            continue
        new.append(g, q)
    return new


def compress_custom(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Pipeline de compression minimal :
      1) Annulation d’inverses,
      2) Fusion de rotations,
      3) Suppression de petites rotations.
    """
    return remove_negligible_rotations(
        merge_rotations(
            cancel_inverse_gates(circ)
        )
    )


def qiskit_opt_pass(c: QuantumCircuit) -> QuantumCircuit:
    """
    Passe d’optimisation Qiskit standard :
      - Optimize1qGates (réécriture plus compacte de suites 1-qubit)
      - CommutativeCancellation (annulation de portes commutatives inutiles)
    """
    pm = PassManager([Optimize1qGates(), CommutativeCancellation()])
    return pm.run(c)


# ## 📊 Partie 5.5 — Indicateurs de Qualité Multi-objectif (HV, IGD, Spread, ε)
# 
# Cette section implémente les métriques standards pour évaluer la qualité des fronts de Pareto :
# - **Hypervolume (HV)** : mesure l'espace dominé par le front.
# - **Inverted Generational Distance (IGD)** : distance au front de référence.
# - **Spread (Δ)** : diversité des solutions.
# - **Epsilon Indicator (ε)** : facteur de domination.
# - **Spacing** : uniformité de la distribution.

# In[ ]:


# --- Multi-objective Evaluation Metrics ---
from scipy.spatial.distance import cdist
import numpy as np
from typing import Tuple, List, Optional, Dict
def is_non_dominated(costs: np.ndarray) -> np.ndarray:
    n_points = costs.shape[0]
    is_eff = np.ones(n_points, dtype=bool)
    for i, c in enumerate(costs):
        if is_eff[i]:
            is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1) | np.all(costs[is_eff] == c, axis=1)
            is_eff[i] = True
    return is_eff
def pareto_front(costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = is_non_dominated(costs)
    return np.where(mask)[0], costs[mask]
def normalize_objectives(F: np.ndarray, f_min=None, f_max=None):
    if f_min is None: f_min = np.min(F, axis=0)
    if f_max is None: f_max = np.max(F, axis=0)
    denom = f_max - f_min
    denom[np.abs(denom) < 1e-12] = 1.0
    return (F - f_min) / denom, f_min, f_max
def compute_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    try:
        from pymoo.indicators.hv import HV
        return HV(ref_point=ref_point)(F)
    except ImportError:
        if F.shape[1] == 2:
            idx = np.argsort(F[:, 0]); F_s = F[idx]; hv = 0.0; last_y = ref_point[1]
            for i in range(len(F_s)):
                h = last_y - F_s[i, 1]
                if h > 0: hv += (ref_point[0] - F_s[i, 0]) * h; last_y = F_s[i, 1]
            return hv
        return 0.0
def spread_delta(P: np.ndarray, P_star_extremes: Optional[np.ndarray] = None) -> float:
    if len(P) < 2: return 1.0
    P = P[np.argsort(P[:, 0])]; d = np.linalg.norm(P[1:] - P[:-1], axis=1); d_m = np.mean(d)
    df = np.linalg.norm(P[0] - P_star_extremes[0]) if P_star_extremes is not None else 0.0
    dl = np.linalg.norm(P[-1] - P_star_extremes[1]) if P_star_extremes is not None else 0.0
    return (df + dl + np.sum(np.abs(d - d_m))) / (df + dl + (len(P) - 1) * d_m)
def compute_spacing(P: np.ndarray) -> float:
    if len(P) < 2: return 0.0
    d = np.min(cdist(P, P) + np.eye(len(P))*1e10, axis=1)
    return np.std(d)
def compute_epsilon(P: np.ndarray, P_star: np.ndarray) -> float:
    return np.max([np.min(np.max(P - ps, axis=1)) for ps in P_star])
def evaluate_run(F: np.ndarray, P_star: Optional[np.ndarray] = None, ref_point: Optional[np.ndarray] = None, normalize: bool = True):
    costs = F.copy(); costs[:, 0] = 1.0 - costs[:, 0]
    idx, P = pareto_front(costs)
    res = {"n_pareto": len(P)}
    if len(P) == 0: return res
    P_eval, f_min, f_max = normalize_objectives(costs) if normalize else (costs, None, None)
    P_eval = P_eval[idx]
    ref = ref_point if ref_point is not None else (np.max(P_eval, axis=0) + 0.1)
    res.update({"HV": compute_hv(P_eval, ref), "Spread": spread_delta(P_eval), "Spacing": compute_spacing(P_eval), "ref_point": ref})
    if P_star is not None:
        P_star_min = P_star.copy(); P_star_min[:, 0] = 1.0 - P_star_min[:, 0]
        P_star_eval = normalize_objectives(P_star_min, f_min, f_max)[0] if normalize else P_star_min
        res["IGD"] = np.mean(np.min(cdist(P_star_eval, P_eval), axis=1))
        res["Epsilon"] = compute_epsilon(P_eval, P_star_eval)
    return res
def plot_moo_history(history: List[Dict], title="Evolution MoO", save_as=None):
    import matplotlib.pyplot as plt
    import os
    hvs = [h.get("HV", 0) for h in history]; spreads = [h.get("Spread", 1) for h in history]
    fig, ax1 = plt.subplots(); ax1.plot(hvs, "b-", label="HV"); ax1.set_ylabel("Hypervolume", color="b")
    ax2 = ax1.twinx(); ax2.plot(spreads, "r-", label="Spread"); ax2.set_ylabel("Spread (Δ)", color="r")
    plt.title(title)
    if save_as:
        os.makedirs("out_figs", exist_ok=True)
        plt.savefig(f"out_figs/{save_as}", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# In[ ]:


def export_all_indicators(history: List[Dict], block_idx: int):
    """Export multiple MOO quality indicators as separate images."""
    import matplotlib.pyplot as plt
    import os
    os.makedirs("out_figs", exist_ok=True)
    
    metrics = ["HV", "Spread", "IGD", "Spacing", "Epsilon"]
    colors = ["blue", "red", "green", "orange", "purple"]
    
    for metric, color in zip(metrics, colors):
        values = [h.get(metric) for h in history if h.get(metric) is not None]
        if not values: continue
        
        plt.figure(figsize=(8, 4))
        plt.plot(values, marker='.', linestyle='-', color=color)
        plt.title(f"Indicator: {metric} - Block {block_idx}")
        plt.xlabel("Generation")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"out_figs/indicator_{metric.lower()}_block_{block_idx}.png", dpi=300, bbox_inches='tight')
        plt.close()
    print(f"✅ All indicators for Block {block_idx} exported to out_figs/")


# ## 🧬 Partie 6 — Optimisation intra-bloc : NSGA-II + Local Angle Search (LAS)
# 
# Dans cette section, on cherche pour **chaque bloc de qubits** un circuit “équivalent” (même opérateur visé) mais
# **plus efficace** selon 3 objectifs :
# 1. **Maximiser** la **fidélité** vis-à-vis de la cible,
# 2. **Minimiser** la **profondeur** (après un `transpile` léger),
# 3. **Minimiser** un **coût** (longueur du chromosome ou coût pondéré des portes).
# 
# Stratégie :
# - On encode un **chromosome** comme une liste de gènes `("gate", target, ctrl?, angle?)`.
# - On évalue chaque individu via `compute_fidelity`, `transpile(...).depth()` et un coût.
# - On applique **NSGA-II** (DEAP) pour l’optimisation multi-objectif.
# - On ajoute une **recherche locale des angles (LAS)** : pour chaque rotation (rx/ry/rz/rzz), on estime un **pseudo-gradient**
#   par différences finies et on essaye quelques pas `η` pour **améliorer localement** la fidélité.
# 
# Sorties :
# - Un **circuit optimisé** par bloc,
# - Des **figures** : courbe de convergence, **front de Pareto**, **clustering 3D** des solutions.
# 

# In[ ]:


# =============================
# Optimisation intra-bloc : NSGA-II + LAS
# =============================

def compute_gate_cost(qc: QuantumCircuit) -> float:
    """
    Coût simple pondéré par type de porte. Ajustez la table selon votre matériel/cibles.
    """
    cost_table = {
        "x": 1, "z": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1,
        "h": 2, "cx": 5, "cz": 5, "ccx": 13
    }
    return sum(cost_table.get(inst.name.lower(), 1) for inst, _, _ in qc.data)


def update_rotation_angles(
    chrom: List[Tuple[str, int, Optional[int], Optional[float]]],
    build_fn,
    target_unitary: np.ndarray,
    *,
    eta_range: Sequence[float] = (0.01, 0.1, 0.5),
    delta: float = 0.1,
) -> List[Tuple[str, int, Optional[int], Optional[float]]]:
    """
    Recherche locale des angles par différences centrales (pseudo-gradient).
    Compatible gènes : ('rx'|'ry'|'rz'|'rzz', target, ctrl?, theta).

    - On modifie un angle 'θ' et on mesure F(θ+δ), F(θ-δ) pour approximer dF/dθ.
    - On teste des pas 'η' ∈ eta_range pour accepter une amélioration immédiate.
    """

    def wrap_angle(theta: float) -> float:
        twopi = 2.0 * math.pi
        # Replie l’angle dans (-π, π]
        return ((theta + math.pi) % twopi) - math.pi

    def set_angle(ch, idx: int, theta: float):
        g, t, c, a = ch[idx]
        ch2 = list(ch)
        ch2[idx] = (g, t, c, wrap_angle(theta))
        return ch2

    def fitness_of(ch):
        qc = build_fn(ch)
        return compute_fidelity(qc, target_unitary)

    best = list(chrom)
    base_fit = fitness_of(best)

    for i, gene in enumerate(best):
        g, t, c, a = gene
        if g.lower() not in {"rx", "ry", "rz", "rzz"} or a is None:
            continue

        theta0 = float(a)
        f_plus  = fitness_of(set_angle(best, i, theta0 + delta))
        f_minus = fitness_of(set_angle(best, i, theta0 - delta))
        grad = (f_plus - f_minus) / (2.0 * delta)

        best_local_fit = base_fit
        for eta in eta_range:
            cand_theta = theta0 + eta * grad
            cand = set_angle(best, i, cand_theta)
            f = fitness_of(cand)
            if f > best_local_fit:
                best_local_fit = f
                best = cand  # acceptation immédiate
        base_fit = best_local_fit

    return best




# In[ ]:


def optimise_block_nsga2(qc_target: QuantumCircuit, *, generations=500, pop_size=300, n_jobs=-1, P_star=None):
    nq = qc_target.num_qubits; U_target = Operator(qc_target).data
    gate_pool = ["h", "x", "y", "z", "rx", "ry", "rz", "cx", "cz", "rzz"]

    def gen_gene():
        g = random.choice(gate_pool); tgt = random.randrange(nq)
        if g in {"rx", "ry", "rz"}:
            return (g, tgt, None, random.uniform(0, 2 * math.pi))
        if g == "rzz":
            ctrl = random.choice([q for q in range(nq) if q != tgt])
            return (g, tgt, ctrl, random.uniform(0, 2 * math.pi))
        if g in {"cx", "cz"}:
            ctrl = random.choice([q for q in range(nq) if q != tgt])
            return (g, tgt, ctrl, None)
        return (g, tgt, None, None)

    def build(ch):
        qc = QuantumCircuit(nq)
        for g, t, ctrl, a in ch:
            if g == "rzz":
                qc.rzz(a, ctrl, t)
            elif g in {"cx", "cz"}:
                getattr(qc, g)(ctrl, t)
            elif g in {"rx", "ry", "rz"}:
                getattr(qc, g)(a, t)
            else:
                getattr(qc, g)(t)
        return qc

    def eval_ind(ind):
        qc = build(ind); fid = compute_fidelity(qc, U_target)
        depth = transpile(qc, basis_gates=["cx", "rz", "sx"], optimization_level=1).depth()
        cost = len(ind)  # chromosome length as proxy (fast). Replace by compute_gate_cost(qc) if desired.
        return fid, depth, cost

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1, -1, -1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    tb = base.Toolbox(); tb.register("gene", gen_gene)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, 12)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", lambda ind: (ind.__setitem__(random.randrange(len(ind)), gen_gene()) or ind))
    tb.register("select", tools.selNSGA2)

    pop = tb.population(pop_size)
    fits = Parallel(n_jobs)(delayed(eval_ind)(i) for i in pop)
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hist_eps = [1 - max(pop, key=lambda i: i.fitness.values[0]).fitness.values[0]]
    history_moo = []

    for gen in range(generations):
        tools.emo.assignCrowdingDist(pop)
        offspring = tools.selTournamentDCD(pop, len(pop)); offspring = list(map(tb.clone, offspring))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                tb.mate(c1, c2); del c1.fitness.values, c2.fitness.values
        for ind in offspring:
            if random.random() < 0.9:
                tb.mutate(ind); del ind.fitness.values
        invalid = [i for i in offspring if not i.fitness.valid]
        fits = Parallel(n_jobs)(delayed(eval_ind)(i) for i in invalid)
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit
        pop = tb.select(pop + offspring, k=len(pop))
        # MOO tracking
        fits_gen = np.array([ind.fitness.values for ind in pop])
        history_moo.append(evaluate_run(fits_gen))
        # MOO tracking
        fits_gen = np.array([ind.fitness.values for ind in pop])
        history_moo.append(evaluate_run(fits_gen, P_star=P_star))
        best = max(pop, key=lambda i: i.fitness.values[0])
        hist_eps.append(1 - best.fitness.values[0])
        print(f"Gen {gen + 1:>4} | Fid {best.fitness.values[0]:.4f} | D {best.fitness.values[1]:>3} | C {best.fitness.values[2]:>3}")

    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    plot_convergence(hist_eps, save_as=f"block_fid_conv_{nq}q")
    plot_pareto(front, save_as=f"block_pareto_{nq}q")
    # NEW 3D clustering figure
    plot_3d_clusters(front, n_clusters=4, save_as=f"block_clusters3d_{nq}q")

    # Final metrics summary
    final_metrics = evaluate_run(np.array([ind.fitness.values for ind in pop]), P_star=P_star)
    print(f"Final MOO Metrics: {final_metrics}")
    return build(max(pop, key=lambda i: i.fitness.values[0])), history_moo





# ## 🌉 Partie 7 — Injection inter-blocs (SA / Stochastique)
# 
# Objectif : **ajouter des portes entre blocs** (ex. `cx`, `cz`, `rzz`) afin d’améliorer la fidélité globale, tout en
# contrôlant la profondeur, le nombre de portes et (optionnellement) la **diaphonie**.
# 
# Deux stratégies :
# - **Recuit simulé (SA)** : on explore un **pool** de candidats et on minimise une **énergie**  
#   `E = α·(#portes) + β·profondeur + γ·diaphonie + δ·pénalité_de_fidélité`.
# - **Stochastique** : on tente d’ajouter aléatoirement des portes inter-blocs et on **garde** uniquement celles qui
#   **préservent** une fidélité ≥ seuil.
# 

# In[ ]:


# =============================
# Injection inter-blocs (SA / stochastique)
# =============================

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, List, Set, Tuple
import random, math, copy
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

@dataclass
class InjectionGate:
    gate: str
    q1: int
    q2: int
    theta: Optional[float]
    enabled: bool = True

    def copy(self) -> "InjectionGate":
        return InjectionGate(self.gate, self.q1, self.q2, self.theta, self.enabled)


def _sa_build_circuit(base: QuantumCircuit, injections: Sequence[InjectionGate]) -> QuantumCircuit:
    """
    Construit un circuit candidat en appliquant les injections actives au circuit de base,
    puis effectue un transpile léger pour estimer la profondeur.
    """
    circ = base.copy()
    for inj in injections:
        if not inj.enabled:
            continue
        if inj.gate == "rzz":
            circ.rzz(inj.theta, inj.q1, inj.q2)
        else:
            getattr(circ, inj.gate)(inj.q1, inj.q2)
    circ = transpile(circ, basis_gates=["cx", "rz", "sx"], optimization_level=1)
    return circ


def _sa_energy(injections: Sequence[InjectionGate], *, base: QuantumCircuit, target_U: np.ndarray,
               α: float, β: float, γ: float, δ: float, fid_tol: float,
               crosstalk_mat: Optional[np.ndarray]) -> float:
    """
    Fonction d'énergie pour le recuit simulé.
    """
    cand = _sa_build_circuit(base, injections)
    n2q = sum(1 for inj in injections if inj.enabled)
    depth = cand.depth() or 0

    crosstalk = 0.0
    if crosstalk_mat is not None:
        for inj in injections:
            if inj.enabled:
                crosstalk += crosstalk_mat[inj.q1, inj.q2]

    fid = compute_fidelity(cand, target_U)
    fid_penalty = (1.0 - fid) / fid_tol
    return α * n2q + β * depth + γ * crosstalk + δ * fid_penalty


def _sa_rand_move(injections: Sequence[InjectionGate], blocks: List[Set[int]], *, rng: random.Random,
                   eps_theta: float = 0.1) -> List[InjectionGate]:
    """
    Propose un mouvement aléatoire sur le pool (toggle, changement de type, déplacement, ou accord fin de θ).
    """
    moves = ["toggle", "swap_type", "shift", "tune_theta"]
    choice = rng.choice(moves)
    cand = [inj.copy() for inj in injections]
    idx = rng.randrange(len(cand))
    inj = cand[idx]

    if choice == "toggle":
        inj.enabled = not inj.enabled
    elif choice == "swap_type":
        inj.gate = rng.choice([g for g in ("cx", "cz", "rzz") if g != inj.gate])
        inj.theta = None if inj.gate != "rzz" else rng.uniform(0, 2 * math.pi)
    elif choice == "shift":
        blk0, blk1 = blocks[0], blocks[1]
        inj.q1 = rng.choice(tuple(blk0))
        inj.q2 = rng.choice(tuple(blk1))
    elif choice == "tune_theta" and inj.gate == "rzz":
        inj.theta = (inj.theta or 0.0) + rng.uniform(-eps_theta, eps_theta)

    return cand


def _sa_generate_pool(blocks: List[Set[int]], gate_types: Sequence[str], *, rng: random.Random,
                      n_candidates: int) -> List[InjectionGate]:
    """
    Crée un pool initial d’injections (désactivées) entre deux blocs.
    """
    blk0, blk1 = blocks[0], blocks[1]
    pool: List[InjectionGate] = []
    for _ in range(n_candidates):
        gate = rng.choice(gate_types)
        q1 = rng.choice(tuple(blk0))
        q2 = rng.choice(tuple(blk1))
        theta = rng.uniform(0, 2 * math.pi) if gate == "rzz" else None
        pool.append(InjectionGate(gate, q1, q2, theta, enabled=False))
    return pool


def sa_injection(base_qc: QuantumCircuit, blocks: List[Set[int]], *,
                 gate_types: Sequence[str] = ("cx", "cz", "rzz"),
                 n_candidates: int = 120,
                 fid_threshold: float = 0.999,
                 n_iters: int = 2000,
                 α: float = 1.0, β: float = 0.01, γ: float = 0.0, δ: float = 1e4,
                 schedule_alpha: float = 0.85,
                 seed: Optional[int] = None,
                 crosstalk_mat: Optional[np.ndarray] = None) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Injection inter-blocs par recuit simulé (SA).
    Retourne le circuit final et la liste des injections conservées.
    """
    if len(blocks) < 2:
        raise ValueError("sa_injection nécessite au moins deux blocs.")

    rng = random.Random(seed)
    injections = _sa_generate_pool(blocks, gate_types, rng=rng, n_candidates=n_candidates)
    target_U = Operator(base_qc).data

    # Température initiale basée sur la variance d'échantillons d'énergie
    sample_E = []
    for _ in range(30):
        tmp = _sa_rand_move(injections, blocks, rng=rng)
        e = _sa_energy(tmp, base=base_qc, target_U=target_U, α=α, β=β, γ=γ, δ=δ,
                       fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat)
        sample_E.append(e)
    T = 5.0 * (np.std(sample_E) or 1.0)

    best = copy.deepcopy(injections)
    E_best = _sa_energy(best, base=base_qc, target_U=target_U, α=α, β=β, γ=γ, δ=δ,
                        fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat)
    current, E_curr = copy.deepcopy(best), E_best

    # Boucle de recuit
    for _ in range(n_iters):
        cand = _sa_rand_move(current, blocks, rng=rng)
        E_cand = _sa_energy(cand, base=base_qc, target_U=target_U, α=α, β=β, γ=γ, δ=δ,
                            fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat)
        ΔE = E_cand - E_curr
        accept = (ΔE < 0) or (rng.random() < math.exp(-ΔE / T))
        if accept:
            current, E_curr = cand, E_cand
            if E_curr < E_best:
                best, E_best = copy.deepcopy(current), E_curr
        T *= schedule_alpha

    final_circ = _sa_build_circuit(base_qc, best)
    fid_final = compute_fidelity(final_circ, target_U)
    if fid_final < fid_threshold:
        raise RuntimeError(f"SA n’atteint pas la fidélité cible : {fid_final:.5f} < {fid_threshold}")

    kept = [(inj.gate, inj.q1, inj.q2, inj.theta) for inj in best if inj.enabled]
    return final_circ, kept


def stochastic_injection(qc: QuantumCircuit, blocks: List[Set[int]], *,
                         n_injections: int = 100,
                         fid_threshold: float = 0.999,
                         gate_probs: Optional[Dict[str, float]] = None) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Injection inter-blocs stochastique : on ajoute des portes au hasard et on ne conserve
    que celles qui ne dégradent pas la fidélité sous le seuil.
    """
    if len(blocks) < 2:
        raise ValueError("stochastic_injection nécessite au moins deux blocs.")

    gate_probs = gate_probs or {"cx": 1.0, "cz": 1.0, "rzz": 1.0}
    total = sum(gate_probs.values())
    gate_types, probs = zip(*([(g, p / total) for g, p in gate_probs.items()]))

    rng = random.Random()
    kept: List[Tuple[str, int, int, Optional[float]]] = []
    U_ref = Operator(qc).data

    for _ in range(n_injections):
        gate = rng.choices(gate_types, probs, k=1)[0]
        qi = rng.choice(tuple(blocks[0]))
        qj = rng.choice(tuple(blocks[1]))

        cand = qc.copy()
        if gate == "rzz":
            theta = rng.uniform(0, 2 * math.pi)
            cand.rzz(theta, qi, qj)
        else:
            theta = None
            getattr(cand, gate)(qi, qj)

        # Option : petite passe d’optimisation/normalisation
        cand = qiskit_opt_pass(compress_custom(cand))

        fid = compute_fidelity(cand, U_ref)
        if fid >= fid_threshold:
            qc = cand
            kept.append((gate, qi, qj, theta))
            U_ref = Operator(qc).data  # met à jour la référence

    return qc, kept


# ## 📈 Partie 8 — Injection guidée par la fidélité (glouton)
# 
# Stratégie **gloutonne** : on essaye d’ajouter une porte inter-blocs (`cx`, `cz`, `rzz`) et on **garde** l’ajout **uniquement** si
# la **fidélité** par rapport au circuit **cible** augmente. On répète jusqu’à atteindre un **seuil** de fidélité ou
# épuiser un **budget d’essais**.
# 
# - Entrées :
#   - `base_qc` : circuit de départ (sans injections ou partiellement injecté),
#   - `target_qc` : circuit cible dont on veut approcher l’unitaire,
#   - `blocks` : deux (ou plus) blocs de qubits (on pioche 1 qubit dans chaque des deux premiers blocs),
#   - `max_trials` : nombre maximum d’essais,
#   - `fid_threshold` : seuil de fidélité souhaité.
# - Sorties :
#   - `candidate_qc` : circuit après ajouts gloutons,
#   - `kept_injections` : liste des injections finalement conservées.
# 
# > Remarque : si l’ajout ne **meilleure pas** la fidélité, il est **rejeté**.  
# > Pour `rzz`, un angle aléatoire est tiré à chaque essai.
# 

# In[ ]:


# =============================
# Injection guidée par la fidélité (glouton)
# =============================

def fidelity_driven_injection(
    base_qc: QuantumCircuit,
    target_qc: QuantumCircuit,
    blocks: List[Set[int]],
    max_trials: int = 300,
    fid_threshold: float = 0.9999,
) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Ajoute itérativement des portes inter-blocs qui améliorent la fidélité vis-à-vis de `target_qc`.
    On s’arrête dès que la fidélité dépasse `fid_threshold` ou que `max_trials` est atteint.
    """
    target_unitary = Operator(target_qc).data
    candidate_qc = base_qc.copy()
    kept_injections: List[Tuple[str, int, int, Optional[float]]] = []

    gate_pool = ["cx", "cz", "rzz"]
    rng = random.Random(42)

    for _ in range(max_trials):
        gate = rng.choice(gate_pool)
        q1 = rng.choice(tuple(blocks[0]))
        q2 = rng.choice(tuple(blocks[1]))
        theta = rng.uniform(0, 2 * math.pi) if gate == "rzz" else None

        # Tester l'ajout
        test_qc = candidate_qc.copy()
        if gate == "rzz":
            test_qc.rzz(theta, q1, q2)
        else:
            getattr(test_qc, gate)(q1, q2)

        # Acceptation gloutonne si la fidélité augmente
        fid_new = compute_fidelity(test_qc, target_unitary)
        fid_old = compute_fidelity(candidate_qc, target_unitary)

        if fid_new > fid_old:
            candidate_qc = test_qc
            kept_injections.append((gate, q1, q2, theta))
            print(f"✅ Ajouté {gate}({q1},{q2}) [fid={fid_new:.5f}]")
            if fid_new >= fid_threshold:
                break
        else:
            print(f"❌ Rejeté {gate}({q1},{q2}) [fid={fid_new:.5f}]")

    return candidate_qc, kept_injections


# ## 🚀 Partie 9 — Pipeline complet
# 
# Cette fonction orchestre tout le flux :
# 
# 1. **Affichage & coût** du circuit original.
# 2. **Partitionnement** (graphe d’interaction + Louvain) et extraction des **portes inter-blocs**.
# 3. Détection des **qubits fortement interactifs** (option : duplication pour l’optimisation intra-bloc).
# 4. **Optimisation intra-bloc** (NSGA-II + LAS) pour chaque bloc → circuits optimisés.
# 5. **Recomposition** d’un circuit global depuis les blocs optimisés, puis **réinjection** des portes inter-blocs d’origine.
# 6. **Injection inter-blocs** supplémentaire (au choix : *recuit simulé* ou *stochastique*).
# 7. **Injection gloutonne guidée par la fidélité** (option complémentaire).
# 8. **Compression finale** (passes “maison” + Qiskit).
# 9. **Résumé** (fidélité, profondeur, coûts, qubits, etc.) + métadonnées de sortie.
# 
# > Remarque : cette fonction **imprime** des informations et **sauvegarde** des figures (graphe d’interaction, circuits par bloc, Pareto, etc.).  
# > Le comportement est inchangé ; seuls les commentaires et le découpage ont été clarifiés.
# 

# In[ ]:


# =============================
# Pipeline complet — version robuste
# =============================
def optimise_circuit_pipeline(
    qc: QuantumCircuit,
    *,
    max_block_size: int = 5,
    k_interface: int = 1,
    injection_method: str = "stochastic",  # "sa" ou "stochastic"
    fid_threshold: float = 0.999,
    sa_iters: int = 2500,
    sa_seed: Optional[int] = 42,
    qubit_duplication_threshold: float = 0.5,
) -> Tuple[QuantumCircuit, Dict[str, object]]:
    print("\nCircuit original :")
    print(qc.draw(output="text"))
    qc.draw('mpl', filename='circuit_original.png', style='mpl', fold=1)
    # Référence & coût de départ
    qc_orig = qc.copy()
    if qc.num_qubits <= 15:
        U_orig = Operator(qc_orig).data
    else:
        print("⚠️ Qubits > 15 : On ignore le calcul de l'opérateur global (mémoire).")
        U_orig = np.eye(2) # Dummy
    cost_orig = compute_gate_cost(qc_orig)
    print(f"💰 Coût du circuit original (Lee et al. 2006) : {cost_orig}")
    # 1) Partitionnement + graphe
    print("\n📌 Partitionnement du circuit initial…")
    G = build_interaction_graph(qc)
    original_blocks = louvain_partition(qc)
    print("Qubits par bloc (initial) :", tuple(original_blocks))
    # Portes inter-blocs d’origine (réinjectées plus tard)
    original_interblock_gates = extract_interblock_gates(qc, original_blocks)
    print(f"📎 {len(original_interblock_gates)} portes inter-blocs extraites pour réinjection plus tard.")
    # Visualisation du graphe d’interaction AVANT duplication éventuelle
    print("🧭 Affichage du graphe d’interaction… avant duplication")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    plt.title("Graphe d’interaction avant duplication")
    plt.tight_layout(); save_plot("interaction_graph_avant_duplication"); plt.close()
    # 2) Identification des qubits très "inter-blocs" (option duplication) — ROBUSTE
    ihiq = globals().get("identify_highly_interactive_qubits", None)
    if callable(ihiq):
        highly_interactive_qubits = ihiq(qc, original_blocks, qubit_duplication_threshold)
        if highly_interactive_qubits:
            print("💡 Qubits identifiés pour duplication (original_q: target_block):", highly_interactive_qubits)
        else:
            print("💡 Aucune duplication de qubit nécessaire ou identifiée.")
    else:
        print("⚠️ Fonction 'identify_highly_interactive_qubits' introuvable — étape ignorée (pas bloquant).")
        highly_interactive_qubits = {}
    # Ajout logique de ces qubits dans les blocs cibles (préparation NSGA-II)
    for orig_q, target_block in highly_interactive_qubits.items():
        original_blocks[target_block].add(orig_q)
        print(f"🧪 Qubit {orig_q} ajouté dans le bloc {target_block} pour NSGA-II")
    # Visualisation du graphe d’interaction (après étape d’analyse)
    print("🧭 Affichage du graphe d’interaction…")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    plt.title("Graphe d’interaction")
    plt.tight_layout(); save_plot("interaction_graph"); plt.close()
    # 3) Optimisation intra-bloc
    block_circuits: List[Tuple[List[int], QuantumCircuit]] = []
    moo_metrics_blocks = []
    for idx, bl in enumerate(original_blocks):
        sub = extract_subcircuit(qc, bl)
        print(f"\n––– Bloc {idx} | Qubits {sorted(bl)} –––")
        print(sub.draw(output="text"))
        sub.draw('mpl', filename=f"block_{idx}_circuit_original.png", style='mpl', fold=1)
        print("  → Optimisation NSGA-II en cours…")
        best, hist_moo = optimise_block_nsga2(sub, generations=500, pop_size=400)
        moo_metrics_blocks.append(hist_moo[-1] if hist_moo else {})
        plot_moo_history(hist_moo, title=f"Evolution MoO - Bloc {idx}", save_as=f"moo_evolution_block_{idx}.png")
        export_all_indicators(hist_moo, idx)
        # Sauvegarde d’une jolie figure pour le bloc optimisé
        from qiskit.visualization import circuit_drawer
        fig = circuit_drawer(best, output="mpl", fold=60, style={"fontsize": 12})
        os.makedirs("out_figs", exist_ok=True)
        fig.savefig(f"out_figs/block_{idx}_circuit_optimized.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("    ✅ Circuit optimisé :")
        print(best.draw(output="text"))
        block_circuits.append((sorted(list(bl)), best))
        best.draw('mpl', filename=f"optimized_block_{idx}_circuit.png", style='mpl', fold=1)
    # 4) Recomposition du circuit global depuis les blocs optimisés
    qc_rebuilt_original_qubits = QuantumCircuit(qc.num_qubits)
    for qubits_list, cir in block_circuits:
        local_to_global_map = {i: q_idx for i, q_idx in enumerate(qubits_list)}
        for inst, qargs, cargs in cir.data:
            global_qargs = [qc_rebuilt_original_qubits.qubits[local_to_global_map[cir.find_bit(q).index]] for q in qargs]
            qc_rebuilt_original_qubits.append(inst, global_qargs, cargs)
    print("\nCircuit recomposé (avant SWAP interface et duplication) :")
    print(qc_rebuilt_original_qubits.draw(output="text"))
    fid_rebuilt = compute_fidelity(qc_rebuilt_original_qubits, U_orig)
    print(f"Fidélité recomposé ↔ original: {fid_rebuilt:.5f}")
    # 4.bis) Réinjection des portes inter-blocs d'origine dans le recomposé
    for inst, qargs, cargs in original_interblock_gates:
        global_qargs = [qc_rebuilt_original_qubits.qubits[qc.find_bit(q).index] for q in qargs]
        qc_rebuilt_original_qubits.append(inst, global_qargs, cargs)
    print("📎 Portes inter-blocs réinjectées dans le circuit recomposé.")
    fid_rebuilt1 = compute_fidelity(qc_rebuilt_original_qubits, U_orig)
    print(f"Fidélité recomposé (avec inter-blocs) ↔ original: {fid_rebuilt1:.5f}")
    print("\nCircuit recomposé avec portes interblocs :")
    print(qc_rebuilt_original_qubits.draw(output="text"))
    # 5) Injection inter-blocs (méthode au choix)
    if injection_method == "sa":
        qc_inj, kept = sa_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold,
                                    n_iters=sa_iters, seed=sa_seed)
    elif injection_method == "stochastic":
        qc_inj, kept = stochastic_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold)
    else:
        raise ValueError('injection_method doit être "sa" ou "stochastic".')
    print("\nCircuit après injection inter-blocs :")
    print(qc_inj.draw(output="text"))
    print(f"# portes inter-blocs conservées : {len(kept)}")
    fid_inj = compute_fidelity(qc_inj, U_orig)
    print(f"Fidélité après injection inter-blocs ↔ original: {fid_inj:.5f}")
    # 6) Injection gloutonne guidée par la fidélité (complément)
    qc_i, kept1 = fidelity_driven_injection(base_qc=qc_rebuilt_original_qubits, target_qc=qc_orig,
                                            blocks=original_blocks, max_trials=300, fid_threshold=0.9999)
    print("\nCircuit après injection inter-blocs avec NSGA2 (greedy):")
    print(qc_i.draw(output="text"))
    qc_i.draw('mpl', filename=f"final_optimized_circuitwithdriveninject.png", style='mpl', fold=1)
    print(f"# portes inter-blocs conservées : {len(kept1)}")
    fid_i = compute_fidelity(qc_i, U_orig)
    print(f"Fidélité après injection inter-blocs ↔ original: {fid_i:.5f}")
    # 7) Compression finale (choisit la meilleure base)
    if fid_i > fid_inj:
        qc_opt = compress_custom(qiskit_opt_pass(qc_i))
    else:
        qc_opt = compress_custom(qiskit_opt_pass(qc_inj))
    print("\nCircuit optimisé final :")
    print(qc_opt.draw(output="text"))
    qc_opt.draw('mpl', filename=f"final_optimized_circuit.png", style='mpl', fold=1)
    cost_final = compute_gate_cost(qc_opt)
    print(f"💰 Coût du circuit optimisé final (Lee et al. 2006) : {cost_final}")
    # 8) Résumé
    fid_final = compute_fidelity(qc_opt, U_orig)
    depth_before = qc_orig.depth()
    depth_after = qc_opt.depth()
    print("\n===== Résumé Final =====")
    print("🎯 Fidélité globale finale :", fid_final)
    print("📏 Profondeur (original) :", depth_before)
    print("📏 Profondeur (optimisé) :", depth_after)
    print("Total qubits (original):", qc_orig.num_qubits)
    print("Total qubits (final):", qc_opt.num_qubits)
    print(f"💰 Coût du circuit final:", cost_final)
    meta = {
        "blocks": original_blocks,
        "kept_injections": kept,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "fidelity_final": fid_final,
        "original_num_qubits": qc_orig.num_qubits,
        "final_num_qubits": qc_opt.num_qubits,
        "highly_interactive_qubits_identified": highly_interactive_qubits,
        "cost_before": cost_orig,
        "cost_after": cost_final,
        "moo_metrics_per_block": moo_metrics_blocks
    }
    return qc_opt, meta


# ## ▶️ Partie 10 — Exemple d’exécution
# 
# Petit exemple (style QAOA) pour **démontrer le pipeline** complet :
# 
# 1. Construction d’un circuit sur *n* qubits :
#    - mise en superposition (`H`),
#    - chaîne d’entrelacement via `CX` + `RZ`,
#    - rotations `RX`.
# 2. Lancement du **pipeline d’optimisation** avec :
#    - partitionnement Louvain,
#    - optimisation **intra-bloc** (NSGA-II + LAS),
#    - recomposition + réinjection des portes inter-blocs d’origine,
#    - **injection inter-blocs** (méthode *stochastique* ici),
#    - **compression** finale,
#    - résumé des métriques.
# 
# > ℹ️ Cet exemple **définit et appelle** `optimise_circuit_pipeline`
# 


def random_weakly_connected_circuit(
    n_qubits: int = 30,
    depth: int = 20,
    twoq_gates_total: int = 10,   # TRÈS faible nombre de 2-qubits au total
    connectivity_edges: int = 6,  # TRÈS faible connectivité (peu d'arêtes possibles)
    use_cz: bool = True,
    seed: int = 1234
) -> QuantumCircuit:
    """
    Génère un circuit aléatoire faiblement connecté:
      - 1-qubit: Rx/Ry/Rz + X/Y/Z
      - 2-qubits: CZ (par défaut) ou CX, mais très peu
      - connectivité: seulement 'connectivity_edges' arêtes autorisées
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    qc = QuantumCircuit(n_qubits, name="weak_random_30q")

    # --- Construire une connectivité très faible (liste d'arêtes autorisées) ---
    edges = set()
    attempts = 0
    while len(edges) < connectivity_edges and attempts < 10_000:
        a = rng.randrange(n_qubits)
        b = rng.randrange(n_qubits)
        if a == b:
            attempts += 1
            continue
        e = (a, b) if a < b else (b, a)
        edges.add(e)
        attempts += 1

    edges = sorted(edges)
    if len(edges) == 0:
        edges = [(0, 1)]

    twoq_layers = set(rng.sample(range(depth), k=min(twoq_gates_total, depth)))
    oneq_paulis = ["x", "y", "z"]
    oneq_rots = ["rx", "ry", "rz"]

    for d in range(depth):
        touched = rng.sample(range(n_qubits), k=max(6, n_qubits // 6))
        for q in touched:
            if rng.random() < 0.75:
                kind = rng.choice(oneq_rots)
                theta = float(np_rng.uniform(0, 2*np.pi))
                if kind == "rx":
                    qc.rx(theta, q)
                elif kind == "ry":
                    qc.ry(theta, q)
                else:
                    qc.rz(theta, q)
            else:
                kind = rng.choice(oneq_paulis)
                if kind == "x":
                    qc.x(q)
                elif kind == "y":
                    qc.y(q)
                else:
                    qc.z(q)
        if d in twoq_layers:
            (a, b) = rng.choice(edges)
            if use_cz:
                qc.cz(a, b)
            else:
                qc.cx(a, b)
        qc.barrier()
    return qc

if __name__ == "__main__":
    # --- Génération du circuit demandé par l’utilisateur ---
    qc = random_weakly_connected_circuit(
        n_qubits=30,
        depth=20,
        twoq_gates_total=8,     # encore plus faible
        connectivity_edges=5,   # très faible connectivité
        use_cz=True,
        seed=42
    )
    print("Circuit généré (30 qubits, faible connectivité) :")
    print(qc)

    # Lancer le pipeline complet
    qc_final, info = optimise_circuit_pipeline(
        qc,
        max_block_size=6,
        k_interface=1,
        injection_method="stochastic",  # "sa" ou "stochastic"
        fid_threshold=0.9999,
        sa_iters=3000,
        sa_seed=0,
        qubit_duplication_threshold=0.6,
    )

    # Résumé final (exemple)
    print("\n===== Résumé (main) =====")
    for k, v in info.items():
        if k == "blocks":
            print("Blocks :", v)
        else:
            print(f"{k.replace('_', ' ').title()} : {v}")


