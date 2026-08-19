#!/usr/bin/env python
# coding: utf-8

# # 🧪 Quantum Circuit Optimization — Modular Pipeline
#
# Objective: **reduce the cost/depth** while maintaining **high fidelity** to the reference circuit

from __future__ import annotations

# --- Python standard ---
import copy
import math

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
import os

# --- Scientific analysis ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D visualizations

# --- Graphs & clustering ---
import networkx as nx
import community  # Louvain algorithm (python-louvain)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Quantique (Qiskit) ---
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutationAnalysis, CommutativeCancellation, Optimize1qGates
)

# --- Evolutionary (DEAP) ---
from deap import base, creator, tools

# --- Parallelization ---
from joblib import Parallel, delayed


# ## 🎨 Part 2 — Visualization utility functions
#
# This section groups the plotting and saving functions used to analyze:
# - the convergence of fidelity across generations,
# - the Pareto front of solutions,
# - and the 3D clustering of optimized individuals.
#
# They make it easier to interpret the results produced by the evolutionary algorithm.
#

# In[ ]:


# =============================
# Utils: Plotting and saving functions
# =============================

def save_plot(name: str, out_dir="out_figs"):
    """
    Saves a matplotlib figure in PNG format.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=300)


def plot_convergence(hist_eps, save_as: Optional[str] = None):
    """
    Plots the convergence of the error (1 - fidelity).
    Y axis on a logarithmic scale.
    """
    plt.figure()
    plt.plot(range(len(hist_eps)), hist_eps, marker="o")
    plt.yscale("log")
    plt.xlabel("Generation")
    plt.ylabel(r"$1\!-\!F$ (log)")
    plt.title("Fidelity convergence")
    plt.grid(True)
    plt.tight_layout()

    if save_as:
        save_plot(save_as)
    plt.close()


def plot_pareto(front, save_as: Optional[str] = None):
    """
    Displays the Pareto front: cost vs depth, colored by error (1 - F).
    """
    costs = [i.fitness.values[2] for i in front]
    depths = [i.fitness.values[1] for i in front]
    eps = [1 - i.fitness.values[0] for i in front]

    plt.figure()
    sc = plt.scatter(costs, depths, c=eps, cmap="viridis")
    plt.colorbar(sc, label=r"$\varepsilon$ (1-F)")
    plt.xlabel("Chrom. cost")
    plt.ylabel("Depth")
    plt.title("Local Pareto front")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_as:
        save_plot(save_as)
    plt.close()


def plot_3d_clusters(pareto, n_clusters: int = 4, save_as: Optional[str] = None):
    """
    3D scatter (Depth, Cost, Error) with K-Means clustering.

    Axes:
      - X = depth
      - Y = chromosome cost (length or estimated cost)
      - Z = error ε = 1 - fidelity
    """
    if not pareto:
        return

    # Data extraction
    data = np.array([
        [ind.fitness.values[1], ind.fitness.values[2], 1.0 - ind.fitness.values[0]]
        for ind in pareto
    ])

    # Normalization + K-means
    k = max(1, min(n_clusters, len(data)))
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    # 3D display
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Cost')
    ax.set_zlabel('Error ε = 1 - F')
    ax.set_title('3D Clustering (K-Means) — Pareto population')

    if save_as:
        save_plot(save_as)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ## 🔗 Part 3 — Partitioning & Interaction graph
#
# This section groups the functions that make it possible to:
# - **identify inter-block gates** in a circuit,
# - **build an interaction graph** between qubits,
# - **apply different partitioning algorithms** (Louvain, Metis, Kernighan-Lin, recursive),
# - **refine and analyze partitions** (inter-block cost, highly interactive qubits),
# - prepare the circuit for modular optimizations.
#

# In[ ]:


# =============================
# Partition & Interaction graph
# =============================

def extract_interblock_gates(qc: QuantumCircuit, blocks: List[Set[int]]) -> List[Tuple]:
    """
    Extracts the gates that connect two distinct blocks.
    """
    bmap = {q: i for i, bl in enumerate(blocks) for q in bl}
    interblock_gates = []
    for ci in qc.data:
        qargs = ci.qubits
        if len(qargs) < 2:
            continue
        qubit_indices = {qc.find_bit(q).index for q in qargs}
        involved_blocks = {bmap.get(q) for q in qubit_indices}
        if len(involved_blocks) > 1:
            interblock_gates.append((ci.operation, qargs, ci.clbits))
    return interblock_gates


def louvain_partition(qc: QuantumCircuit) -> List[Set[int]]:
    """
    Partitions the qubits via the Louvain algorithm
    on a weighted interaction graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))

    for ci in qc.data:
        qargs = ci.qubits
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
    Builds a graph where the vertices = qubits,
    and the edges = number of 2-qubit gates between them.
    """
    G = nx.Graph()
    G.add_nodes_from(range(qc.num_qubits))
    for ci in qc.data:
        if ci.operation.name in {"cx", "cz", "rzz"} and len(ci.qubits) == 2:
            i, j = [qc.find_bit(q).index for q in ci.qubits]
            w = G.get_edge_data(i, j, default={"weight": 0})["weight"] + 1
            G.add_edge(i, j, weight=w)
    return G


# --- Recursive partitioning with Metis or Kernighan-Lin ---
def _partition_metis(graph: nx.Graph) -> List[Set[int]]:
    import nxmetis  # requires the `nxmetis` package
    _, parts = nxmetis.partition(graph, 2)
    return [set(p) for p in parts]


def _partition_kl(graph: nx.Graph) -> List[Set[int]]:
    from networkx.algorithms.community import kernighan_lin_bisection
    a, b = kernighan_lin_bisection(graph)
    return [set(a), set(b)]


def multilevel_partition(graph: nx.Graph, max_block_size: int) -> List[Set[int]]:
    """
    Applies recursive partitioning until each
    block is smaller than `max_block_size`.
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
    Computes the cost (number of gates) connecting two given blocks.
    """
    cost = 0
    for ci in qc.data:
        qargs = ci.qubits
        if len(qargs) < 2:
            continue
        qs = {qc.find_bit(q).index for q in qargs}
        if qs & blk0 and qs & blk1:
            cost += 1
    return cost


def refine_partition_kl(qc: QuantumCircuit, blocks: List[Set[int]], *, max_iter: int = 10) -> List[Set[int]]:
    """
    Refines a partition by moving qubits between blocks
    to reduce the inter-block cost (Kernighan-Lin method).
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

        # Try moving a → b
        for q in list(a):
            gain = best_cost - _interblock_gate_cost(qc, a - {q}, b | {q})
            if gain > gain_best:
                gain_best, q_best, side = gain, q, "a2b"

        # Try moving b → a
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


# ## 🧩 Part 4 — Sub-circuits & Recomposition
#
# In this section, we isolate the **sub-circuits per qubit block** and then **recompose** a global circuit from
# the optimized versions of each block.
#
# - `extract_subcircuit(qc, qubits)`: extracts the gates that are **strictly internal** to the `qubits` set, **locally
#   reindexing** the qubits (0..|qubits|-1) to make the sub-circuit self-contained.
# - `recompose_from_blocks(qc_original, block_subcircuits)`: reinserts, **in the order of the original circuit**, the gates
#   coming from the optimized sub-circuits when the operation entirely concerns a block; otherwise, it **keeps** the
#   original gate as-is (useful for inter-block gates).
#
# > Tip: after block-by-block optimization, use `recompose_from_blocks` to obtain a coherent global circuit again,
# > while benefiting from the local improvements.
#

# In[ ]:


# =============================
# Sub-circuits & Recomposition
# =============================

def extract_subcircuit(qc: QuantumCircuit, qubits: Set[int]) -> QuantumCircuit:
    """
    Extract the sub-circuit containing only the gates whose targets ALL
    belong to `qubits`. The qubits are locally remapped from 0 to |qubits|-1.

    Args:
        qc: Original global circuit.
        qubits: Set of qubit indices for the block.

    Returns:
        Self-contained local QuantumCircuit on |qubits| qubits.
    """
    # Create a local circuit with as many qubits as in the block
    sub = QuantumCircuit(len(qubits))

    # Global->local remapping: the i-th sorted qubit of the block becomes local index i
    local_index = {q: i for i, q in enumerate(sorted(list(qubits)))}

    # We only keep the gates that act EXCLUSIVELY on qubits of the block
    for ci in qc.data:
        qargs = ci.qubits
        if all(qc.find_bit(q).index in qubits for q in qargs):
            remapped_qargs = [sub.qubits[local_index[qc.find_bit(q).index]] for q in qargs]
            sub.append(ci.operation, remapped_qargs, ci.clbits)

    return sub


def recompose_from_blocks(
    qc_original: QuantumCircuit,
    block_subcircuits: List[Tuple[Set[int], QuantumCircuit]]
) -> QuantumCircuit:
    """
    Recompose a global circuit from (optimized) sub-circuits per block,
    respecting the order of operations of the original circuit.

    Principle:
    - We iterate over the gates of `qc_original` in their order.
    - If a gate belongs entirely to a block B, we insert instead the next
      corresponding gate from B's (already optimized) sub-circuit, remapped to global indices.
    - Otherwise (inter-block gate), we keep the original gate as-is.

    Args:
        qc_original: Reference circuit whose order of operations is respected.
        block_subcircuits: List of tuples (block_qubits, optimized_subcircuit).

    Returns:
        Recomposed global QuantumCircuit.
    """
    # Prepare the remapping structures and read cursors
    block_maps = []
    for block_qubits, sub in block_subcircuits:
        sorted_block = sorted(block_qubits)
        global_to_local = {q: i for i, q in enumerate(sorted_block)}  # useful if needed
        block_maps.append((set(sorted_block), sub, global_to_local))

    # New global circuit (same number of qubits as the original)
    qc_recomposed = QuantumCircuit(qc_original.num_qubits)

    # Cursor to track where we are in EACH sub-circuit
    subcircuit_cursors = [0 for _ in block_subcircuits]

    # Iterate over the gates of the original circuit, in order
    for ci in qc_original.data:
        qargs = ci.qubits
        # Global indices touched by the current gate
        q_indices = [qc_original.find_bit(q).index for q in qargs]
        inserted = False

        # Check whether the gate belongs to a specific block
        for idx, (block_qubits, sub, g2l_unused) in enumerate(block_maps):
            if all(q in block_qubits for q in q_indices):
                # Consume the next gate from the block's sub-circuit
                if subcircuit_cursors[idx] >= len(sub.data):
                    raise ValueError(f"Too many gates requested for block {idx} relative to its sub-circuit.")

                ci_opt = sub.data[subcircuit_cursors[idx]]
                inst_opt, qargs_opt, cargs_opt = ci_opt.operation, ci_opt.qubits, ci_opt.clbits
                subcircuit_cursors[idx] += 1

                # Remap the sub-circuit's local qubits to their original GLOBAL indices
                sorted_block = sorted(block_qubits)
                mapped_qargs = [qc_recomposed.qubits[sorted_block[sub.find_bit(q).index]] for q in qargs_opt]

                qc_recomposed.append(inst_opt, mapped_qargs, cargs_opt)
                inserted = True
                break

        # If the gate does not belong to a single block (inter-block gate), keep the original gate
        if not inserted:
            mapped_qargs = [qc_recomposed.qubits[i] for i in q_indices]
            qc_recomposed.append(ci.operation, mapped_qargs, ci.clbits)

    return qc_recomposed


# ## 🎯 Part 5 — Fidelity & Compression
#
# This section groups:
# - **`compute_fidelity(circ, target)`**: computes the operator-operator fidelity
#   \( F = \frac{|\mathrm{Tr}(U_{\text{circ}}\,U_{\text{target}}^\dagger)|}{2^n} \), aligning the size if needed.
# - **"Home-made" compression passes**:
#   - `cancel_inverse_gates`: cancels adjacent pairs of inverse gates (e.g. `x` followed by `x`, `cx` followed by `cx`, etc.).
#   - `merge_rotations`: merges successive rotations of the same axis on the **same qubit**.
#   - `remove_negligible_rotations`: removes rotations of very low amplitude (threshold `th`).
#   - `compress_custom`: minimal pipeline combining the three steps above.
# - **Qiskit pass**:
#   - `qiskit_opt_pass` applies `Optimize1qGates` and `CommutativeCancellation` to simplify further.
#

# In[ ]:


# =============================
# Fidelity & Compression
# =============================

from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

def compute_fidelity(circ: QuantumCircuit, target: np.ndarray) -> float:
    """
    Computes the operator-operator fidelity between the circuit 'circ' and the operator 'target'.
    If the circuit has MORE qubits than the target, the target is 'padded' with the identity.
    If the circuit has FEWER qubits, an error is raised (case not handled here).
    If the number of qubits is too high (> 15), returns 1.0 by default to avoid a memory explosion.
    """
    if circ.num_qubits > 15:
        # print("⚠️ Qubits > 15: Fidelity computation skipped (memory).")
        return 1.0

    try:
        circ_op = Operator(circ).data
        target_nqubits = int(np.log2(target.shape[0]))

        if circ.num_qubits > target_nqubits:
            # Extend the target to the circuit's dimension by placing it in the top-left corner
            target_op_padded = np.eye(2**circ.num_qubits, dtype=complex)
            target_op_padded[:target.shape[0], :target.shape[1]] = target
            target = target_op_padded
        elif circ.num_qubits < target_nqubits:
            raise ValueError("The circuit has fewer qubits than the target operator — fidelity not directly defined.")

        # F = |Tr(Uc * Ut^\dagger)| / 2^n
        return abs(np.trace(circ_op @ target.conj().T)) / (2 ** circ.num_qubits)
    except Exception:
        return 1.0


def cancel_inverse_gates(c: QuantumCircuit) -> QuantumCircuit:
    """
    Removes adjacent pairs of self-inverse gates applied on EXACTLY the same qubits.
    Handled here: {x, y, z, h, cx}. (add more if desired)
    """
    new = QuantumCircuit(c.num_qubits)
    skip = set()

    for i in range(len(c.data) - 1):
        if i in skip:
            continue
        g1, q1 = c.data[i].operation, c.data[i].qubits
        g2, q2 = c.data[i + 1].operation, c.data[i + 1].qubits

        if g1.name == g2.name and q1 == q2 and g1.name in {"x", "y", "z", "h", "cx"}:
            # g then g => identity
            skip.add(i + 1)
            continue
        new.append(g1, q1)

    # Last gate if not skipped
    if (len(c.data) - 1) not in skip and len(c.data) > 0:
        g, q = c.data[-1].operation, c.data[-1].qubits
        new.append(g, q)
    return new


def merge_rotations(c: QuantumCircuit) -> QuantumCircuit:
    """
    Merges successive rotations of the same type (rx/ry/rz) on the same qubit:
    rx(a) ; rx(b)  ->  rx(a+b)
    """
    new = QuantumCircuit(c.num_qubits)
    i = 0
    while i < len(c.data):
        g, q = c.data[i].operation, c.data[i].qubits
        if g.name in {"rx", "ry", "rz"}:
            angle = g.params[0]
            j = i + 1
            # Accumulate as long as the type AND target are identical
            while j < len(c.data):
                g2, q2 = c.data[j].operation, c.data[j].qubits
                if g2.name == g.name and q2 == q:
                    angle += g2.params[0]
                    j += 1
                else:
                    break
            # Emit a single rotation with the merged angle
            getattr(new, g.name)(angle, q[0])
            i = j
        else:
            new.append(g, q)
            i += 1
    return new


def remove_negligible_rotations(c: QuantumCircuit, *, th: float = 1e-4) -> QuantumCircuit:
    """
    Removes rx/ry/rz rotations of very low amplitude (|theta| < th).
    """
    new = QuantumCircuit(c.num_qubits)
    for ci in c.data:
        g, q = ci.operation, ci.qubits
        if g.name in {"rx", "ry", "rz"} and abs(float(g.params[0])) < th:
            # ignore this small rotation
            continue
        new.append(g, q)
    return new


def compress_custom(circ: QuantumCircuit) -> QuantumCircuit:
    """
    Minimal compression pipeline:
      1) Cancel inverses,
      2) Merge rotations,
      3) Remove small rotations.
    """
    return remove_negligible_rotations(
        merge_rotations(
            cancel_inverse_gates(circ)
        )
    )


def qiskit_opt_pass(c: QuantumCircuit) -> QuantumCircuit:
    """
    Standard Qiskit optimization pass:
      - Optimize1qGates (more compact rewriting of 1-qubit sequences)
      - CommutativeCancellation (cancellation of unnecessary commutative gates)
    """
    pm = PassManager([Optimize1qGates(), CommutativeCancellation()])
    return pm.run(c)


# ## 📊 Part 5.5 — Multi-objective Quality Indicators (HV, IGD, Spread, ε)
#
# This section implements the standard metrics for evaluating the quality of Pareto fronts:
# - **Hypervolume (HV)**: measures the space dominated by the front.
# - **Inverted Generational Distance (IGD)**: distance to the reference front.
# - **Spread (Δ)**: diversity of the solutions.
# - **Epsilon Indicator (ε)**: domination factor.
# - **Spacing**: uniformity of the distribution.

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


# ## 🧬 Part 6 — Intra-block optimization: NSGA-II + Local Angle Search (LAS)
#
# In this section, for **each qubit block** we look for an "equivalent" circuit (same targeted operator) that is
# **more efficient** according to 3 objectives:
# 1. **Maximize** the **fidelity** with respect to the target,
# 2. **Minimize** the **depth** (after a light `transpile`),
# 3. **Minimize** a **cost** (chromosome length or weighted gate cost).
#
# Strategy:
# - We encode a **chromosome** as a list of genes `("gate", target, ctrl?, angle?)`.
# - We evaluate each individual via `compute_fidelity`, `transpile(...).depth()` and a cost.
# - We apply **NSGA-II** (DEAP) for multi-objective optimization.
# - We add a **Local Angle Search (LAS)**: for each rotation (rx/ry/rz/rzz), we estimate a **pseudo-gradient**
#   by finite differences and try a few `η` steps to **locally improve** the fidelity.
#
# Outputs:
# - An **optimized circuit** per block,
# - **Figures**: convergence curve, **Pareto front**, **3D clustering** of the solutions.
#

# In[ ]:


# =============================
# Intra-block optimization: NSGA-II + LAS
# =============================

def compute_gate_cost(qc: QuantumCircuit) -> float:
    """
    Simple cost weighted by gate type. Adjust the table according to your hardware/targets.
    """
    cost_table = {
        "x": 1, "z": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1,
        "h": 2, "cx": 5, "cz": 5, "ccx": 13
    }
    return sum(cost_table.get(ci.operation.name.lower(), 1) for ci in qc.data)


def update_rotation_angles(
    chrom: List[Tuple[str, int, Optional[int], Optional[float]]],
    build_fn,
    target_unitary: np.ndarray,
    *,
    eta_range: Sequence[float] = (0.01, 0.1, 0.5),
    delta: float = 0.1,
) -> List[Tuple[str, int, Optional[int], Optional[float]]]:
    """
    Local angle search by central differences (pseudo-gradient).
    Compatible genes: ('rx'|'ry'|'rz'|'rzz', target, ctrl?, theta).

    - We modify an angle 'θ' and measure F(θ+δ), F(θ-δ) to approximate dF/dθ.
    - We test steps 'η' ∈ eta_range to accept an immediate improvement.
    """

    def wrap_angle(theta: float) -> float:
        twopi = 2.0 * math.pi
        # Wrap the angle into (-π, π]
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
                best = cand  # immediate acceptance
        base_fit = best_local_fit

    return best




# In[ ]:


def optimise_block_nsga2(qc_target: QuantumCircuit, *, generations=500, pop_size=300, n_jobs=-1, P_star=None):
    nq = qc_target.num_qubits; U_target = Operator(qc_target).data
    gate_pool = ["h", "x", "y", "z", "rx", "ry", "rz"]
    if nq >= 2:
        gate_pool += ["cx", "cz", "rzz"]

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

    fid_cache: Dict[Tuple, Tuple[float, int, int]] = {}

    def evaluate_population(individuals):
        # Dedupe identical genomes (common once the population converges and
        # low-probability mate/mutate skips reproduce an already-seen chromosome)
        # before dispatching to Parallel — cache lives in the parent process only,
        # so it stays correct under joblib's default process (loky) backend.
        to_run, seen_keys = [], set()
        for ind in individuals:
            key = tuple(ind)
            if key not in fid_cache and key not in seen_keys:
                to_run.append(ind)
                seen_keys.add(key)
        fresh = Parallel(n_jobs)(delayed(eval_ind)(i) for i in to_run)
        for ind, fit in zip(to_run, fresh):
            fid_cache[tuple(ind)] = fit
        return [fid_cache[tuple(ind)] for ind in individuals]

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
    fits = evaluate_population(pop)
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
        fits = evaluate_population(invalid)
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





# ## 🌉 Part 7 — Inter-block injection (SA / Stochastic)
#
# Objective: **add gates between blocks** (e.g. `cx`, `cz`, `rzz`) in order to improve the overall fidelity, while
# controlling the depth, the number of gates, and (optionally) the **crosstalk**.
#
# Two strategies:
# - **Simulated annealing (SA)**: we explore a **pool** of candidates and minimize an **energy**
#   `E = α·(#gates) + β·depth + γ·crosstalk + δ·fidelity_penalty`.
# - **Stochastic**: we try to randomly add inter-block gates and **keep** only those that
#   **preserve** a fidelity ≥ threshold.
#

# In[ ]:


# =============================
# Inter-block injection (SA / stochastic)
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
    Builds a candidate circuit by applying the active injections to the base circuit,
    then performs a light transpile to estimate the depth.
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
    Energy function for simulated annealing.
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
    Proposes a random move on the pool (toggle, type swap, shift, or fine-tuning of θ).
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
    Creates an initial pool of (disabled) injections between two blocks.
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
    Inter-block injection via simulated annealing (SA).
    Returns the final circuit and the list of retained injections.
    """
    if len(blocks) < 2:
        raise ValueError("sa_injection requires at least two blocks.")

    rng = random.Random(seed)
    injections = _sa_generate_pool(blocks, gate_types, rng=rng, n_candidates=n_candidates)
    target_U = Operator(base_qc).data

    # Initial temperature based on the variance of energy samples
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

    # Annealing loop
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
        raise RuntimeError(f"SA does not reach the target fidelity: {fid_final:.5f} < {fid_threshold}")

    kept = [(inj.gate, inj.q1, inj.q2, inj.theta) for inj in best if inj.enabled]
    return final_circ, kept


def stochastic_injection(qc: QuantumCircuit, blocks: List[Set[int]], *,
                         n_injections: int = 100,
                         fid_threshold: float = 0.999,
                         gate_probs: Optional[Dict[str, float]] = None,
                         seed: Optional[int] = None) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Stochastic inter-block injection: we add gates at random and only keep
    those that do not degrade the fidelity below the threshold.
    """
    if len(blocks) < 2:
        raise ValueError("stochastic_injection requires at least two blocks.")

    gate_probs = gate_probs or {"cx": 1.0, "cz": 1.0, "rzz": 1.0}
    total = sum(gate_probs.values())
    gate_types, probs = zip(*([(g, p / total) for g, p in gate_probs.items()]))

    rng = random.Random(seed)
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

        # Option: small optimization/normalization pass
        cand = qiskit_opt_pass(compress_custom(cand))

        fid = compute_fidelity(cand, U_ref)
        if fid >= fid_threshold:
            qc = cand
            kept.append((gate, qi, qj, theta))
            U_ref = Operator(qc).data  # update the reference

    return qc, kept


# ## 📈 Part 8 — Fidelity-driven injection (greedy)
#
# **Greedy** strategy: we try to add an inter-block gate (`cx`, `cz`, `rzz`) and **keep** the addition **only** if
# the **fidelity** with respect to the **target** circuit increases. We repeat until a fidelity **threshold** is
# reached or a **trial budget** is exhausted.
#
# - Inputs:
#   - `base_qc`: starting circuit (without injections or partially injected),
#   - `target_qc`: target circuit whose unitary we want to approach,
#   - `blocks`: two (or more) qubit blocks (we draw 1 qubit from each of the first two blocks),
#   - `max_trials`: maximum number of trials,
#   - `fid_threshold`: desired fidelity threshold.
# - Outputs:
#   - `candidate_qc`: circuit after greedy additions,
#   - `kept_injections`: list of injections ultimately retained.
#
# > Note: if the addition does not **improve** the fidelity, it is **rejected**.
# > For `rzz`, a random angle is drawn at each trial.
#

# In[ ]:


# =============================
# Fidelity-driven injection (greedy)
# =============================

def fidelity_driven_injection(
    base_qc: QuantumCircuit,
    target_qc: QuantumCircuit,
    blocks: List[Set[int]],
    max_trials: int = 300,
    fid_threshold: float = 0.9999,
) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Iteratively adds inter-block gates that improve the fidelity with respect to `target_qc`.
    Stops as soon as the fidelity exceeds `fid_threshold` or `max_trials` is reached.
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

        # Test the addition
        test_qc = candidate_qc.copy()
        if gate == "rzz":
            test_qc.rzz(theta, q1, q2)
        else:
            getattr(test_qc, gate)(q1, q2)

        # Greedy acceptance if the fidelity increases
        fid_new = compute_fidelity(test_qc, target_unitary)
        fid_old = compute_fidelity(candidate_qc, target_unitary)

        if fid_new > fid_old:
            candidate_qc = test_qc
            kept_injections.append((gate, q1, q2, theta))
            print(f"✅ Added {gate}({q1},{q2}) [fid={fid_new:.5f}]")
            if fid_new >= fid_threshold:
                break
        else:
            print(f"❌ Rejected {gate}({q1},{q2}) [fid={fid_new:.5f}]")

    return candidate_qc, kept_injections


# ## 🚀 Part 9 — Complete pipeline
#
# This function orchestrates the whole flow:
#
# 1. **Display & cost** of the original circuit.
# 2. **Partitioning** (interaction graph + Louvain) and extraction of **inter-block gates**.
# 3. Detection of **highly interactive qubits** (option: duplication for intra-block optimization).
# 4. **Intra-block optimization** (NSGA-II + LAS) for each block -> optimized circuits.
# 5. **Recomposition** of a global circuit from the optimized blocks, then **reinjection** of the original inter-block gates.
# 6. Additional **inter-block injection** (choice of: *simulated annealing* or *stochastic*).
# 7. **Fidelity-driven greedy injection** (complementary option).
# 8. **Final compression** ("home-made" passes + Qiskit).
# 9. **Summary** (fidelity, depth, costs, qubits, etc.) + output metadata.
#
# > Note: this function **prints** information and **saves** figures (interaction graph, per-block circuits, Pareto, etc.).
# > The behavior is unchanged; only the comments and the breakdown have been clarified.
#

# In[ ]:


# =============================
# Complete pipeline — robust version
# =============================
def optimise_circuit_pipeline(
    qc: QuantumCircuit,
    *,
    max_block_size: int = 5,
    k_interface: int = 1,
    injection_method: str = "stochastic",  # "sa" or "stochastic"
    fid_threshold: float = 0.999,
    sa_iters: int = 2500,
    sa_seed: Optional[int] = 42,
    qubit_duplication_threshold: float = 0.5,
    generations: int = 500,
    pop_size: int = 400,
) -> Tuple[QuantumCircuit, Dict[str, object]]:
    print("\nOriginal circuit:")
    print(qc.draw(output="text"))
    qc.draw('mpl', filename='circuit_original.png', style='mpl', fold=1)
    # Reference & starting cost
    qc_orig = qc.copy()
    if qc.num_qubits <= 15:
        U_orig = Operator(qc_orig).data
    else:
        print("⚠️ Qubits > 15: Skipping computation of the global operator (memory).")
        U_orig = np.eye(2) # Dummy
    cost_orig = compute_gate_cost(qc_orig)
    print(f"💰 Cost of the original circuit (Lee et al. 2006): {cost_orig}")
    # 1) Partitioning + graph
    t_partitioning_start = time.perf_counter()
    print("\n📌 Partitioning the initial circuit...")
    G = build_interaction_graph(qc)
    original_blocks = louvain_partition(qc)
    print("Qubits per block (initial):", tuple(original_blocks))
    # Original inter-block gates (reinjected later)
    original_interblock_gates = extract_interblock_gates(qc, original_blocks)
    print(f"📎 {len(original_interblock_gates)} inter-block gates extracted for later reinjection.")
    # Visualization of the interaction graph BEFORE any duplication
    print("🧭 Displaying the interaction graph... before duplication")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    plt.title("Interaction graph before duplication")
    plt.tight_layout(); save_plot("interaction_graph_avant_duplication"); plt.close()
    # 2) Identification of highly "inter-block" qubits (duplication option) — ROBUST
    ihiq = globals().get("identify_highly_interactive_qubits", None)
    if callable(ihiq):
        highly_interactive_qubits = ihiq(qc, original_blocks, qubit_duplication_threshold)
        if highly_interactive_qubits:
            print("💡 Qubits identified for duplication (original_q: target_block):", highly_interactive_qubits)
        else:
            print("💡 No qubit duplication necessary or identified.")
    else:
        print("⚠️ Function 'identify_highly_interactive_qubits' not found — step skipped (not blocking).")
        highly_interactive_qubits = {}
    # Logically add these qubits into the target blocks (NSGA-II preparation)
    for orig_q, target_block in highly_interactive_qubits.items():
        original_blocks[target_block].add(orig_q)
        print(f"🧪 Qubit {orig_q} added to block {target_block} for NSGA-II")
    # Visualization of the interaction graph (after the analysis step)
    print("🧭 Displaying the interaction graph...")
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='red')
    plt.title("Interaction graph")
    plt.tight_layout(); save_plot("interaction_graph"); plt.close()
    t_partitioning = time.perf_counter() - t_partitioning_start
    # 3) Intra-block optimization
    t_block_optimization_start = time.perf_counter()
    block_circuits: List[Tuple[List[int], QuantumCircuit]] = []
    moo_metrics_blocks = []
    for idx, bl in enumerate(original_blocks):
        sub = extract_subcircuit(qc, bl)
        print(f"\n––– Block {idx} | Qubits {sorted(bl)} –––")
        print(sub.draw(output="text"))
        sub.draw('mpl', filename=f"block_{idx}_circuit_original.png", style='mpl', fold=1)
        print("  → NSGA-II optimization in progress...")
        best, hist_moo = optimise_block_nsga2(sub, generations=generations, pop_size=pop_size)
        moo_metrics_blocks.append(hist_moo[-1] if hist_moo else {})
        plot_moo_history(hist_moo, title=f"Evolution MoO - Block {idx}", save_as=f"moo_evolution_block_{idx}.png")
        export_all_indicators(hist_moo, idx)
        # Save a nice figure for the optimized block
        from qiskit.visualization import circuit_drawer
        fig = circuit_drawer(best, output="mpl", fold=60, style={"fontsize": 12})
        os.makedirs("out_figs", exist_ok=True)
        fig.savefig(f"out_figs/block_{idx}_circuit_optimized.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("    ✅ Optimized circuit:")
        print(best.draw(output="text"))
        block_circuits.append((sorted(list(bl)), best))
        best.draw('mpl', filename=f"optimized_block_{idx}_circuit.png", style='mpl', fold=1)
    t_block_optimization = time.perf_counter() - t_block_optimization_start
    # 4) Recomposition of the global circuit from the optimized blocks
    qc_rebuilt_original_qubits = QuantumCircuit(qc.num_qubits)
    for qubits_list, cir in block_circuits:
        local_to_global_map = {i: q_idx for i, q_idx in enumerate(qubits_list)}
        for ci in cir.data:
            global_qargs = [qc_rebuilt_original_qubits.qubits[local_to_global_map[cir.find_bit(q).index]] for q in ci.qubits]
            qc_rebuilt_original_qubits.append(ci.operation, global_qargs, ci.clbits)
    print("\nRecomposed circuit (before interface SWAP and duplication):")
    print(qc_rebuilt_original_qubits.draw(output="text"))
    fid_rebuilt = compute_fidelity(qc_rebuilt_original_qubits, U_orig)
    print(f"Recomposed <-> original fidelity: {fid_rebuilt:.5f}")
    # 4.bis) Reinjection of the original inter-block gates into the recomposed circuit
    for inst, qargs, cargs in original_interblock_gates:
        global_qargs = [qc_rebuilt_original_qubits.qubits[qc.find_bit(q).index] for q in qargs]
        qc_rebuilt_original_qubits.append(inst, global_qargs, cargs)
    print("📎 Inter-block gates reinjected into the recomposed circuit.")
    fid_rebuilt1 = compute_fidelity(qc_rebuilt_original_qubits, U_orig)
    print(f"Recomposed (with inter-block gates) <-> original fidelity: {fid_rebuilt1:.5f}")
    print("\nRecomposed circuit with inter-block gates:")
    print(qc_rebuilt_original_qubits.draw(output="text"))
    # 5) Inter-block injection (method of choice)
    t_injection_start = time.perf_counter()
    if injection_method == "sa":
        qc_inj, kept = sa_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold,
                                    n_iters=sa_iters, seed=sa_seed)
    elif injection_method == "stochastic":
        qc_inj, kept = stochastic_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold,
                                            seed=sa_seed)
    else:
        raise ValueError('injection_method must be "sa" or "stochastic".')
    print("\nCircuit after inter-block injection:")
    print(qc_inj.draw(output="text"))
    print(f"# inter-block gates retained: {len(kept)}")
    fid_inj = compute_fidelity(qc_inj, U_orig)
    print(f"Fidelity after inter-block injection <-> original: {fid_inj:.5f}")
    # 6) Fidelity-driven greedy injection (complement)
    qc_i, kept1 = fidelity_driven_injection(base_qc=qc_rebuilt_original_qubits, target_qc=qc_orig,
                                            blocks=original_blocks, max_trials=300, fid_threshold=0.9999)
    print("\nCircuit after inter-block injection with NSGA2 (greedy):")
    print(qc_i.draw(output="text"))
    qc_i.draw('mpl', filename=f"final_optimized_circuitwithdriveninject.png", style='mpl', fold=1)
    print(f"# inter-block gates retained: {len(kept1)}")
    fid_i = compute_fidelity(qc_i, U_orig)
    print(f"Fidelity after inter-block injection <-> original: {fid_i:.5f}")
    t_injection = time.perf_counter() - t_injection_start
    # 7) Final compression (choose the best base)
    t_compression_start = time.perf_counter()
    if fid_i > fid_inj:
        qc_opt = compress_custom(qiskit_opt_pass(qc_i))
    else:
        qc_opt = compress_custom(qiskit_opt_pass(qc_inj))
    print("\nFinal optimized circuit:")
    print(qc_opt.draw(output="text"))
    qc_opt.draw('mpl', filename=f"final_optimized_circuit.png", style='mpl', fold=1)
    cost_final = compute_gate_cost(qc_opt)
    print(f"💰 Cost of the final optimized circuit (Lee et al. 2006): {cost_final}")
    t_compression = time.perf_counter() - t_compression_start
    # 8) Summary
    fid_final = compute_fidelity(qc_opt, U_orig)
    depth_before = qc_orig.depth()
    depth_after = qc_opt.depth()
    print("\n===== Final Summary =====")
    print("🎯 Final overall fidelity:", fid_final)
    print("📏 Depth (original):", depth_before)
    print("📏 Depth (optimized):", depth_after)
    print("Total qubits (original):", qc_orig.num_qubits)
    print("Total qubits (final):", qc_opt.num_qubits)
    print(f"💰 Cost of the final circuit:", cost_final)
    meta = {
        "blocks": original_blocks,
        "kept_injections": kept,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "fidelity_final": fid_final,
        "original_num_qubits": qc_orig.num_qubits,
        "final_num_qubits": qc_opt.num_qubits,
        "highly_interactive_qubits_identified": highly_interactive_qubits,
        "stage_timings_s": {
            "partitioning": t_partitioning,
            "block_optimization": t_block_optimization,
            "injection": t_injection,
            "compression": t_compression,
        },
        "cost_before": cost_orig,
        "cost_after": cost_final,
        "moo_metrics_per_block": moo_metrics_blocks
    }
    return qc_opt, meta


# ## ▶️ Part 10 — Execution example
#
# Small example (QAOA-style) to **demonstrate the complete pipeline**:
#
# 1. Building a circuit on *n* qubits:
#    - putting into superposition (`H`),
#    - entanglement chain via `CX` + `RZ`,
#    - `RX` rotations.
# 2. Launching the **optimization pipeline** with:
#    - Louvain partitioning,
#    - **intra-block** optimization (NSGA-II + LAS),
#    - recomposition + reinjection of the original inter-block gates,
#    - **inter-block injection** (*stochastic* method here),
#    - final **compression**,
#    - metrics summary.
#
# > ℹ️ This example **defines and calls** `optimise_circuit_pipeline`
#


def random_weakly_connected_circuit(
    n_qubits: int = 30,
    depth: int = 20,
    twoq_gates_total: int = 10,   # VERY low number of 2-qubit gates in total
    connectivity_edges: int = 6,  # VERY low connectivity (few possible edges)
    use_cz: bool = True,
    seed: int = 1234,
    **_ignored,
) -> QuantumCircuit:
    """
    Generates a weakly connected random circuit:
      - 1-qubit: Rx/Ry/Rz + X/Y/Z
      - 2-qubit: CZ (by default) or CX, but very few
      - connectivity: only 'connectivity_edges' allowed edges
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    qc = QuantumCircuit(n_qubits, name="weak_random_30q")

    # --- Build a very low connectivity (list of allowed edges) ---
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


def qaoa_maxcut_circuit(
    n_qubits: int = 12,
    p: int = 2,
    gammas: Optional[Sequence[float]] = None,
    betas: Optional[Sequence[float]] = None,
    seed: int = 0,
    **_ignored,
) -> QuantumCircuit:
    """
    QAOA circuit for MaxCut on a ring + opposite-chord 3-regular graph
    (generalizes the notebook's fixed 12-qubit qaoa_maxcut_12qubits to
    arbitrary n_qubits). For odd n_qubits the last unmatched chord is
    simply dropped (range(n_qubits // 2) chords for n_qubits ring edges).
    If gammas/betas aren't given, p values each are drawn from a seeded RNG
    (gamma in [0.3, 1.5], beta in [0.2, 0.8]) instead of the notebook's
    fixed constants, so multi-seed sweeps actually vary the circuit.
    """
    rng = random.Random(seed)
    ring_edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    chord_edges = [(i, (i + n_qubits // 2) % n_qubits) for i in range(n_qubits // 2)]
    edges = ring_edges + chord_edges

    if gammas is None:
        gammas = [rng.uniform(0.3, 1.5) for _ in range(p)]
    if betas is None:
        betas = [rng.uniform(0.2, 0.8) for _ in range(p)]

    qc = QuantumCircuit(n_qubits, name=f"QAOA_MaxCut_{n_qubits}q")
    for q in range(n_qubits):
        qc.h(q)
    for layer in range(p):
        for i, j in edges:
            qc.rzz(2 * gammas[layer], i, j)
        for q in range(n_qubits):
            qc.rx(2 * betas[layer], q)
    return qc


def w_state_circuit(n_qubits: int = 5, seed: int = 0, **_ignored) -> QuantumCircuit:
    """
    W-state preparation via a ry+cx ladder (generalizes the notebook's
    fixed 5-qubit w_state_5qubits: theta_k = 2*arccos(sqrt((n-k)/(n-k+1)))).
    Deterministic; seed accepted only for CIRCUIT_GENERATORS call-site
    signature consistency.
    """
    qc = QuantumCircuit(n_qubits, name=f"W_state_{n_qubits}q")
    qc.ry(2 * math.acos(math.sqrt((n_qubits - 1) / n_qubits)), 0)
    for k in range(1, n_qubits - 1):
        qc.cx(k - 1, k)
        qc.ry(2 * math.acos(math.sqrt((n_qubits - k - 1) / (n_qubits - k))), k)
    qc.cx(n_qubits - 2, n_qubits - 1)
    return qc


from qiskit.synthesis.qft import synth_qft_full


def qft_circuit(n_qubits: int = 4, seed: int = 0, **_ignored) -> QuantumCircuit:
    """
    Standard QFT via qiskit's own synth_qft_full, replacing the notebook's
    hand-written 4-qubit h/cp/swap ladder with the library's arbitrary-n
    equivalent of the same transform (synth_qft_full rather than the older
    QFT circuit-library class, which is deprecated as of Qiskit 2.1 and
    slated for removal in Qiskit 3.0). seed unused (deterministic).
    """
    qc = synth_qft_full(n_qubits)
    qc.name = f"QFT_{n_qubits}q"
    return qc


def hw_efficient_ansatz_circuit(
    n_qubits: int = 4, reps: int = 1, seed: int = 0, **_ignored
) -> QuantumCircuit:
    """
    Generic hardware-efficient-style circuit (H layer, then `reps` x
    [ry+rz layer, linear cx chain]) generalizing final_test_AG/vqe's fixed
    4-qubit circuit. Despite that source notebook's folder name, this is
    not an actual VQE (no cost Hamiltonian, no variational optimization
    loop) -- named for what it structurally is rather than that label.
    Angles are seeded-random rather than the notebook's fixed constants,
    so multi-seed sweeps vary the circuit.
    """
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits, name=f"HWEfficientAnsatz_{n_qubits}q")
    for q in range(n_qubits):
        qc.h(q)
    for _ in range(reps):
        for q in range(n_qubits):
            qc.ry(rng.uniform(0, 2 * math.pi), q)
            qc.rz(rng.uniform(0, 2 * math.pi), q)
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


if __name__ == "__main__":
    # --- Generation of the circuit requested by the user ---
    qc = random_weakly_connected_circuit(
        n_qubits=30,
        depth=20,
        twoq_gates_total=8,     # even lower
        connectivity_edges=5,   # very low connectivity
        use_cz=True,
        seed=42
    )
    print("Generated circuit (30 qubits, low connectivity):")
    print(qc)

    # Launch the complete pipeline
    qc_final, info = optimise_circuit_pipeline(
        qc,
        max_block_size=6,
        k_interface=1,
        injection_method="stochastic",  # "sa" or "stochastic"
        fid_threshold=0.9999,
        sa_iters=3000,
        sa_seed=0,
        qubit_duplication_threshold=0.6,
    )

    # Final summary (example)
    print("\n===== Summary (main) =====")
    for k, v in info.items():
        if k == "blocks":
            print("Blocks:", v)
        else:
            print(f"{k.replace('_', ' ').title()}: {v}")


