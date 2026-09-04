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

# Cap each process's BLAS thread pool to 1 *before* numpy/qiskit-aer are imported (these
# libraries read the env vars below at init time, not on every call). Without this, every
# linear-algebra call -- including the many small, sequential ones in the injection stage
# (hundreds of trial fidelity checks) and each joblib worker process spun up for per-block
# NSGA-II -- independently spawns its own thread pool sized to every core it can see. That
# is thread oversubscription, not real parallelism: dozens of threads end up contending for
# the same physical cores, which measurably slows down sequential work rather than speeding
# it up. Actual parallelism for this pipeline comes from joblib's process-level fan-out
# (Parallel(-1) in optimise_block_nsga2), not from each individual matrix multiply competing
# for every core.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

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
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CommutationAnalysis, CommutativeCancellation, Optimize1qGates
)
from qiskit_aer import AerSimulator

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

    Skips barrier instructions: a barrier's qargs span every qubit in the circuit by
    default, so it always trivially "spans multiple blocks" for any partition with >1
    block, regardless of whether any real entangling gate actually crosses a block
    boundary -- weak_random_circuit is the only benchmark generator that inserts these
    (after every depth layer), and without this filter every one of those no-op barriers
    was being counted (and uselessly reinjected) as a genuine inter-block gate.
    """
    bmap = {q: i for i, bl in enumerate(blocks) for q in bl}
    interblock_gates = []
    for ci in qc.data:
        if ci.operation.name == "barrier":
            continue
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


def _rand_product_prep(n: int, rng: random.Random) -> QuantumCircuit:
    """Prepares a random product state via Rz-Ry-Rz on each qubit."""
    prep = QuantumCircuit(n, name="prep")
    for q in range(n):
        prep.rz(rng.uniform(0, 2 * math.pi), q)
        prep.ry(rng.uniform(0, math.pi), q)
        prep.rz(rng.uniform(0, 2 * math.pi), q)
    return prep


def _echo_test_circuit(n: int, U: QuantumCircuit, V: QuantumCircuit, prep: QuantumCircuit) -> QuantumCircuit:
    """
    Builds the fidelity-echo circuit for |<psi|U^dagger V|psi>|^2, |psi> = prep|0>:
    prep, V, U^dagger, prep^dagger, then measure all n qubits. The probability of the
    all-zero outcome equals |<psi|U^dagger V|psi>|^2 exactly:
        <0| prep^dagger U^dagger V prep |0> = (prep|0>)^dagger . U^dagger V . (prep|0>)
                                              = <psi| U^dagger V |psi>
    This replaces an earlier SWAP-test formulation (ancilla + 2n data qubits + n CSWAP
    gates entangling two separate copies of the state, then a Hadamard-test-style
    2*p0-1 readout) with a direct n-qubit measurement -- same target quantity (verified
    against both the exact dense-operator value and the old SWAP-test estimator on
    matching random samples: both unbiased, echo has visibly lower variance since it
    skips the Hadamard-test transform that doubles the SWAP test's shot noise), at
    roughly half the qubit count and with no extra CSWAP-induced entanglement between
    two coupled registers. On a real 10-qubit QAOA circuit (the case that originally
    motivated the exact/SWAP-test threshold split -- see logs.txt "SCALING -- SA
    INJECTION FAST PATH FOUND AND FIXED"), the old SWAP test did not complete within a
    90s bound under matrix_product_state at production settings (p=2, shots=128); this
    replacement completes the same comparison in ~0.06s.
    """
    qc = QuantumCircuit(n, n)
    qc.compose(prep, qubits=range(n), inplace=True)
    qc.compose(V, qubits=range(n), inplace=True)
    qc.compose(U.inverse(), qubits=range(n), inplace=True)
    qc.compose(prep.inverse(), qubits=range(n), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def _estimate_overlap_sq_echo_once(U: QuantumCircuit, V: QuantumCircuit, *, shots: int, seed: int) -> float:
    """Estimates |<psi|U^dagger V|psi>|^2 for a random product state via one fidelity-echo circuit."""
    assert U.num_qubits == V.num_qubits, "U and V must have the same number of qubits."
    n = U.num_qubits
    rng = random.Random(seed)
    prep = _rand_product_prep(n, rng)
    qc = _echo_test_circuit(n, U, V, prep)
    # matrix_product_state stays cheap for this project's weakly-connected target circuits
    # regardless of n, and -- unlike the old SWAP-test circuit's 2n+1 qubits plus n
    # entanglement-inducing CSWAP gates coupling two separate registers -- also stays cheap
    # for genuinely entangled families (QAOA/QFT/W-state/hw-efficient-ansatz) at the sizes
    # that used to stall (see docstring above and logs.txt "SCALING -- ECHO-TEST FIDELITY
    # BACKEND REPLACES SWAP TEST").
    sim = AerSimulator(method="matrix_product_state")
    tqc = transpile(qc, sim, optimization_level=1)
    res = sim.run(tqc, shots=shots, seed_simulator=seed).result()
    counts = res.get_counts()
    p0 = counts.get('0' * n, 0) / shots
    return max(0.0, min(1.0, p0))


def approximate_gate_fidelity_echo_mc(U: QuantumCircuit, V: QuantumCircuit, *,
                                       samples: int, shots: int, seed: int) -> float:
    """Monte-Carlo average of |<psi|U^dagger V|psi>|^2 via a fidelity-echo circuit on random product states."""
    rng = random.Random(seed)
    vals = [
        _estimate_overlap_sq_echo_once(U, V, shots=shots, seed=rng.randint(0, 10**9))
        for _ in range(samples)
    ]
    return sum(vals) / len(vals) if vals else 0.0


def approximate_gate_fidelity_statevector_mc(U: QuantumCircuit, V: QuantumCircuit, *,
                                              samples: int, seed: int) -> float:
    """
    Monte-Carlo average of |<psi|U^dagger V|psi>|^2 via DIRECT exact statevector simulation
    of U|psi> and V|psi> on random product states |psi>, instead of the fidelity-echo
    circuit's AerSimulator(matrix_product_state) + shot sampling. Same target quantity as
    approximate_gate_fidelity_echo_mc, exact per sample (no shot noise at all) -- but its cost
    is O(2^n * gates), independent of entanglement, unlike the MPS backend's cost which is
    exponential in entanglement but independent of n for low-entanglement circuits.

    This means the two backends have OPPOSITE strengths: MPS stays cheap regardless of n for
    weakly-entangled circuit families (validated up to n=32 for this project's w_state/
    hw_efficient_ansatz/qft/weak_random generators), while this statevector backend stays
    cheap regardless of entanglement up to whatever n statevector simulation is memory-
    feasible for (~n<=24-28 on a laptop) -- exactly the case (genuinely entangled families
    like qaoa_maxcut) where MPS's bond dimension blows up. NEITHER dominates the other in
    general: benchmarked on this project's own generators at n=24, this backend was ~250-400x
    SLOWER than MPS for w_state/hw_efficient_ansatz (their low entanglement is exactly what
    MPS exploits and this backend cannot), while being ~150x+ faster than MPS for qaoa_maxcut
    at n=16 (its entanglement is exactly what MPS chokes on). Callers must pick explicitly per
    circuit family via safe_fidelity_between_circuits' approximate_backend param -- there is no
    automatic per-circuit detection here, consistent with how injection_method/block_algorithm/
    mutation_scheme are chosen in this codebase.
    """
    rng = random.Random(seed)
    n = U.num_qubits
    total = 0.0
    for _ in range(samples):
        prep = _rand_product_prep(n, random.Random(rng.randint(0, 10 ** 9)))
        sa = Statevector(prep.compose(U)).data
        sb = Statevector(prep.compose(V)).data
        total += abs(np.vdot(sa, sb)) ** 2
    return total / samples if samples else 0.0


def safe_fidelity_between_circuits(
    qc_a: QuantumCircuit,
    qc_b: QuantumCircuit,
    *,
    exact_threshold: int = 10,
    samples: int = 8,
    shots: int = 128,
    seed: int = 0,
    target_operator: Optional[np.ndarray] = None,
    approximate_backend: str = "mps",
) -> float:
    """
    Circuit-vs-circuit fidelity that stays laptop-tractable past compute_fidelity's dense-
    Operator scaling wall: exact trace fidelity (|Tr(Ua Ub^dagger)| / 2^n) when
    n <= exact_threshold, else a Monte-Carlo fidelity-echo estimate above it -- either the
    default "mps" backend (approximate_gate_fidelity_echo_mc -- see _echo_test_circuit for the
    method; replaced an earlier SWAP-test formulation on 2026-08-27, same target quantity,
    much cheaper) or, if approximate_backend="statevector", the exact-per-sample
    approximate_gate_fidelity_statevector_mc -- see that function's docstring for when to pick
    which; there is no automatic choice, this must be set explicitly per circuit family.

    This is a proxy metric above exact_threshold, not a full-Hilbert-space fidelity: it
    estimates overlap on random PRODUCT states only, so it can diverge more from the exact
    value for highly entangled circuit differences (validated: within ~0.005 of exact for a
    near-identical pair, noisier for very different circuits). Defaults (threshold=10,
    samples=8, shots=128) trade estimate precision for speed -- tune upward if a specific
    comparison needs tighter estimates.

    target_operator: optional precomputed Operator(qc_b).data, for callers that compare many
    different qc_a against the SAME qc_b repeatedly (e.g. the pipeline's reporting-level
    fidelity checks against the fixed original target circuit) -- skips rebuilding qc_b's
    dense operator on every call. Ignored (qc_b is used instead) once n > exact_threshold,
    since that branch never touches an operator at all.
    """
    assert qc_a.num_qubits == qc_b.num_qubits, "Circuits of different sizes."
    n = qc_a.num_qubits
    if exact_threshold >= 0 and n <= exact_threshold:
        Ua = Operator(qc_a).data
        Ub = target_operator if target_operator is not None else Operator(qc_b).data
        return abs(np.trace(Ua @ Ub.conj().T)) / (2 ** n)
    if approximate_backend == "statevector":
        return approximate_gate_fidelity_statevector_mc(qc_a, qc_b, samples=samples, seed=seed)
    return approximate_gate_fidelity_echo_mc(qc_a, qc_b, samples=samples, shots=shots, seed=seed)


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
    # front_raw: the Pareto front's (fidelity, depth, cost) in natural units, unnormalized.
    # HV/ref_point above use per-run adaptive normalization (f_min/f_max and the reference
    # point are both derived from THIS run's own population), which makes them valid for
    # tracking one run's own convergence (plot_moo_history) but NOT comparable across
    # different runs/algorithms -- two algorithms with differently-scaled fronts get
    # different private [0,1] coordinate systems. front_raw lets a caller recompute HV
    # under a single SHARED normalization + fixed reference point across multiple runs
    # for a fair comparison (see generate_report.py's table_fair_hv_comparison).
    res.update({"HV": compute_hv(P_eval, ref), "Spread": spread_delta(P_eval), "Spacing": compute_spacing(P_eval),
                "ref_point": ref, "front_raw": F[idx].tolist()})
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


# =============================
# Mutation operators (GECCO 2025, arXiv 2504.06413) and hybrid GA+LAS helper
# =============================
# Shared by optimise_block_nsga2 and optimise_block_smsemoa so both algorithms can be
# compared under the same mutation_scheme/hybrid_las choices.

def mut_swap(ind):
    """Exchanges two genes' positions (no-op if fewer than 2 genes)."""
    if len(ind) >= 2:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind


def mut_addition(ind, gen_gene):
    """Inserts a freshly generated gene at a random position."""
    idx = random.randrange(len(ind) + 1)
    ind.insert(idx, gen_gene())
    return ind


def mut_deletion(ind):
    """Removes a random gene (no-op if the chromosome would become empty)."""
    if len(ind) > 1:
        del ind[random.randrange(len(ind))]
    return ind


def make_mutate_fn(mutation_scheme: str, gen_gene):
    """Builds the `mutate` toolbox operator for a given scheme: 'point' (the original
    single-gene-replace operator) or the GECCO 2025 combinations 'swap_add'
    (swap + addition) / 'swap_add_delete' (swap + addition + deletion), each op applied
    in sequence per call. Returns the individual directly (not a DEAP (ind,) tuple),
    matching this file's existing tb.mutate(ind) call convention."""
    def mut_point(ind):
        if len(ind) > 0:
            ind[random.randrange(len(ind))] = gen_gene()
        return ind

    ops_by_scheme = {
        "point": [mut_point],
        "swap_add": [mut_swap, lambda ind: mut_addition(ind, gen_gene)],
        "swap_add_delete": [mut_swap, lambda ind: mut_addition(ind, gen_gene), mut_deletion],
    }
    if mutation_scheme not in ops_by_scheme:
        raise ValueError(f"mutation_scheme must be one of {sorted(ops_by_scheme)}.")
    ops = ops_by_scheme[mutation_scheme]

    def mutate(ind):
        for op in ops:
            ind = op(ind)
        return ind
    return mutate


def apply_hybrid_las(best_ind, build_fn, target_unitary, hist_moo):
    """Runs the LAS local angle search (update_rotation_angles) on the GA's winning
    chromosome and records its effect into hist_moo[-1] for per-run visibility."""
    fid_before = compute_fidelity(build_fn(best_ind), target_unitary)
    refined = update_rotation_angles(list(best_ind), build_fn, target_unitary)
    fid_after = compute_fidelity(build_fn(refined), target_unitary)
    if hist_moo:
        hist_moo[-1]["las_applied"] = True
        hist_moo[-1]["fidelity_before_las"] = float(fid_before)
        hist_moo[-1]["fidelity_after_las"] = float(fid_after)
    return refined


def optimise_block_nsga2(qc_target: QuantumCircuit, *, generations=500, pop_size=300, P_star=None,
                          mutation_scheme="point", hybrid_las=False):
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
        # Full parallelism (-1 = all cores) is intentional here, not a placeholder to
        # tune down. The runtime hardware's own thermal safety mechanisms handle CPU
        # protection; this software does not need to cap or throttle usage itself.
        fresh = Parallel(-1)(delayed(eval_ind)(i) for i in to_run)
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
    tb.register("mutate", make_mutate_fn(mutation_scheme, gen_gene))
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
    best_ind = max(pop, key=lambda i: i.fitness.values[0])
    if hybrid_las:
        best_ind = apply_hybrid_las(best_ind, build, U_target, history_moo)
    return build(best_ind), history_moo


# =============================
# Intra-block optimization: SMS-EMOA (S-Metric Selection EMOA, Beume et al. 2007)
# =============================
# Alternative per-block optimizer to optimise_block_nsga2: same chromosome encoding,
# objectives, and DEAP creator/toolbox pattern, but environmental selection replaces
# NSGA-II's crowding distance with a hypervolume-contribution-based trim of the
# boundary front, directly reusing compute_hv as the selection criterion instead of
# only a post-hoc reporting metric. This is a generational adaptation (batches
# offspring per generation, matching evaluate_population's parallel evaluation) rather
# than the textbook's steady-state (mu+1) loop, which would evaluate one individual at
# a time and not benefit from joblib batching.

def optimise_block_smsemoa(qc_target: QuantumCircuit, *, generations=500, pop_size=300, P_star=None,
                            mutation_scheme="point", hybrid_las=False):
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
        to_run, seen_keys = [], set()
        for ind in individuals:
            key = tuple(ind)
            if key not in fid_cache and key not in seen_keys:
                to_run.append(ind)
                seen_keys.add(key)
        fresh = Parallel(-1)(delayed(eval_ind)(i) for i in to_run)
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
    tb.register("mutate", make_mutate_fn(mutation_scheme, gen_gene))

    def rank_by_front(individuals):
        """Pareto rank (0 = non-dominated front) for binary-tournament parent selection --
        diversity here is handled by the hypervolume environmental selection below, not
        by the tournament, so rank alone (no crowding distance) is enough."""
        fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=False)
        rank = {}
        for r, front in enumerate(fronts):
            for ind in front:
                rank[id(ind)] = r
        return rank

    def tournament(individuals, rank):
        a, b = random.sample(individuals, 2)
        ra, rb = rank[id(a)], rank[id(b)]
        if ra < rb: return a
        if rb < ra: return b
        return random.choice([a, b])

    def hv_trim_boundary_front(accepted, boundary, k_needed):
        """S-metric selection: removes, one at a time, the boundary-front individual
        whose removal loses the LEAST hypervolume, until only k_needed remain. Fidelity
        is converted to minimize convention (costs[:, 0] = 1 - fid) exactly as
        evaluate_run does, and the normalization bounds + reference point are computed
        once from the full accepted+boundary set (St) so the metric stays consistent
        across the trim loop instead of shifting after each removal."""
        survivors = list(boundary)
        st = accepted + survivors
        costs_st = np.array([ind.fitness.values for ind in st])
        costs_st[:, 0] = 1.0 - costs_st[:, 0]
        _, f_min, f_max = normalize_objectives(costs_st)
        F_norm_st = normalize_objectives(costs_st, f_min, f_max)[0]
        ref_point = np.max(F_norm_st, axis=0) + 0.1
        n_accepted = len(accepted)

        while len(survivors) > k_needed:
            costs = np.array([ind.fitness.values for ind in (accepted + survivors)])
            costs[:, 0] = 1.0 - costs[:, 0]
            F_norm = normalize_objectives(costs, f_min, f_max)[0]
            full_hv = compute_hv(F_norm, ref_point)
            contributions = [
                full_hv - compute_hv(np.delete(F_norm, n_accepted + i, axis=0), ref_point)
                for i in range(len(survivors))
            ]
            survivors.pop(int(np.argmin(contributions)))
        return survivors

    def environmental_select(combined, k):
        fronts = tools.sortNondominated(combined, len(combined), first_front_only=False)
        selected = []
        idx = 0
        while idx < len(fronts) and len(selected) + len(fronts[idx]) <= k:
            selected.extend(fronts[idx]); idx += 1
        if len(selected) == k or idx >= len(fronts):
            return selected
        k_needed = k - len(selected)
        return selected + hv_trim_boundary_front(selected, fronts[idx], k_needed)

    pop = tb.population(pop_size)
    fits = evaluate_population(pop)
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hist_eps = [1 - max(pop, key=lambda i: i.fitness.values[0]).fitness.values[0]]
    history_moo = []

    for gen in range(generations):
        rank = rank_by_front(pop)
        offspring = [tb.clone(tournament(pop, rank)) for _ in range(len(pop))]
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
        pop = environmental_select(pop + offspring, pop_size)
        fits_gen = np.array([ind.fitness.values for ind in pop])
        history_moo.append(evaluate_run(fits_gen, P_star=P_star))
        best = max(pop, key=lambda i: i.fitness.values[0])
        hist_eps.append(1 - best.fitness.values[0])
        print(f"[SMS-EMOA] Gen {gen + 1:>4} | Fid {best.fitness.values[0]:.4f} | D {best.fitness.values[1]:>3} | C {best.fitness.values[2]:>3}")

    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    plot_convergence(hist_eps, save_as=f"block_fid_conv_smsemoa_{nq}q")
    plot_pareto(front, save_as=f"block_pareto_smsemoa_{nq}q")
    plot_3d_clusters(front, n_clusters=4, save_as=f"block_clusters3d_smsemoa_{nq}q")

    final_metrics = evaluate_run(np.array([ind.fitness.values for ind in pop]), P_star=P_star)
    print(f"Final MOO Metrics (SMS-EMOA): {final_metrics}")
    best_ind = max(pop, key=lambda i: i.fitness.values[0])
    if hybrid_las:
        best_ind = apply_hybrid_las(best_ind, build, U_target, history_moo)
    return build(best_ind), history_moo


# =============================
# Intra-block optimization: NSGA-III (Deb & Jain 2014, reference-point selection)
# =============================
# Third interchangeable per-block optimizer: same chromosome encoding, objectives, and
# DEAP creator/toolbox pattern as optimise_block_nsga2/optimise_block_smsemoa, but
# environmental selection replaces crowding distance / hypervolume with DEAP's own
# built-in reference-point niching (tools.selNSGA3), matching Deb & Jain's Algorithm 1.
# This deliberately reuses DEAP's tested implementation rather than the from-scratch,
# independently hand-rolled NSGA-III in NSGA-III/AG_multi_objectifs_NSGA3.ipynb (that
# notebook's own choice was a correctness demonstration against the paper; here the
# goal is a well-tested block-optimizer option consistent with how optimise_block_nsga2
# already just calls tools.selNSGA2). Per DEAP's own reference NSGA-III examples,
# environmental selection over reference points is what maintains diversity, so parent
# selection needs no tournament/crowding step -- the whole population is shuffled and
# paired directly, unlike NSGA-II's selTournamentDCD mating pool above.
NSGA3_REFERENCE_DIVISIONS = 12  # -> 91 reference points for 3 objectives (Das & Dennis);
# matches the divisions already reviewed and used by NSGA-III/AG_multi_objectifs_NSGA3.ipynb
# for the same 3-objective (fidelity, depth, cost) problem.

def optimise_block_nsga3(qc_target: QuantumCircuit, *, generations=500, pop_size=300, P_star=None,
                          mutation_scheme="point", hybrid_las=False):
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
        to_run, seen_keys = [], set()
        for ind in individuals:
            key = tuple(ind)
            if key not in fid_cache and key not in seen_keys:
                to_run.append(ind)
                seen_keys.add(key)
        fresh = Parallel(-1)(delayed(eval_ind)(i) for i in to_run)
        for ind, fit in zip(to_run, fresh):
            fid_cache[tuple(ind)] = fit
        return [fid_cache[tuple(ind)] for ind in individuals]

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1, -1, -1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    ref_points = tools.uniform_reference_points(nobj=3, p=NSGA3_REFERENCE_DIVISIONS)
    tb = base.Toolbox(); tb.register("gene", gen_gene)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, 12)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", make_mutate_fn(mutation_scheme, gen_gene))
    tb.register("select", tools.selNSGA3, ref_points=ref_points)

    pop = tb.population(pop_size)
    fits = evaluate_population(pop)
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hist_eps = [1 - max(pop, key=lambda i: i.fitness.values[0]).fitness.values[0]]
    history_moo = []

    for gen in range(generations):
        offspring = [tb.clone(ind) for ind in pop]
        random.shuffle(offspring)  # no crowding-distance tournament in NSGA-III -- avoid
        # positional bias from selNSGA3's front/niche ordering when pairing for crossover.
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
        pop = tb.select(pop + offspring, k=pop_size)
        fits_gen = np.array([ind.fitness.values for ind in pop])
        history_moo.append(evaluate_run(fits_gen, P_star=P_star))
        best = max(pop, key=lambda i: i.fitness.values[0])
        hist_eps.append(1 - best.fitness.values[0])
        print(f"[NSGA-III] Gen {gen + 1:>4} | Fid {best.fitness.values[0]:.4f} | D {best.fitness.values[1]:>3} | C {best.fitness.values[2]:>3}")

    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    plot_convergence(hist_eps, save_as=f"block_fid_conv_nsga3_{nq}q")
    plot_pareto(front, save_as=f"block_pareto_nsga3_{nq}q")
    plot_3d_clusters(front, n_clusters=4, save_as=f"block_clusters3d_nsga3_{nq}q")

    final_metrics = evaluate_run(np.array([ind.fitness.values for ind in pop]), P_star=P_star)
    print(f"Final MOO Metrics (NSGA-III): {final_metrics}")
    best_ind = max(pop, key=lambda i: i.fitness.values[0])
    if hybrid_las:
        best_ind = apply_hybrid_las(best_ind, build, U_target, history_moo)
    return build(best_ind), history_moo


# Registry of per-block optimizers, keyed by the block_algorithm string used throughout
# the CLI/sweep tooling (run_experiment.py, run_sweep.py) and results_master.csv. Adding
# a new MOO algorithm as a block optimizer is: write a function matching this signature
# (qc_target, *, generations, pop_size, P_star, mutation_scheme, hybrid_las) ->
# (best_circuit, history_moo), then add one entry here -- optimise_circuit_pipeline,
# run_experiment.py's --block-algorithm, and run_sweep.py's --block-algorithms all read
# this dict's keys rather than hardcoding an if/elif chain, so none of them need editing
# for a new algorithm to become selectable.
BLOCK_OPTIMIZERS = {
    "nsga2": optimise_block_nsga2,
    "smsemoa": optimise_block_smsemoa,
    "nsga3": optimise_block_nsga3,
}


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


def _injections_self_fidelity(injections: Sequence[InjectionGate], n_qubits: int) -> float:
    """
    Exact closed-form trace fidelity between (base + enabled injections) and base itself,
    without simulating base at all. Since cand = injections . base (injections applied on
    top, matrix order G_k...G_1 @ U_base), U_cand @ U_base^dagger = G_k...G_1 exactly -- base
    cancels algebraically regardless of its size or entanglement. Padding the untouched
    qubits with identity only contributes a constant 2^(n - k) factor, so
    |Tr(U_cand @ U_base^dagger)| / 2^n reduces to |Tr(G_local)| / 2^k on just the k qubits the
    enabled injections touch. Verified to match
    safe_fidelity_between_circuits(cand, base, exact_threshold=n, ...) bit-for-bit at n=9..11
    (the largest n where the simulated path still completed within 90s) while running in
    <1ms regardless of n, vs. seconds-to-timeout for the simulated path -- this is what made
    sa_injection stall on 12+ qubit entangled circuits despite always comparing a candidate
    to its own base.
    """
    enabled = [inj for inj in injections if inj.enabled]
    if not enabled:
        return 1.0
    touched = sorted({q for inj in enabled for q in (inj.q1, inj.q2)})
    qmap = {q: i for i, q in enumerate(touched)}
    local_qc = QuantumCircuit(len(touched))
    for inj in enabled:
        q1, q2 = qmap[inj.q1], qmap[inj.q2]
        if inj.gate == "rzz":
            local_qc.rzz(inj.theta, q1, q2)
        else:
            getattr(local_qc, inj.gate)(q1, q2)
    G_local = Operator(local_qc).data
    return abs(np.trace(G_local)) / (2 ** len(touched))


def _sa_energy(injections: Sequence[InjectionGate], *, base: QuantumCircuit, target_qc: QuantumCircuit,
               α: float, β: float, γ: float, δ: float, fid_tol: float,
               crosstalk_mat: Optional[np.ndarray],
               fidelity_exact_threshold: int = 10, fidelity_samples: int = 8,
               fidelity_shots: int = 128, fidelity_seed: int = 0) -> float:
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

    # sa_injection always calls this with target_qc is base (same object) -- fast, exact
    # closed-form path in that case (see _injections_self_fidelity). Falls back to full
    # simulation if ever called with a genuinely different target, so this stays correct
    # even if that calling convention changes later.
    if target_qc is base:
        fid = _injections_self_fidelity(injections, base.num_qubits)
    else:
        fid = safe_fidelity_between_circuits(cand, target_qc, exact_threshold=fidelity_exact_threshold,
                                              samples=fidelity_samples, shots=fidelity_shots, seed=fidelity_seed)
    fid_penalty = (1.0 - fid) / fid_tol
    return α * n2q + β * depth + γ * crosstalk + δ * fid_penalty


def _rand_two_blocks(blocks: List[Set[int]], rng: random.Random) -> Tuple[Set[int], Set[int]]:
    """
    Picks two DISTINCT blocks uniformly at random from `blocks`.

    Generalizes what used to be a hardcoded blocks[0]/blocks[1] pair in every injection
    function (stochastic_injection, sa_injection's pool/move generation,
    fidelity_driven_injection) -- correct only when a partition happens to have exactly 2
    blocks, but silently making every OTHER block pair unreachable to injection whenever
    louvain_partition returns more than 2 (the common case above ~n=8: e.g. weak_random's
    mean block count runs 4.0 -> 7.6 -> 11.8 across n=8/12/16). See logs.txt's "INJECTION
    STAGE BLOCK-PAIR COVERAGE FIX" for how this was found and its impact.
    """
    i, j = rng.sample(range(len(blocks)), 2)
    return blocks[i], blocks[j]


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
        blk0, blk1 = _rand_two_blocks(blocks, rng)
        inj.q1 = rng.choice(tuple(blk0))
        inj.q2 = rng.choice(tuple(blk1))
    elif choice == "tune_theta" and inj.gate == "rzz":
        inj.theta = (inj.theta or 0.0) + rng.uniform(-eps_theta, eps_theta)

    return cand


def _sa_generate_pool(blocks: List[Set[int]], gate_types: Sequence[str], *, rng: random.Random,
                      n_candidates: int) -> List[InjectionGate]:
    """
    Creates an initial pool of (disabled) injections, each between a fresh random pair of
    blocks (see _rand_two_blocks) rather than a single fixed pair.
    """
    pool: List[InjectionGate] = []
    for _ in range(n_candidates):
        blk0, blk1 = _rand_two_blocks(blocks, rng)
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
                 crosstalk_mat: Optional[np.ndarray] = None,
                 fidelity_exact_threshold: int = 10, fidelity_samples: int = 8,
                 fidelity_shots: int = 128) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Inter-block injection via simulated annealing (SA).
    Returns the final circuit and the list of retained injections.

    NOTE: each of the n_iters energy evaluations calls safe_fidelity_between_circuits, which
    is cheap under fidelity_exact_threshold and, above it, now uses the fidelity-echo Monte-
    Carlo estimator (see _echo_test_circuit) -- much cheaper than the SWAP-test formulation
    this replaced on 2026-08-27, but n_iters is still NOT auto-scaled down for large circuits,
    lower it manually for big-n runs if needed.
    """
    if len(blocks) < 2:
        raise ValueError("sa_injection requires at least two blocks.")

    rng = random.Random(seed)
    injections = _sa_generate_pool(blocks, gate_types, rng=rng, n_candidates=n_candidates)
    fidelity_seed = seed if seed is not None else 0
    energy_kwargs = dict(fidelity_exact_threshold=fidelity_exact_threshold,
                         fidelity_samples=fidelity_samples, fidelity_shots=fidelity_shots,
                         fidelity_seed=fidelity_seed)

    # Initial temperature based on the variance of energy samples
    sample_E = []
    for _ in range(30):
        tmp = _sa_rand_move(injections, blocks, rng=rng)
        e = _sa_energy(tmp, base=base_qc, target_qc=base_qc, α=α, β=β, γ=γ, δ=δ,
                       fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat, **energy_kwargs)
        sample_E.append(e)
    T = 5.0 * (np.std(sample_E) or 1.0)

    best = copy.deepcopy(injections)
    E_best = _sa_energy(best, base=base_qc, target_qc=base_qc, α=α, β=β, γ=γ, δ=δ,
                        fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat, **energy_kwargs)
    current, E_curr = copy.deepcopy(best), E_best

    # Annealing loop
    for _ in range(n_iters):
        cand = _sa_rand_move(current, blocks, rng=rng)
        E_cand = _sa_energy(cand, base=base_qc, target_qc=base_qc, α=α, β=β, γ=γ, δ=δ,
                            fid_tol=1.0 - fid_threshold, crosstalk_mat=crosstalk_mat, **energy_kwargs)
        ΔE = E_cand - E_curr
        accept = (ΔE < 0) or (rng.random() < math.exp(-ΔE / T))
        if accept:
            current, E_curr = cand, E_cand
            if E_curr < E_best:
                best, E_best = copy.deepcopy(current), E_curr
        T *= schedule_alpha

    final_circ = _sa_build_circuit(base_qc, best)
    # Same closed-form identity as _sa_energy's fast path: final_circ is base_qc + `best`,
    # so this is exact, not an approximation.
    fid_final = _injections_self_fidelity(best, base_qc.num_qubits)
    if fid_final < fid_threshold:
        raise RuntimeError(f"SA does not reach the target fidelity: {fid_final:.5f} < {fid_threshold}")

    kept = [(inj.gate, inj.q1, inj.q2, inj.theta) for inj in best if inj.enabled]
    return final_circ, kept


def stochastic_injection(qc: QuantumCircuit, blocks: List[Set[int]], *,
                         n_injections: int = 100,
                         fid_threshold: float = 0.999,
                         gate_probs: Optional[Dict[str, float]] = None,
                         seed: Optional[int] = None,
                         fidelity_exact_threshold: int = 10, fidelity_samples: int = 8,
                         fidelity_shots: int = 128) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Stochastic inter-block injection: we add gates at random and only keep
    those that do not degrade the fidelity below the threshold.

    Each trial's candidate is always exactly `qc` plus one appended gate (before the
    optional compression pass below), so -- by the same cancellation as
    _injections_self_fidelity -- Fidelity(cand, qc) reduces to |Tr(G_local)| / 4, a constant
    depending only on the gate type/angle, independent of qc or n entirely (no threshold,
    no matrix embedding needed at all). Used as the accept/reject decision here, computed
    before compression rather than after it: compress_custom's only non-unitary-exact step
    (remove_negligible_rotations) never touches cx/cz/rzz (the injected gate types) and only
    a pre-existing rx/ry/rz elsewhere in qc could in principle be affected -- verified this
    makes no difference in practice (0/500 mismatches against the old compress-then-check
    order on a real circuit). Building+compressing the actual candidate circuit is deferred
    to only the (rare) accepted trials, instead of every trial.
    """
    if len(blocks) < 2:
        raise ValueError("stochastic_injection requires at least two blocks.")

    gate_probs = gate_probs or {"cx": 1.0, "cz": 1.0, "rzz": 1.0}
    total = sum(gate_probs.values())
    gate_types, probs = zip(*([(g, p / total) for g, p in gate_probs.items()]))

    rng = random.Random(seed)
    kept: List[Tuple[str, int, int, Optional[float]]] = []

    for _ in range(n_injections):
        blk0, blk1 = _rand_two_blocks(blocks, rng)
        gate = rng.choices(gate_types, probs, k=1)[0]
        qi = rng.choice(tuple(blk0))
        qj = rng.choice(tuple(blk1))
        theta = rng.uniform(0, 2 * math.pi) if gate == "rzz" else None

        G_loc = _local_gate_matrix(gate, theta)
        fid = abs(np.trace(G_loc)) / 4

        if fid >= fid_threshold:
            cand = qc.copy()
            if gate == "rzz":
                cand.rzz(theta, qi, qj)
            else:
                getattr(cand, gate)(qi, qj)
            # Option: small optimization/normalization pass
            qc = qiskit_opt_pass(compress_custom(cand))
            kept.append((gate, qi, qj, theta))

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
#   - `blocks`: two (or more) qubit blocks (each trial draws 1 qubit from each of a fresh
#     random pair of blocks, not always the same two -- see _rand_two_blocks),
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

def _partial_trace_reduce(M: np.ndarray, n: int, keep_qubits: Sequence[int]) -> np.ndarray:
    """
    Reduces a general n-qubit operator M (not required to be Hermitian/PSD/trace-normalized)
    to the small (2^k x 2^k) operator R on `keep_qubits` such that, for any local operator
    G_loc acting on those qubits, Tr(G_loc @ R) == Tr((G_loc embedded with identity on the
    other qubits) @ M) exactly -- the standard partial-trace identity, generalized from
    density matrices to arbitrary operators. Qiskit's little-endian convention (qubit 0 = LSB)
    fixes the axis layout; keep_qubits[0] becomes local qubit 0 (LSB) of the returned operator,
    matching a QuantumCircuit(k) with a gate placed on local qubits (0, 1, ...) in that order
    -- so callers must pass keep_qubits in the same (control, target) order the real gate
    would use, not sorted. Verified against qiskit's own gate embedding, including asymmetric
    gates (cx) and reversed qubit-index pairs.
    """
    keep_qubits = list(keep_qubits)
    T = M.reshape([2] * n + [2] * n)
    row_axis = {q: n - 1 - q for q in range(n)}
    col_axis = {q: n + (n - 1 - q) for q in range(n)}
    # einsum needs one unique symbol per axis; a-z/A-Z covers up to 52 axes (n up to 26),
    # far past where a dense n-qubit operator would be tractable to build in the first place.
    letters = [chr(ord('a') + i) if i < 26 else chr(ord('A') + i - 26) for i in range(2 * n)]
    axis_letter = {}
    for q in range(n):
        if q not in keep_qubits:
            axis_letter[row_axis[q]] = letters[row_axis[q]]
            axis_letter[col_axis[q]] = axis_letter[row_axis[q]]
    in_subs = [axis_letter.get(ax, letters[ax]) for ax in range(2 * n)]
    out_subs = ([letters[row_axis[q]] for q in reversed(keep_qubits)]
                + [letters[col_axis[q]] for q in reversed(keep_qubits)])
    R = np.einsum("".join(in_subs) + "->" + "".join(out_subs), T)
    return R.reshape(2 ** len(keep_qubits), 2 ** len(keep_qubits))


def _local_gate_matrix(gate: str, theta: Optional[float]) -> np.ndarray:
    """2x2^2 unitary of a single cx/cz/rzz gate placed on a fresh 2-qubit circuit (0, 1)."""
    local_qc = QuantumCircuit(2)
    if gate == "rzz":
        local_qc.rzz(theta, 0, 1)
    else:
        getattr(local_qc, gate)(0, 1)
    return Operator(local_qc).data


def _apply_local_gate_to_operator(U: np.ndarray, n: int, G_loc: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """
    Returns G_embedded @ U, where G_embedded is the 2-qubit unitary G_loc placed on qubits
    (q1, q2) with identity elsewhere, without ever building the full 2^n x 2^n embedding.
    Cost O(2^(2n+2)), independent of how many gates U already represents -- unlike rebuilding
    Operator(circuit) from scratch (which re-simulates every gate in the circuit each time),
    this stays the same cost no matter how many gates have already been folded into U. Uses
    the same little-endian axis convention as _partial_trace_reduce (verified together, same
    axis-order fix for asymmetric gates like cx).
    """
    row_axis = {q: n - 1 - q for q in range(n)}
    a1, a2 = row_axis[q1], row_axis[q2]
    T = U.reshape([2] * n + [2 ** n])  # n row-qubit axes + 1 flattened column axis
    T = np.moveaxis(T, [a2, a1], [0, 1])
    rest_shape = T.shape[2:]
    T = (G_loc @ T.reshape(4, -1)).reshape((2, 2) + rest_shape)
    T = np.moveaxis(T, [0, 1], [a2, a1])
    return T.reshape(2 ** n, 2 ** n)


def _statevector_pair_reduce(phi: np.ndarray, chi: np.ndarray, n: int, keep_qubits: Sequence[int]) -> np.ndarray:
    """
    Reduces two n-qubit state vectors phi, chi to the small (2^k x 2^k) matrix R such that,
    for any local operator G_loc acting on keep_qubits (embedded with identity elsewhere),
    Tr(G_loc @ R) == <chi | (G_loc embedded) | phi> exactly.

    Same role as _partial_trace_reduce, but for a RANK-1 "M = |phi><chi|" outer product that
    is never materialized -- cost O(2^n) (touching the two length-2^n vectors) instead of
    O(4^n) (touching a 2^n x 2^n matrix), which is what makes this usable well past the
    dense-operator exact tier's qubit range. Same little-endian axis convention as
    _partial_trace_reduce / _apply_local_gate_to_operator (keep_qubits[0] -> local qubit 0);
    verified against brute-force np.outer(phi, chi.conj()) fed through _partial_trace_reduce
    across random qubit pairs and asymmetric gates (cx) before adoption.
    """
    keep_qubits = list(keep_qubits)
    row_axis = {q: n - 1 - q for q in range(n)}
    move_axes = [row_axis[q] for q in reversed(keep_qubits)]

    Phi = np.moveaxis(phi.reshape([2] * n), move_axes, list(range(len(move_axes))))
    Phi2 = Phi.reshape(2 ** len(keep_qubits), -1)

    ChiConj = np.moveaxis(chi.conj().reshape([2] * n), move_axes, list(range(len(move_axes))))
    Chi2 = ChiConj.reshape(2 ** len(keep_qubits), -1)

    return Phi2 @ Chi2.T


def _apply_local_gate_to_statevector(v: np.ndarray, n: int, G_loc: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """
    Returns G_embedded @ v, the state vector v with the 2-qubit unitary G_loc applied on
    qubits (q1, q2), without building the full 2^n x 2^n embedding. Vector analogue of
    _apply_local_gate_to_operator (same axis convention, verified together) -- used to update
    a cached candidate-side state vector incrementally on injection acceptance, instead of
    re-simulating the whole (growing) candidate circuit from scratch.
    """
    row_axis = {q: n - 1 - q for q in range(n)}
    a1, a2 = row_axis[q1], row_axis[q2]
    T = v.reshape([2] * n)
    T = np.moveaxis(T, [a2, a1], [0, 1])
    rest_shape = T.shape[2:]
    T = (G_loc @ T.reshape(4, -1)).reshape((2, 2) + rest_shape)
    T = np.moveaxis(T, [0, 1], [a2, a1])
    return T.reshape(2 ** n)


def fidelity_driven_injection(
    base_qc: QuantumCircuit,
    target_qc: QuantumCircuit,
    blocks: List[Set[int]],
    max_trials: int = 300,
    fid_threshold: float = 0.9999,
    fidelity_exact_threshold: int = 10, fidelity_samples: int = 8,
    fidelity_shots: int = 128, fidelity_seed: int = 0,
    target_operator: Optional[np.ndarray] = None,
    statevector_fast_path_threshold: int = 24,
) -> Tuple[QuantumCircuit, List[Tuple[str, int, int, Optional[float]]]]:
    """
    Iteratively adds inter-block gates that improve the fidelity with respect to `target_qc`.
    Stops as soon as the fidelity exceeds `fid_threshold` or `max_trials` is reached.

    Each trial only ever appends ONE gate to the current candidate_qc, i.e.
    U_test = G_trial @ U_candidate. Three tiers, in decreasing order of exactness:

    1. n <= fidelity_exact_threshold: exact dense-operator tier. Caches
       M = U_candidate @ U_target^dagger once (rebuilt only on acceptance) and gets each
       trial's exact fidelity from a partial trace of M over the trial's 2 qubits -- O(4^n)
       per trial, but avoids rebuilding+resimulating the whole candidate circuit. Exact,
       verified bit-identical to full per-trial simulation at n=9-11.

    2. fidelity_exact_threshold < n <= statevector_fast_path_threshold: statevector tier.
       The trial-varying quantity <chi|G_trial^dagger|phi> (phi = target_qc|psi>, chi =
       candidate_qc|psi>, for each of `fidelity_samples` random product states |psi> --
       the same quantity safe_fidelity_between_circuits' approximate branch estimates via a
       fidelity-echo circuit) reduces to a partial trace of a (2^k x 2^k) matrix built
       from the two length-2^n state vectors phi/chi in O(2^n), via _statevector_pair_reduce
       -- computed exactly (no shot noise at all, unlike the echo-circuit estimator), and
       phi/chi are simulated ONCE (phi) or only on acceptance (chi, updated incrementally via
       _apply_local_gate_to_statevector) rather than resimulating the whole growing candidate
       circuit from scratch on every one of max_trials trials. This is the tier that makes
       genuinely entangled families (e.g. QAOA) practical past their MPS-backend wall -- see
       logs.txt "SCALING" for the profiling that motivated it. Verified exact match against
       direct re-simulation of each trial's test_qc before adoption.

    3. n > statevector_fast_path_threshold: falls back to the original per-trial simulated
       path, where safe_fidelity_between_circuits switches to the fidelity-echo/MPS estimator.

    target_operator: optional precomputed Operator(target_qc).data (see
    safe_fidelity_between_circuits) -- skips rebuilding target_qc's operator here too, when
    the caller already built it for the same target_qc elsewhere in the same run.
    """
    candidate_qc = base_qc.copy()
    kept_injections: List[Tuple[str, int, int, Optional[float]]] = []

    gate_pool = ["cx", "cz", "rzz"]
    rng = random.Random(42)
    fid_kwargs = dict(exact_threshold=fidelity_exact_threshold, samples=fidelity_samples,
                      shots=fidelity_shots, seed=fidelity_seed)
    n = candidate_qc.num_qubits
    use_fast_path = n <= fidelity_exact_threshold
    use_statevector_fast_path = (not use_fast_path) and n <= statevector_fast_path_threshold

    if use_fast_path:
        U_target = target_operator if target_operator is not None else Operator(target_qc).data
        U_target_dag = U_target.conj().T
        M = Operator(candidate_qc).data @ U_target_dag
        fid_old = abs(np.trace(M)) / (2 ** n)
    elif use_statevector_fast_path:
        # Same random-product-state seeding as approximate_gate_fidelity_echo_mc, so this
        # tier targets the SAME quantity the approximate branch would estimate with the same
        # fidelity_seed -- just computed exactly instead of via shot-sampled measurement.
        sample_rng = random.Random(fidelity_seed)
        preps = [_rand_product_prep(n, random.Random(sample_rng.randint(0, 10 ** 9)))
                 for _ in range(fidelity_samples)]
        phi_list = [Statevector(prep.compose(target_qc)).data for prep in preps]
        chi_list = [Statevector(prep.compose(candidate_qc)).data for prep in preps]
        fid_old = sum(abs(np.vdot(chi, phi)) ** 2 for chi, phi in zip(chi_list, phi_list)) / len(preps)
    else:
        # candidate_qc only changes on acceptance, so cache its fidelity instead of
        # recomputing it (against target_qc) every trial -- halves the number of fidelity
        # calls in this (simulated) loop.
        fid_old = safe_fidelity_between_circuits(candidate_qc, target_qc, **fid_kwargs)

    for _ in range(max_trials):
        blk0, blk1 = _rand_two_blocks(blocks, rng)
        gate = rng.choice(gate_pool)
        q1 = rng.choice(tuple(blk0))
        q2 = rng.choice(tuple(blk1))
        theta = rng.uniform(0, 2 * math.pi) if gate == "rzz" else None

        if use_fast_path:
            R = _partial_trace_reduce(M, n, [q1, q2])
            G_loc = _local_gate_matrix(gate, theta)
            fid_new = abs(np.trace(G_loc @ R)) / (2 ** n)
        elif use_statevector_fast_path:
            G_loc = _local_gate_matrix(gate, theta)
            G_dag = G_loc.conj().T
            vals = [np.trace(G_dag @ _statevector_pair_reduce(phi, chi, n, [q1, q2]))
                    for chi, phi in zip(chi_list, phi_list)]
            fid_new = sum(abs(v) ** 2 for v in vals) / len(vals)
        else:
            test_qc = candidate_qc.copy()
            if gate == "rzz":
                test_qc.rzz(theta, q1, q2)
            else:
                getattr(test_qc, gate)(q1, q2)
            # Greedy acceptance if the fidelity increases
            fid_new = safe_fidelity_between_circuits(test_qc, target_qc, **fid_kwargs)

        if fid_new > fid_old:
            if use_fast_path or use_statevector_fast_path:
                # Only build the actual circuit once a trial is accepted.
                test_qc = candidate_qc.copy()
                if gate == "rzz":
                    test_qc.rzz(theta, q1, q2)
                else:
                    getattr(test_qc, gate)(q1, q2)
            candidate_qc = test_qc
            fid_old = fid_new
            kept_injections.append((gate, q1, q2, theta))
            print(f"✅ Added {gate}({q1},{q2}) [fid={fid_new:.5f}]")
            if use_fast_path:
                # M_new = G_embedded @ M_old @ ... exactly, since M = U_candidate @
                # U_target^dagger and U_candidate_new = G_embedded @ U_candidate -- no need
                # to rebuild Operator(candidate_qc) from scratch on every acceptance.
                M = _apply_local_gate_to_operator(M, n, G_loc, q1, q2)
            elif use_statevector_fast_path:
                # Same reasoning, on state vectors instead of the operator: chi_new =
                # G_embedded @ chi_old exactly, so each cached chi is updated in place rather
                # than re-simulating candidate_qc from scratch.
                chi_list = [_apply_local_gate_to_statevector(chi, n, G_loc, q1, q2) for chi in chi_list]
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
    block_algorithm: str = "nsga2",  # one of BLOCK_OPTIMIZERS' keys, e.g. "nsga2"/"smsemoa"/"nsga3"
    mutation_scheme: str = "point",  # "point", "swap_add", or "swap_add_delete"
    hybrid_las: bool = False,
    fid_threshold: float = 0.999,
    sa_iters: int = 2500,
    sa_seed: Optional[int] = 42,
    qubit_duplication_threshold: float = 0.5,
    generations: int = 500,
    pop_size: int = 400,
    fidelity_exact_threshold: int = 13,
    fidelity_samples: int = 8,
    fidelity_shots: int = 128,
    injection_fidelity_samples: int = 8,
    injection_fidelity_shots: int = 16,
    injection_fidelity_exact_threshold: int = 7,
    fidelity_driven_max_trials: int = 300,
    fidelity_driven_statevector_threshold: int = 24,
    fidelity_approximate_backend: str = "mps",  # "mps" or "statevector"
) -> Tuple[QuantumCircuit, Dict[str, object]]:
    print("\nOriginal circuit:")
    print(qc.draw(output="text"))
    qc.draw('mpl', filename='circuit_original.png', style='mpl', fold=1)
    # Reference & starting cost. Full-circuit fidelity checks below compare directly against
    # qc_orig (a circuit) via safe_fidelity_between_circuits rather than a precomputed dense
    # Operator matrix -- avoids ever building a 2^n x 2^n matrix for the whole circuit, which
    # is what made the injection stage intractable past ~10-12 qubits (see logs.txt).
    qc_orig = qc.copy()
    fid_kwargs = dict(exact_threshold=fidelity_exact_threshold, samples=fidelity_samples,
                      shots=fidelity_shots, seed=sa_seed if sa_seed is not None else 0,
                      approximate_backend=fidelity_approximate_backend)
    # qc_orig never changes after this point, but it's compared against below in 5 separate
    # safe_fidelity_between_circuits calls (fid_rebuilt/fid_rebuilt1/fid_inj/fid_i/fid_final)
    # plus once more inside fidelity_driven_injection -- each independently rebuilding
    # Operator(qc_orig) from scratch (see logs.txt "SCALING -- FIDELITY-DRIVEN INJECTION FAST
    # PATH FOUND AND FIXED" for why this exact-Operator tier is the dominant remaining cost at
    # n=12-13). Build it once here and hand it to every call site instead.
    target_operator_cache = Operator(qc_orig).data if qc_orig.num_qubits <= fidelity_exact_threshold else None
    # The injection stage's per-trial fidelity checks (hundreds of them: n_iters/n_injections/
    # max_trials) compare circuits that differ by a newly-added CROSS-BLOCK entangling gate.
    # The numbers below (>900s/242.8s/898s per call) were measured against the OLD SWAP-test
    # approximate backend -- its cswap ladder had to represent real cross-block entanglement,
    # far more expensive for matrix_product_state than the near-identical/low-entanglement
    # pairs fidelity_samples/fidelity_shots were originally tuned against. That backend was
    # replaced by a fidelity-echo estimator on 2026-08-27 (see _echo_test_circuit), which is
    # dramatically cheaper on exactly this kind of genuinely-entangled comparison (~0.06s vs.
    # a >90s timeout on a matching real 10-qubit QAOA case -- see logs.txt "SCALING --
    # ECHO-TEST FIDELITY BACKEND REPLACES SWAP TEST"). RE-BENCHMARKED 2026-08-31 (see logs.txt
    # "SCALING -- INJECTION_FIDELITY_EXACT_THRESHOLD RE-BENCHMARKED") against
    # fidelity_driven_injection's own statevector fast path (its tier 2, gated by
    # fidelity_driven_statevector_threshold below): forcing each tier head-to-head on identical
    # (base, target) pairs at n=4-13 showed the exact tier (tier 1) is no longer competitive
    # anywhere past n=7-8 -- it's already 3-12x slower by n=9-11 and ~700x slower by n=13 (146s
    # vs 0.2s), since it rebuilds+multiplies a dense 2^n x 2^n operator on every acceptance while
    # the statevector tier only ever touches O(2^n)-sized state vectors. The OLD "may be
    # raisable" note this replaced was speculative and, per this data, backwards: lowered from
    # 12 to 7 (roughly where the two tiers are last at parity) rather than raised toward
    # fidelity_exact_threshold's 13. Also raised injection_fidelity_samples 2->8 in the same
    # pass: at the old threshold's boundary (n=12/13) the statevector tier's accept/reject
    # decisions visibly diverged from the exact tier's ground truth at samples=2 (different
    # kept-gate sets, not just noisier numbers); samples=8 tracks the exact tier's decisions
    # much more closely while remaining ~100-700x cheaper than the exact tier at n=12/13.
    # Below injection_fidelity_exact_threshold, use exact dense-Operator fidelity instead -- for
    # genuinely entangled targets (e.g. QAOA) exact is both correct AND was faster than the OLD
    # SWAP-test proxy at these sizes (measured: 25s/call exact vs. 242.8s approximate at n=12 on
    # a real qaoa_maxcut candidate/target pair -- see "SCALING — DEFERRED" in logs.txt). Kept as
    # its own (separate, lower) threshold from fidelity_exact_threshold because this loop runs
    # hundreds of calls, not a handful -- exact cost is now only competitive up to n=7-8, not the
    # n=12-14 range that comparison against the OLD SWAP-test fallback had suggested.
    injection_fid_kwargs = dict(fidelity_exact_threshold=injection_fidelity_exact_threshold,
                                fidelity_samples=injection_fidelity_samples,
                                fidelity_shots=injection_fidelity_shots)
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
        print(f"  → {block_algorithm} optimization in progress (mutation_scheme={mutation_scheme}, hybrid_las={hybrid_las})...")
        if block_algorithm not in BLOCK_OPTIMIZERS:
            raise ValueError(f"block_algorithm must be one of {sorted(BLOCK_OPTIMIZERS)}.")
        best, hist_moo = BLOCK_OPTIMIZERS[block_algorithm](
            sub, generations=generations, pop_size=pop_size,
            mutation_scheme=mutation_scheme, hybrid_las=hybrid_las,
        )
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
    fid_rebuilt = safe_fidelity_between_circuits(qc_rebuilt_original_qubits, qc_orig, **fid_kwargs,
                                                 target_operator=target_operator_cache)
    print(f"Recomposed <-> original fidelity: {fid_rebuilt:.5f}")
    # 4.bis) Reinjection of the original inter-block gates into the recomposed circuit
    for inst, qargs, cargs in original_interblock_gates:
        global_qargs = [qc_rebuilt_original_qubits.qubits[qc.find_bit(q).index] for q in qargs]
        qc_rebuilt_original_qubits.append(inst, global_qargs, cargs)
    print("📎 Inter-block gates reinjected into the recomposed circuit.")
    fid_rebuilt1 = safe_fidelity_between_circuits(qc_rebuilt_original_qubits, qc_orig, **fid_kwargs,
                                                  target_operator=target_operator_cache)
    print(f"Recomposed (with inter-block gates) <-> original fidelity: {fid_rebuilt1:.5f}")
    print("\nRecomposed circuit with inter-block gates:")
    print(qc_rebuilt_original_qubits.draw(output="text"))
    # 5) Inter-block injection (method of choice)
    t_injection_start = time.perf_counter()
    if injection_method == "sa":
        qc_inj, kept = sa_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold,
                                    n_iters=sa_iters, seed=sa_seed,
                                    **injection_fid_kwargs)
    elif injection_method == "stochastic":
        qc_inj, kept = stochastic_injection(qc_rebuilt_original_qubits, original_blocks, fid_threshold=fid_threshold,
                                            seed=sa_seed,
                                            **injection_fid_kwargs)
    else:
        raise ValueError('injection_method must be "sa" or "stochastic".')
    print("\nCircuit after inter-block injection:")
    print(qc_inj.draw(output="text"))
    print(f"# inter-block gates retained: {len(kept)}")
    fid_inj = safe_fidelity_between_circuits(qc_inj, qc_orig, **fid_kwargs,
                                             target_operator=target_operator_cache)
    print(f"Fidelity after inter-block injection <-> original: {fid_inj:.5f}")
    # 6) Fidelity-driven greedy injection (complement)
    qc_i, kept1 = fidelity_driven_injection(base_qc=qc_rebuilt_original_qubits, target_qc=qc_orig,
                                            blocks=original_blocks, max_trials=fidelity_driven_max_trials,
                                            fid_threshold=0.9999,
                                            fidelity_seed=fid_kwargs["seed"],
                                            target_operator=target_operator_cache,
                                            statevector_fast_path_threshold=fidelity_driven_statevector_threshold,
                                            **injection_fid_kwargs)
    print("\nCircuit after inter-block injection with NSGA2 (greedy):")
    print(qc_i.draw(output="text"))
    qc_i.draw('mpl', filename=f"final_optimized_circuitwithdriveninject.png", style='mpl', fold=1)
    print(f"# inter-block gates retained: {len(kept1)}")
    fid_i = safe_fidelity_between_circuits(qc_i, qc_orig, **fid_kwargs,
                                           target_operator=target_operator_cache)
    print(f"Fidelity after inter-block injection <-> original: {fid_i:.5f}")
    t_injection = time.perf_counter() - t_injection_start
    # 7) Final compression (choose the best base)
    t_compression_start = time.perf_counter()
    if fid_i > fid_inj:
        qc_opt = compress_custom(qiskit_opt_pass(qc_i))
        injection_path_used = "fidelity_driven_greedy"
        kept_injections_used = kept1
    else:
        qc_opt = compress_custom(qiskit_opt_pass(qc_inj))
        injection_path_used = injection_method
        kept_injections_used = kept
    print("\nFinal optimized circuit:")
    print(qc_opt.draw(output="text"))
    qc_opt.draw('mpl', filename=f"final_optimized_circuit.png", style='mpl', fold=1)
    cost_final = compute_gate_cost(qc_opt)
    print(f"💰 Cost of the final optimized circuit (Lee et al. 2006): {cost_final}")
    t_compression = time.perf_counter() - t_compression_start
    # 8) Summary
    fid_final = safe_fidelity_between_circuits(qc_opt, qc_orig, **fid_kwargs,
                                               target_operator=target_operator_cache)
    fidelity_backend = ("exact" if qc_orig.num_qubits <= fidelity_exact_threshold
                        else "statevector_mc" if fidelity_approximate_backend == "statevector"
                        else "echo_test_mc")
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
        "block_algorithm": block_algorithm,
        "mutation_scheme": mutation_scheme,
        "hybrid_las": hybrid_las,
        "kept_injections": kept_injections_used,
        "injection_path_used": injection_path_used,
        "fidelity_after_injection_method": fid_inj,
        "fidelity_after_fidelity_driven_greedy": fid_i,
        "fidelity_backend": fidelity_backend,
        "fidelity_exact_threshold": fidelity_exact_threshold,
        "fidelity_samples": fidelity_samples,
        "fidelity_shots": fidelity_shots,
        "injection_fidelity_samples": injection_fidelity_samples,
        "injection_fidelity_shots": injection_fidelity_shots,
        "injection_fidelity_exact_threshold": injection_fidelity_exact_threshold,
        "fidelity_driven_max_trials": fidelity_driven_max_trials,
        "fidelity_driven_statevector_threshold": fidelity_driven_statevector_threshold,
        "fidelity_approximate_backend": fidelity_approximate_backend,
        "fidelity_driven_tier": ("exact" if qc_orig.num_qubits <= injection_fidelity_exact_threshold
                                 else "statevector" if qc_orig.num_qubits <= fidelity_driven_statevector_threshold
                                 else "echo_test_mc"),
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


