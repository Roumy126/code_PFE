from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community  # python-louvain
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, Optimize1qGates
from deap import base, creator, tools
from joblib import Parallel, delayed
import pandas as pd



def random_weakly_connected_circuit(
    n_qubits: int = 40,
    depth: int = 20,
    twoq_gates_total: int = 10,   # TRÈS faible nombre de 2-qubits au total
    connectivity_edges: int = 6,  # TRÈS faible connectivité (peu d'arêtes possibles)
    use_cz: bool = True,
    seed: int = 1234
) -> QuantumCircuit:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits, name="weak_random_30q")
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

# --- Extraction logic from final_m1.ipynb ---
# Instead of copy-pasting everything, I will use nbconvert to execute the notebook 

if __name__ == "__main__":
    # Create the circuit
    qc = random_weakly_connected_circuit(
        n_qubits=30,
        depth=20,
        twoq_gates_total=8,
        connectivity_edges=5,
        use_cz=True,
        seed=42
    )
    print("Circuit généré (30 qubits, faible connectivité) :")
    print(qc.draw())
    

