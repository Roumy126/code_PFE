"""
circuit_64qubits.py
====================
Generates a WEAKLY CONNECTED quantum circuit with 64 qubits,
then launches the optimization pipeline from final_m1_script.py on this circuit.

Circuit parameters (low connectivity):
  - n_qubits          : 64
  - depth             : 24   (moderate depth)
  - twoq_gates_total  : 12   (very few 2-qubit gates)
  - connectivity_edges: 8    (very few possible links between qubits)
  - seed              : 2024
"""

from __future__ import annotations
import random
import numpy as np
from qiskit import QuantumCircuit

# ──────────────────────────────────────────────────────────────────────────────
# 1. Weakly connected circuit generator – 64 qubits
# ──────────────────────────────────────────────────────────────────────────────

def weakly_connected_circuit_64q(
    n_qubits: int = 64,
    depth: int = 24,
    twoq_gates_total: int = 12,   # very low: few 2-qubit gates
    connectivity_edges: int = 8,  # very low: few possible edges
    use_cz: bool = True,
    seed: int = 2024,
) -> QuantumCircuit:
    """
    Creates a random quantum circuit with `n_qubits` qubits with VERY LOW
    connectivity between qubits (small number of allowed edges and small
    total number of 2-qubit gates).

    Parameters
    ----------
    n_qubits          : number of qubits (default 64)
    depth             : number of layers / circuit depth
    twoq_gates_total  : total number of randomly placed 2-qubit gates
    connectivity_edges: maximum number of edges describing the topology
    use_cz            : if True -> CZ, otherwise CNOT
    seed              : seed for reproducibility

    Returns
    -------
    Qiskit QuantumCircuit
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    qc = QuantumCircuit(n_qubits, name="weak_64q")

    # ── Building the topology (allowed edges) ──────────────────────
    edges: set = set()
    attempts = 0
    while len(edges) < connectivity_edges and attempts < 20_000:
        a = rng.randrange(n_qubits)
        b = rng.randrange(n_qubits)
        if a != b:
            edges.add((min(a, b), max(a, b)))
        attempts += 1

    if not edges:
        edges = {(0, 1)}          # at least one edge guaranteed
    edges = sorted(edges)

    # ── Layers where a 2-qubit gate will be inserted ─────────────────────────────
    twoq_layers = set(rng.sample(range(depth), k=min(twoq_gates_total, depth)))

    oneq_paulis = ["x", "y", "z"]
    oneq_rots   = ["rx", "ry", "rz"]

    # ── Circuit layers ─────────────────────────────────────────────────────
    for d in range(depth):
        # Touch a subset of qubits (low density 1/8)
        k = max(8, n_qubits // 8)
        touched = rng.sample(range(n_qubits), k=k)

        for q in touched:
            if rng.random() < 0.75:          # rotation gate (continuous)
                kind  = rng.choice(oneq_rots)
                theta = float(np_rng.uniform(0, 2 * np.pi))
                if kind == "rx":
                    qc.rx(theta, q)
                elif kind == "ry":
                    qc.ry(theta, q)
                else:
                    qc.rz(theta, q)
            else:                            # Pauli gate (discrete)
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


# ──────────────────────────────────────────────────────────────────────────────
# 2. Main entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    import matplotlib
    matplotlib.use("Agg")          # no graphical display (server mode)
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  Weakly connected circuit – 64 qubits")
    print("=" * 60)

    # ── Circuit generation ──────────────────────────────────────────────────
    qc = weakly_connected_circuit_64q(
        n_qubits=64,
        depth=24,
        twoq_gates_total=12,
        connectivity_edges=8,
        use_cz=True,
        seed=2024,
    )

    print(f"\n  Qubits           : {qc.num_qubits}")
    print(f"  Total gates      : {qc.size()}")
    print(f"  Depth            : {qc.depth()}")
    print(f"  2-qubit gates    : {qc.num_nonlocal_gates()}")

    # ── Saving the circuit drawing ──────────────────────────────────────
    os.makedirs("out_figs", exist_ok=True)
    fig = qc.draw("mpl", fold=40)
    fig.savefig("out_figs/circuit_64q_original.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("\n  Drawing saved -> out_figs/circuit_64q_original.png")

    # ── Launching optimization via final_m1_script ──────────────────────
    print("\n  Launching the optimization pipeline (final_m1_script.py)...")
    print("  (importing the module - this may take a few seconds)\n")

    try:
        import final_m1_script as m1

        # Use the main optimization function if it exists
        if hasattr(m1, "run_optimization_pipeline"):
            result = m1.run_optimization_pipeline(qc)
            print("\n  Optimization result:", result)
        elif hasattr(m1, "optimize_circuit"):
            result = m1.optimize_circuit(qc)
            print("\n  Optimized circuit obtained.")
        else:
            print("  INFO: no optimization function found in")
            print("         final_m1_script.py - circuit generated successfully.")
    except Exception as exc:
        print(f"\n  WARNING: unable to import final_m1_script: {exc}")
        print("  The circuit was still generated and saved.")

    print("\n  Done.\n")
