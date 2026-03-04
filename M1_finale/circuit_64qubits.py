"""
circuit_64qubits.py
====================
Génère un circuit quantique FAIBLEMENT CONNECTÉ à 64 qubits,
puis lance le pipeline d'optimisation de final_m1_script.py sur ce circuit.

Paramètres du circuit (faible connectivité) :
  - n_qubits          : 64
  - depth             : 24   (profondeur modérée)
  - twoq_gates_total  : 12   (très peu de portes à 2 qubits)
  - connectivity_edges: 8    (très peu de liaisons possibles entre qubits)
  - seed              : 2024
"""

from __future__ import annotations
import random
import numpy as np
from qiskit import QuantumCircuit

# ──────────────────────────────────────────────────────────────────────────────
# 1. Générateur de circuit faiblement connecté – 64 qubits
# ──────────────────────────────────────────────────────────────────────────────

def weakly_connected_circuit_64q(
    n_qubits: int = 64,
    depth: int = 24,
    twoq_gates_total: int = 12,   # très faible : peu de portes 2-qubits
    connectivity_edges: int = 8,  # très faible : peu d'arêtes possibles
    use_cz: bool = True,
    seed: int = 2024,
) -> QuantumCircuit:
    """
    Crée un circuit quantique aléatoire à `n_qubits` qubits avec une
    connectivité TRÈS FAIBLE entre les qubits (faible nombre d'arêtes
    autorisées et faible nombre total de portes à 2 qubits).

    Paramètres
    ----------
    n_qubits          : nombre de qubits (défaut 64)
    depth             : nombre de couches / profondeur du circuit
    twoq_gates_total  : nombre total de portes à 2 qubits placées aléatoirement
    connectivity_edges: nombre maximum d'arêtes décrivant la topologie
    use_cz            : si True → CZ, sinon CNOT
    seed              : graine pour la reproductibilité

    Retourne
    --------
    QuantumCircuit Qiskit
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    qc = QuantumCircuit(n_qubits, name="weak_64q")

    # ── Construction de la topologie (arêtes autorisées) ──────────────────────
    edges: set = set()
    attempts = 0
    while len(edges) < connectivity_edges and attempts < 20_000:
        a = rng.randrange(n_qubits)
        b = rng.randrange(n_qubits)
        if a != b:
            edges.add((min(a, b), max(a, b)))
        attempts += 1

    if not edges:
        edges = {(0, 1)}          # au moins une arête garantie
    edges = sorted(edges)

    # ── Layers où une porte 2-qubits sera insérée ─────────────────────────────
    twoq_layers = set(rng.sample(range(depth), k=min(twoq_gates_total, depth)))

    oneq_paulis = ["x", "y", "z"]
    oneq_rots   = ["rx", "ry", "rz"]

    # ── Couches du circuit ─────────────────────────────────────────────────────
    for d in range(depth):
        # On touche un sous-ensemble de qubits (faible densité 1/8)
        k = max(8, n_qubits // 8)
        touched = rng.sample(range(n_qubits), k=k)

        for q in touched:
            if rng.random() < 0.75:          # porte de rotation (continue)
                kind  = rng.choice(oneq_rots)
                theta = float(np_rng.uniform(0, 2 * np.pi))
                if kind == "rx":
                    qc.rx(theta, q)
                elif kind == "ry":
                    qc.ry(theta, q)
                else:
                    qc.rz(theta, q)
            else:                            # porte de Pauli (discrète)
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
# 2. Point d'entrée principal
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    import matplotlib
    matplotlib.use("Agg")          # pas d'affichage graphique (mode serveur)
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  Circuit faiblement connecté – 64 qubits")
    print("=" * 60)

    # ── Génération du circuit ──────────────────────────────────────────────────
    qc = weakly_connected_circuit_64q(
        n_qubits=64,
        depth=24,
        twoq_gates_total=12,
        connectivity_edges=8,
        use_cz=True,
        seed=2024,
    )

    print(f"\n  Qubits           : {qc.num_qubits}")
    print(f"  Portes totales   : {qc.size()}")
    print(f"  Profondeur       : {qc.depth()}")
    print(f"  Portes 2-qubits  : {qc.num_nonlocal_gates()}")

    # ── Sauvegarde du dessin du circuit ──────────────────────────────────────
    os.makedirs("out_figs", exist_ok=True)
    fig = qc.draw("mpl", fold=40)
    fig.savefig("out_figs/circuit_64q_original.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("\n  Dessin sauvegardé → out_figs/circuit_64q_original.png")

    # ── Lancement de l'optimisation via final_m1_script ──────────────────────
    print("\n  Lancement du pipeline d'optimisation (final_m1_script.py)...")
    print("  (importation du module – cela peut prendre quelques secondes)\n")

    try:
        import final_m1_script as m1

        # Utilise la fonction principale d'optimisation si elle existe
        if hasattr(m1, "run_optimization_pipeline"):
            result = m1.run_optimization_pipeline(qc)
            print("\n  Résultat de l'optimisation :", result)
        elif hasattr(m1, "optimize_circuit"):
            result = m1.optimize_circuit(qc)
            print("\n  Circuit optimisé obtenu.")
        else:
            print("  INFO : aucune fonction d'optimisation trouvée dans")
            print("         final_m1_script.py – circuit généré avec succès.")
    except Exception as exc:
        print(f"\n  AVERTISSEMENT : impossible d'importer final_m1_script : {exc}")
        print("  Le circuit a quand même été généré et sauvegardé.")

    print("\n  Terminé.\n")
