"""Fast end-to-end smoke test for M1_finale's optimise_circuit_pipeline.

Runs the full partition -> NSGA-II -> injection -> compression pipeline on a
tiny 6-qubit circuit with tiny GA settings, so it finishes in seconds. Meant
to catch pipeline-breaking regressions before kicking off a long laptop-scale
experiment run, not to check numerical optimization quality.
"""
import random

import numpy as np
import pytest
from qiskit import QuantumCircuit

from M1_finale.final_m1_script import (
    optimise_circuit_pipeline,
    qaoa_maxcut_circuit,
    w_state_circuit,
    qft_circuit,
    hw_efficient_ansatz_circuit,
)


def _two_cluster_circuit() -> QuantumCircuit:
    # Two densely-connected 3-qubit clusters joined by a single weak bridge
    # edge, so Louvain partitioning reliably splits it into >= 2 blocks.
    qc = QuantumCircuit(6)
    for _ in range(2):
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
    qc.rz(0.4, 0)
    for _ in range(2):
        qc.cx(3, 4)
        qc.cx(4, 5)
        qc.cx(3, 5)
    qc.rz(0.4, 4)
    qc.cx(2, 3)  # weak inter-cluster bridge
    return qc


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    random.seed(0)
    np.random.seed(0)

    qc = _two_cluster_circuit()
    qc_opt, meta = optimise_circuit_pipeline(
        qc,
        injection_method="stochastic",
        fid_threshold=0.9,
        generations=3,
        pop_size=8,
        qubit_duplication_threshold=0.6,
    )

    assert qc_opt.num_qubits == qc.num_qubits
    assert 0.0 <= meta["fidelity_final"] <= 1.0 + 1e-6
    assert meta["depth_before"] > 0
    assert meta["depth_after"] >= 0
    assert len(meta["blocks"]) >= 2
    assert len(meta["moo_metrics_per_block"]) == len(meta["blocks"])
    assert set(meta["stage_timings_s"]) == {
        "partitioning", "block_optimization", "injection", "compression",
    }
    assert all(t >= 0 for t in meta["stage_timings_s"].values())
    assert meta["fidelity_backend"] == "exact"  # 6 qubits <= default fidelity_exact_threshold=10


def test_pipeline_runs_end_to_end_via_approximate_fidelity(tmp_path, monkeypatch):
    # Forces the Monte-Carlo SWAP-test path (fidelity_exact_threshold below the circuit's own
    # qubit count) on the same tiny circuit/settings, instead of a bigger real circuit --
    # exercises the approximate backend without the extra runtime a 12+ qubit case would add.
    monkeypatch.chdir(tmp_path)
    random.seed(0)
    np.random.seed(0)

    qc = _two_cluster_circuit()
    qc_opt, meta = optimise_circuit_pipeline(
        qc,
        injection_method="stochastic",
        fid_threshold=0.9,
        generations=3,
        pop_size=8,
        qubit_duplication_threshold=0.6,
        fidelity_exact_threshold=2,
        fidelity_samples=2,
        fidelity_shots=32,
    )

    assert qc_opt.num_qubits == qc.num_qubits
    assert 0.0 <= meta["fidelity_final"] <= 1.0 + 1e-6
    assert meta["fidelity_backend"] == "swap_test_mc"


@pytest.mark.parametrize("make_circuit", [
    lambda: qaoa_maxcut_circuit(n_qubits=6, p=1, seed=0),
    lambda: w_state_circuit(n_qubits=5, seed=0),
    lambda: qft_circuit(n_qubits=4, seed=0),
    lambda: hw_efficient_ansatz_circuit(n_qubits=4, reps=1, seed=0),
], ids=["qaoa_maxcut", "w_state", "qft", "hw_efficient_ansatz"])
def test_benchmark_generators_run_end_to_end(tmp_path, monkeypatch, make_circuit):
    monkeypatch.chdir(tmp_path)
    random.seed(0)
    np.random.seed(0)

    qc = make_circuit()
    qc_opt, meta = optimise_circuit_pipeline(
        qc,
        injection_method="stochastic",
        fid_threshold=0.9,
        generations=3,
        pop_size=8,
        qubit_duplication_threshold=0.6,
    )

    assert qc_opt.num_qubits == qc.num_qubits
    assert 0.0 <= meta["fidelity_final"] <= 1.0 + 1e-6
