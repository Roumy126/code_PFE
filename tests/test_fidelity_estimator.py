"""Correctness tests for safe_fidelity_between_circuits (M1_finale.final_m1_script).

Guards against a real bug found while integrating this estimator: the ported SWAP-test
circuit originally declared an extra implicit classical register, so get_counts() returned
two-register keys like '0 0' and a bare counts.get('0', ...) lookup never matched, silently
returning 0.0 for every estimate -- including for identical circuits, which must be ~1.0.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from M1_finale.final_m1_script import safe_fidelity_between_circuits


def _small_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.rz(0.7, 1)
    qc.cx(0, 1)
    qc.cx(2, 3)
    qc.rx(0.3, 2)
    return qc


def test_identical_circuits_approx_fidelity_is_near_one():
    qc = _small_circuit()
    # exact_threshold=-1 forces the Monte-Carlo SWAP-test branch even for this tiny circuit.
    fid = safe_fidelity_between_circuits(qc, qc, exact_threshold=-1, samples=16, shots=256, seed=0)
    assert fid > 0.9, f"identical circuits should estimate close to 1.0, got {fid}"


def test_identical_circuits_exact_fidelity_is_one():
    qc = _small_circuit()
    fid = safe_fidelity_between_circuits(qc, qc, exact_threshold=10, seed=0)
    assert abs(fid - 1.0) < 1e-9


def test_approx_matches_exact_within_tolerance():
    qc_a = _small_circuit()
    qc_b = qc_a.copy()
    qc_b.rz(0.15, 0)  # small perturbation

    exact = abs(np.trace(Operator(qc_a).data @ Operator(qc_b).data.conj().T)) / (2 ** qc_a.num_qubits)
    approx = safe_fidelity_between_circuits(qc_a, qc_b, exact_threshold=-1, samples=32, shots=512, seed=1)
    assert abs(exact - approx) < 0.1, f"exact={exact}, approx={approx}"


def test_exact_threshold_selects_expected_backend():
    qc = _small_circuit()
    # Below/at threshold -> exact path (deterministic, no seed dependence).
    fid_a = safe_fidelity_between_circuits(qc, qc, exact_threshold=10, seed=1)
    fid_b = safe_fidelity_between_circuits(qc, qc, exact_threshold=10, seed=2)
    assert abs(fid_a - fid_b) < 1e-9
    assert abs(fid_a - 1.0) < 1e-9
