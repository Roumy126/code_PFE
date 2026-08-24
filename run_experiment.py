#!/usr/bin/env python
"""CLI runner for a single configuration of the M1_finale optimization pipeline.

Generates a circuit, runs optimise_circuit_pipeline once, and writes
structured, JSON-serializable artifacts under runs/<run_id>/ instead of
leaving results as stdout prints and loose PNGs in the working directory.
This is the building block for multi-seed sweeps: run it once per
(circuit config, algorithm config, seed) and later aggregate runs/*/metrics.json
into one results table.

Example:
    python run_experiment.py --n-qubits 12 --seed 0 --generations 100 --pop-size 100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from M1_finale.final_m1_script import (  # noqa: E402
    optimise_circuit_pipeline,
    random_weakly_connected_circuit,
    qaoa_maxcut_circuit,
    w_state_circuit,
    qft_circuit,
    hw_efficient_ansatz_circuit,
)

try:
    from qiskit import qpy
except ImportError:  # pragma: no cover
    qpy = None

CIRCUIT_GENERATORS = {
    "weak_random": random_weakly_connected_circuit,
    "qaoa_maxcut": qaoa_maxcut_circuit,
    "w_state": w_state_circuit,
    "qft": qft_circuit,
    "hw_efficient_ansatz": hw_efficient_ansatz_circuit,
}


def make_json_safe(obj):
    """Recursively convert sets/tuples/numpy types into plain JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuit", choices=sorted(CIRCUIT_GENERATORS), default="weak_random")
    p.add_argument("--n-qubits", type=int, default=30)
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--twoq-gates-total", type=int, default=8)
    p.add_argument("--connectivity-edges", type=int, default=5)
    p.add_argument("--qaoa-p", type=int, default=2, help="QAOA layers (--circuit qaoa_maxcut only).")
    p.add_argument("--qaoa-gammas", type=float, nargs="+", default=None,
                    help="QAOA gamma angles, one per layer; random (seeded) if omitted.")
    p.add_argument("--qaoa-betas", type=float, nargs="+", default=None,
                    help="QAOA beta angles, one per layer; random (seeded) if omitted.")
    p.add_argument("--ansatz-reps", type=int, default=1,
                    help="Repetition layers (--circuit hw_efficient_ansatz only).")
    p.add_argument("--injection-method", choices=["sa", "stochastic"], default="stochastic")
    p.add_argument("--fid-threshold", type=float, default=0.9999)
    p.add_argument("--sa-iters", type=int, default=3000)
    p.add_argument("--fidelity-exact-threshold", type=int, default=13,
                    help="Used for the pipeline's own reporting-level fidelity checks (a handful "
                         "of calls per run). Circuits with more qubits than this use an "
                         "approximate Monte-Carlo SWAP-test fidelity instead of the exact "
                         "dense-Operator computation. Benchmarked on real qaoa_maxcut circuits: "
                         "exact costs ~25s/call at n=12, ~145s/call at n=13, ~898s/call at n=14 "
                         "-- 13 is the largest size still cheap enough for ~5 calls/run. Lower "
                         "sa-iters manually for runs above this threshold -- it isn't auto-scaled.")
    p.add_argument("--injection-fidelity-exact-threshold", type=int, default=12,
                    help="Exact-vs-approximate threshold used ONLY inside the injection stage's "
                         "per-trial loop (hundreds of calls/run) -- kept separate from and lower "
                         "than --fidelity-exact-threshold since exact cost there compounds over "
                         "many trials. For genuinely entangled targets (e.g. QAOA) exact is both "
                         "correct and faster than the SWAP-test proxy at n<=12 (measured: 25s "
                         "exact vs. 242.8s approximate per call at n=12) -- see the "
                         "'SWAP-TEST FIDELITY BLOWUP' entry in logs.txt. Not raised to 13/14 "
                         "because per-call exact cost there (145s/898s) times hundreds of trials "
                         "is worse than the approximate path it would replace.")
    p.add_argument("--fidelity-swap-samples", type=int, default=8,
                    help="Random product states averaged per approximate fidelity estimate "
                         "(used for the pipeline's own reporting-level checks, not injection trials).")
    p.add_argument("--fidelity-swap-shots", type=int, default=128,
                    help="Shots per approximate fidelity sample (dominates its cost, ~0.0046s/shot). "
                         "Used for the pipeline's own reporting-level checks, not injection trials.")
    p.add_argument("--injection-fidelity-samples", type=int, default=2,
                    help="Random product states averaged per fidelity estimate INSIDE the injection "
                         "stage's per-trial loop (hundreds of calls). Kept far lower than "
                         "--fidelity-swap-samples because these calls compare circuits that differ by "
                         "a newly-added cross-block entangling gate, which is much more expensive for "
                         "the matrix_product_state simulator than the near-identical pairs the general "
                         "default was tuned against.")
    p.add_argument("--injection-fidelity-shots", type=int, default=16,
                    help="Shots per injection-stage fidelity sample. See --injection-fidelity-samples.")
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--pop-size", type=int, default=400,
                    help="Must be a multiple of 4 (DEAP's NSGA-II tournament selection requires it).")
    p.add_argument("--qubit-duplication-threshold", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0,
                    help="Seeds circuit generation, the GA, and inter-block injection for reproducibility.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--run-id", default=None,
                    help="Defaults to '<circuit>_<n_qubits>q_seed<seed>_<8-char-uuid>'.")
    args = p.parse_args(argv)
    if args.pop_size % 4 != 0:
        p.error(f"--pop-size must be a multiple of 4 (got {args.pop_size}): "
                "DEAP's NSGA-II tournament selection (selTournamentDCD) requires it.")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_id or f"{args.circuit}_{args.n_qubits}q_seed{args.seed}_{uuid.uuid4().hex[:8]}"
    run_dir = (Path(args.runs_dir) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["run_id"] = run_id
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    gen_fn = CIRCUIT_GENERATORS[args.circuit]
    qc = gen_fn(
        n_qubits=args.n_qubits,
        depth=args.depth,
        twoq_gates_total=args.twoq_gates_total,
        connectivity_edges=args.connectivity_edges,
        seed=args.seed,
        p=args.qaoa_p,
        gammas=args.qaoa_gammas,
        betas=args.qaoa_betas,
        reps=args.ansatz_reps,
    )

    # optimise_circuit_pipeline writes figures via relative paths (cwd and
    # cwd/out_figs), so run it from inside the run's own directory to keep
    # each run's artifacts isolated instead of overwriting a shared location.
    prev_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        t0 = time.perf_counter()
        qc_opt, meta = optimise_circuit_pipeline(
            qc,
            injection_method=args.injection_method,
            fid_threshold=args.fid_threshold,
            sa_iters=args.sa_iters,
            sa_seed=args.seed,
            qubit_duplication_threshold=args.qubit_duplication_threshold,
            generations=args.generations,
            pop_size=args.pop_size,
            fidelity_exact_threshold=args.fidelity_exact_threshold,
            fidelity_samples=args.fidelity_swap_samples,
            fidelity_shots=args.fidelity_swap_shots,
            injection_fidelity_samples=args.injection_fidelity_samples,
            injection_fidelity_shots=args.injection_fidelity_shots,
            injection_fidelity_exact_threshold=args.injection_fidelity_exact_threshold,
        )
        wall_clock_s = time.perf_counter() - t0
    finally:
        os.chdir(prev_cwd)

    meta["wall_clock_s"] = wall_clock_s
    (run_dir / "metrics.json").write_text(json.dumps(make_json_safe(meta), indent=2))

    if qpy is not None:
        with open(run_dir / "final_circuit.qpy", "wb") as f:
            qpy.dump(qc_opt, f)

    print(f"\nRun complete: {run_id}")
    print(f"  fidelity_final = {meta['fidelity_final']:.5f}")
    print(f"  depth {meta['depth_before']} -> {meta['depth_after']}")
    print(f"  cost {meta['cost_before']} -> {meta['cost_after']}")
    print(f"  wall clock: {wall_clock_s:.1f}s")
    print(f"  artifacts written to: {run_dir}")
    return meta


if __name__ == "__main__":
    main()
