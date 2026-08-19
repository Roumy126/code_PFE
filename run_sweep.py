#!/usr/bin/env python
"""Resumable multi-seed sweep over run_experiment.py.

Enumerates a grid (circuit family x circuit size x injection method x seed),
skipping any combination whose runs/<run_id>/metrics.json already exists --
so an overnight sweep can be interrupted (or crash on one config) and picked
back up later without re-running what already finished. Each run is launched
as its own subprocess, so one hanging/crashing config can't take down the
rest of the sweep and no run leaks state (matplotlib figures, DEAP's
`creator` registrations, joblib worker pools) into the next.

The grid below is the Phase 2 "fixed benchmark set": all five circuit
generators wired into run_experiment.py's CIRCUIT_GENERATORS, each swept at
a couple of sizes below the 15-qubit fidelity cap plus one at/above it to
deliberately probe the cap -- edit CIRCUITS / CIRCUIT_QUBIT_SIZES /
INJECTION_METHODS / N_SEEDS directly, or use the CLI overrides.

Example:
    python run_sweep.py --dry-run
    python run_sweep.py --circuits weak_random --n-seeds 2 --n-qubits 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CIRCUITS = ["weak_random", "qaoa_maxcut", "w_state", "qft", "hw_efficient_ansatz"]

# Per-circuit qubit sizes. Capped at 8 for now: the injection stage
# (sa_injection / stochastic_injection / fidelity_driven_injection) calls
# compute_fidelity's dense Operator computation on the FULL circuit per
# candidate trial, and that cost scales with the 2^n x 2^n matrix dimension
# -- a single weak_random run at 12 qubits (16x the matrix size of 8 qubits)
# was still running after 1h45m and was killed rather than let finish.
# Raise these once that scaling problem is actually addressed (see
# Final_test/nex_formula's unused SWAP-test approximate-fidelity estimator,
# flagged in logs.txt as a candidate fix) -- not laptop-tractable today.
CIRCUIT_QUBIT_SIZES = {
    "weak_random": [8],
    "qaoa_maxcut": [8],
    "w_state": [8],
    "qft": [8],
    "hw_efficient_ansatz": [8],
}

INJECTION_METHODS = ["stochastic", "sa"]
N_SEEDS = 5  # roadmap ultimately wants 10-20 seeds per config for review

GENERATIONS = 100
POP_SIZE = 100


def build_grid(circuits, n_qubits_override, injection_methods, n_seeds):
    for circuit in circuits:
        sizes = n_qubits_override if n_qubits_override is not None else CIRCUIT_QUBIT_SIZES[circuit]
        for n_qubits in sizes:
            for injection_method in injection_methods:
                for seed in range(n_seeds):
                    yield {
                        "circuit": circuit,
                        "n_qubits": n_qubits,
                        "injection_method": injection_method,
                        "generations": GENERATIONS,
                        "pop_size": POP_SIZE,
                        "seed": seed,
                    }


def run_id_for(cfg: dict) -> str:
    # Encodes every varied parameter so a change to generations/pop_size (or
    # any other swept knob) can't silently collide with a stale prior run.
    return (
        f"{cfg['circuit']}_{cfg['n_qubits']}q_{cfg['injection_method']}"
        f"_g{cfg['generations']}_p{cfg['pop_size']}_seed{cfg['seed']}"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuits", choices=CIRCUITS, nargs="+", default=CIRCUITS)
    p.add_argument("--n-qubits", type=int, nargs="+", default=None,
                    help="Override qubit sizes for every selected circuit "
                         "(default: each circuit's own CIRCUIT_QUBIT_SIZES entry).")
    p.add_argument("--injection-methods", choices=["sa", "stochastic"], nargs="+",
                    default=INJECTION_METHODS)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    grid = list(build_grid(args.circuits, args.n_qubits, args.injection_methods, args.n_seeds))

    print(f"Sweep grid: {len(grid)} configurations")
    if args.dry_run:
        for cfg in grid:
            print(f"  {run_id_for(cfg)}")
        return 0

    n_run, n_skipped, n_failed = 0, 0, 0
    for cfg in grid:
        run_id = run_id_for(cfg)
        if (runs_dir / run_id / "metrics.json").exists():
            print(f"[skip] {run_id} (already done)")
            n_skipped += 1
            continue

        cmd = [
            sys.executable, "run_experiment.py",
            "--run-id", run_id,
            "--runs-dir", str(runs_dir),
            "--circuit", cfg["circuit"],
            "--n-qubits", str(cfg["n_qubits"]),
            "--injection-method", cfg["injection_method"],
            "--generations", str(cfg["generations"]),
            "--pop-size", str(cfg["pop_size"]),
            "--seed", str(cfg["seed"]),
        ]
        print(f"[run]  {run_id}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"[fail] {run_id} (exit code {result.returncode})", file=sys.stderr)
            n_failed += 1
        else:
            n_run += 1

    print(f"\nSweep complete: {n_run} run, {n_skipped} skipped, {n_failed} failed "
          f"(of {len(grid)} total).")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
