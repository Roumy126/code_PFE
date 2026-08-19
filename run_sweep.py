#!/usr/bin/env python
"""Resumable multi-seed sweep over run_experiment.py.

Enumerates a small grid (circuit size x injection method x seed), skipping
any combination whose runs/<run_id>/metrics.json already exists -- so an
overnight sweep can be interrupted (or crash on one config) and picked back
up later without re-running what already finished. Each run is launched as
its own subprocess, so one hanging/crashing config can't take down the rest
of the sweep and no run leaks state (matplotlib figures, DEAP's `creator`
registrations, joblib worker pools) into the next.

The grid below is a placeholder "fixed benchmark set" for laptop-scale
smoke runs -- edit N_QUBITS / INJECTION_METHODS / N_SEEDS directly, or use
the CLI overrides, ahead of the real Phase 2 benchmark circuits.

Example:
    python run_sweep.py --dry-run
    python run_sweep.py --n-seeds 2 --n-qubits 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

N_QUBITS = [8, 12, 20]
INJECTION_METHODS = ["stochastic", "sa"]
N_SEEDS = 5  # roadmap ultimately wants 10-20 seeds per config for review

GENERATIONS = 100
POP_SIZE = 100


def build_grid(n_qubits_list, injection_methods, n_seeds):
    for n_qubits in n_qubits_list:
        for injection_method in injection_methods:
            for seed in range(n_seeds):
                yield {
                    "circuit": "weak_random",
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
    p.add_argument("--n-qubits", type=int, nargs="+", default=N_QUBITS)
    p.add_argument("--injection-methods", choices=["sa", "stochastic"], nargs="+",
                    default=INJECTION_METHODS)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    grid = list(build_grid(args.n_qubits, args.injection_methods, args.n_seeds))

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
