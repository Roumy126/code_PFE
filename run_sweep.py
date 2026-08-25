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
a couple of sizes below the fidelity exact_threshold (10) plus one above it
to exercise the approximate Monte-Carlo SWAP-test fidelity path -- edit
CIRCUITS / CIRCUIT_QUBIT_SIZES / INJECTION_METHODS / N_SEEDS directly, or
use the CLI overrides.

Example:
    python run_sweep.py --dry-run
    python run_sweep.py --circuits weak_random --n-seeds 2 --n-qubits 8
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Tracks the currently-running subprocess's process group so a signal
# handler (Ctrl+C, or an external `kill`/`pkill run_sweep.py`) can clean it
# up. Needed because pkill-by-name on run_sweep.py/run_experiment.py's own
# cmdline does NOT reach joblib/loky worker processes -- their cmdline is
# "python -m joblib.externals.loky.backend.popen_loky_posix", with no
# reference to run_experiment.py, so they can otherwise survive as orphans
# after the parent is killed by name.
_current_pgid = None


def _kill_process_group(pgid, term_timeout=5):
    """Terminate every process in pgid, escalating to SIGKILL if needed."""
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + term_timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)  # signal 0: just probes whether the group is still alive
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _handle_terminate(signum, frame):
    _kill_process_group(_current_pgid)
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_terminate)
signal.signal(signal.SIGINT, _handle_terminate)

CIRCUITS = ["weak_random", "qaoa_maxcut", "w_state", "qft", "hw_efficient_ansatz"]

# Per-circuit qubit sizes. Capped at 8 -- larger tiers (12/16/20q) repeatedly hit
# fidelity/injection-stage performance walls (hangs, multi-hour runs) on genuinely
# entangled circuit families. Scaling past 8 qubits is deferred; see logs.txt's
# "SCALING — DEFERRED" section. Pass --n-qubits to override for a one-off run.
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

# Above this many qubits, sa_injection's default sa_iters=3000 costs real wall-clock per
# iteration via the approximate fidelity backend (measured: ~21.6 min at n=12 vs. ~9.1 min
# for stochastic, which is also the actual floor -- the always-on greedy fidelity_driven_injection
# pass costs the same either way, and reducing sa_iters to 150 made sa's total time converge to
# stochastic's anyway). Combined with the 8-qubit finding that the greedy pass wins 48/50 runs
# regardless of injection method, sa isn't worth its extra cost above this size -- dropped
# from the grid there rather than tuned down further.
LARGE_N_THRESHOLD = 10


def build_grid(circuits, n_qubits_override, injection_methods, n_seeds):
    for circuit in circuits:
        sizes = n_qubits_override if n_qubits_override is not None else CIRCUIT_QUBIT_SIZES[circuit]
        for n_qubits in sizes:
            methods = [m for m in injection_methods if n_qubits <= LARGE_N_THRESHOLD or m != "sa"]
            for injection_method in methods:
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
    global _current_pgid
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
        # start_new_session=True makes this subprocess (and every worker it
        # spawns, e.g. joblib/loky) its own process group, so it can be torn
        # down atomically -- pkill-by-name on the parent alone misses loky's
        # own worker processes (see _current_pgid comment above).
        proc = subprocess.Popen(cmd, start_new_session=True)
        _current_pgid = os.getpgid(proc.pid)
        try:
            returncode = proc.wait()
        finally:
            _kill_process_group(_current_pgid)
            _current_pgid = None
        if returncode != 0:
            print(f"[fail] {run_id} (exit code {returncode})", file=sys.stderr)
            n_failed += 1
        else:
            n_run += 1

    print(f"\nSweep complete: {n_run} run, {n_skipped} skipped, {n_failed} failed "
          f"(of {len(grid)} total).")
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
