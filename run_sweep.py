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

# Per-circuit qubit sizes. Still capped at 8 here as a conservative baseline default,
# NOT a hard technical limit any more -- kept unchanged so a plain `python run_sweep.py`
# stays behavior-unchanged (this project's convention: sweep defaults only change when
# explicitly decided, not silently). Since the original "SCALING — DEFERRED" 8-qubit
# decision, real pilot sweeps (via --n-qubits overrides, not by editing this dict) have
# established n=12/13 as practical for all 5 families (2026-08-26 injection-stage fixes),
# and n=16-32 as practical for 4 of the 5 (weak_random/w_state/hw_efficient_ansatz/qft) --
# qaoa_maxcut remains genuinely expensive past ~n=16 even after the 2026-08-27 fidelity-
# backend fix (see logs.txt's "SCALING — ECHO-TEST FIDELITY BACKEND REPLACES SWAP TEST"),
# and needs --fidelity-driven-max-trials / --fidelity-echo-samples / --fidelity-echo-shots
# lowered to stay tractable at those sizes. See logs.txt's "SCALING" entries for the full
# history. Pass --n-qubits to override for a one-off run.
CIRCUIT_QUBIT_SIZES = {
    "weak_random": [8],
    "qaoa_maxcut": [8],
    "w_state": [8],
    "qft": [8],
    "hw_efficient_ansatz": [8],
}

INJECTION_METHODS = ["stochastic", "sa"]

# Default to the current baseline choice for each new axis, so a plain `python
# run_sweep.py` invocation stays behavior-unchanged. Override one axis at a time via
# the CLI flags below for the staged algorithm/mutation/hybrid-LAS comparisons (see
# logs.txt) instead of sweeping the full factorial, which would be 600+ runs.
BLOCK_ALGORITHMS = ["nsga2"]
MUTATION_SCHEMES = ["point"]
HYBRID_LAS_OPTIONS = [False]

N_SEEDS = 5  # roadmap ultimately wants 10-20 seeds per config for review

GENERATIONS = 100
POP_SIZE = 100

# NOTE: sa was previously dropped from the grid above 10 qubits (sa_injection's default
# sa_iters=3000 cost real wall-clock per iteration via the approximate fidelity backend,
# ~21.6 min at n=12 vs. ~9.1 min for stochastic). That measurement predates the 2026-08-26
# sa_injection fix (see logs.txt "SCALING - SA INJECTION FAST PATH FOUND AND FIXED"):
# sa_injection's energy check always compares a candidate to its own unchanged base, so it
# reduces to an exact closed-form computation unconditionally, at any n -- no longer costs
# more than stochastic at any qubit count. The per-size exclusion was removed accordingly.


def build_grid(circuits, n_qubits_override, injection_methods, n_seeds,
                block_algorithms=BLOCK_ALGORITHMS, mutation_schemes=MUTATION_SCHEMES,
                hybrid_las_options=HYBRID_LAS_OPTIONS,
                fidelity_exact_threshold=None, injection_fidelity_exact_threshold=None):
    for circuit in circuits:
        sizes = n_qubits_override if n_qubits_override is not None else CIRCUIT_QUBIT_SIZES[circuit]
        for n_qubits in sizes:
            for injection_method in injection_methods:
                for block_algorithm in block_algorithms:
                    for mutation_scheme in mutation_schemes:
                        for hybrid_las in hybrid_las_options:
                            for seed in range(n_seeds):
                                yield {
                                    "circuit": circuit,
                                    "n_qubits": n_qubits,
                                    "injection_method": injection_method,
                                    "block_algorithm": block_algorithm,
                                    "mutation_scheme": mutation_scheme,
                                    "hybrid_las": hybrid_las,
                                    "generations": GENERATIONS,
                                    "pop_size": POP_SIZE,
                                    "seed": seed,
                                    "fidelity_exact_threshold": fidelity_exact_threshold,
                                    "injection_fidelity_exact_threshold": injection_fidelity_exact_threshold,
                                }


def run_id_for(cfg: dict) -> str:
    # Encodes every varied parameter so a change to generations/pop_size (or
    # any other swept knob) can't silently collide with a stale prior run.
    run_id = (
        f"{cfg['circuit']}_{cfg['n_qubits']}q_{cfg['injection_method']}"
        f"_{cfg['block_algorithm']}_{cfg['mutation_scheme']}_las{int(cfg['hybrid_las'])}"
        f"_g{cfg['generations']}_p{cfg['pop_size']}_seed{cfg['seed']}"
    )
    # Only appended when explicitly overridden, so existing run_ids (and the runs/ dataset
    # already built on them) are unaffected by this option's addition -- the CLI default
    # (None) means "let run_experiment.py use its own default", same as before this flag
    # existed.
    fet = cfg.get("fidelity_exact_threshold")
    ifet = cfg.get("injection_fidelity_exact_threshold")
    if fet is not None:
        run_id += f"_fet{fet}"
    if ifet is not None:
        run_id += f"_ifet{ifet}"
    return run_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuits", choices=CIRCUITS, nargs="+", default=CIRCUITS)
    p.add_argument("--n-qubits", type=int, nargs="+", default=None,
                    help="Override qubit sizes for every selected circuit "
                         "(default: each circuit's own CIRCUIT_QUBIT_SIZES entry).")
    p.add_argument("--injection-methods", choices=["sa", "stochastic"], nargs="+",
                    default=INJECTION_METHODS)
    p.add_argument("--block-algorithms", choices=["nsga2", "smsemoa"], nargs="+",
                    default=BLOCK_ALGORITHMS)
    p.add_argument("--mutation-schemes", choices=["point", "swap_add", "swap_add_delete"],
                    nargs="+", default=MUTATION_SCHEMES)
    p.add_argument("--hybrid-las-options", type=int, choices=[0, 1], nargs="+",
                    default=[int(v) for v in HYBRID_LAS_OPTIONS],
                    help="0=pure GA, 1=hybrid GA+LAS. Pass both to sweep the axis.")
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--fidelity-exact-threshold", type=int, default=None,
                    help="Passed through to run_experiment.py's flag of the same name "
                         "(reporting-level fidelity calls). Default: let run_experiment.py "
                         "use its own default (13). Needed to sweep circuits above that size "
                         "without falling back to the untouched approximate SWAP-test path.")
    p.add_argument("--injection-fidelity-exact-threshold", type=int, default=None,
                    help="Passed through to run_experiment.py's flag of the same name "
                         "(fidelity_driven_injection's per-trial loop, which always runs "
                         "regardless of --injection-methods). Default: let run_experiment.py "
                         "use its own default (12). E.g. pass 13 to sweep 13-qubit circuits "
                         "on the fixed, exact fast path instead of the approximate one.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--dry-run", action="store_true", help="Print the grid and exit.")
    return p.parse_args(argv)


def main(argv=None):
    global _current_pgid
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    grid = list(build_grid(
        args.circuits, args.n_qubits, args.injection_methods, args.n_seeds,
        block_algorithms=args.block_algorithms,
        mutation_schemes=args.mutation_schemes,
        hybrid_las_options=[bool(v) for v in args.hybrid_las_options],
        fidelity_exact_threshold=args.fidelity_exact_threshold,
        injection_fidelity_exact_threshold=args.injection_fidelity_exact_threshold,
    ))

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
            "--block-algorithm", cfg["block_algorithm"],
            "--mutation-scheme", cfg["mutation_scheme"],
            "--generations", str(cfg["generations"]),
            "--pop-size", str(cfg["pop_size"]),
            "--seed", str(cfg["seed"]),
        ]
        if cfg["hybrid_las"]:
            cmd.append("--hybrid-las")
        if cfg.get("fidelity_exact_threshold") is not None:
            cmd += ["--fidelity-exact-threshold", str(cfg["fidelity_exact_threshold"])]
        if cfg.get("injection_fidelity_exact_threshold") is not None:
            cmd += ["--injection-fidelity-exact-threshold", str(cfg["injection_fidelity_exact_threshold"])]
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
