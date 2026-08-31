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
sizes spanning the exact-fidelity tier (8), a mid-size approximate-tier point
(12), and each family's own pilot-validated ceiling (16 for qaoa_maxcut, 20
for the other four) -- see CIRCUIT_QUBIT_SIZES
and CIRCUIT_FIDELITY_SETTINGS below for what each family's ceiling needed
(a different fidelity-backend/sample/shot setting per family, not a single
global default) and why -- edit CIRCUITS / CIRCUIT_QUBIT_SIZES / INJECTION_
METHODS / N_SEEDS directly, or use the CLI overrides.

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

# Per-circuit qubit sizes. weak_random reaches n=20 with run_experiment.py's plain
# defaults (fidelity_final=0.628, ~150-170s -- a real pilot-confirmed signal). The
# other four families' n=20 (n=16 for qaoa_maxcut) needed CIRCUIT_FIDELITY_SETTINGS
# below FIRST (2026-08-28) -- at plain defaults every one of them either hit the
# echo-test MC estimator's shot-noise floor (w_state: 0.000977 = 1/(8 samples x 128
# shots) exactly; hw_efficient_ansatz/qaoa_maxcut: 0.0, no lucky hit) or, for qft,
# never finished at all (killed after 47+ min -- its Fourier-transform structure is
# inherently globally-entangling, so partitioning it left 180 cross-block gates to
# reinject vs. 3-20 for the others, and its reporting-level fidelity checks hit the
# same MPS/entanglement wall qaoa_maxcut needed the statevector backend for). None
# of this was caught by wall-clock cost alone -- see CIRCUIT_FIDELITY_SETTINGS for
# the fix each family needed and the validated numbers after it. Pass --n-qubits to
# override for a one-off run at a different size.
CIRCUIT_QUBIT_SIZES = {
    "weak_random": [8, 12, 20],
    "qaoa_maxcut": [8, 12, 16],
    "w_state": [8, 12, 20],
    "qft": [8, 12, 20],
    "hw_efficient_ansatz": [8, 12, 20],
}

# Per-circuit overrides for run_experiment.py's fidelity-estimation flags, layered
# in ON TOP OF its own defaults (fidelity_approximate_backend="mps", fidelity_echo_
# samples=8, fidelity_echo_shots=128) -- keys omitted here just mean "use the
# default". Only matters above fidelity_exact_threshold (13); a no-op for the n=8/12
# sizes every family also sweeps, so it can't change already-collected data there.
#
# Each entry was chosen from a real pilot run at that family's new qubit-size
# ceiling (2026-08-28), not assumed from logs.txt's isolated fidelity-cost
# benchmarks -- see CIRCUIT_QUBIT_SIZES's comment for what went wrong without this:
#   - qaoa_maxcut (n=16), qft (n=20): entanglement, not shot noise, was the problem
#     (qft's 180 cross-block gates especially) -- the statevector backend fixed
#     both the wall-clock AND, for qaoa_maxcut, the floor-clipping, since it's exact
#     per-sample (no shot noise) as well as entanglement-agnostic. Validated: qaoa_
#     maxcut@16 now 91.7s (was 236-256s under mps) with fidelity_final=2.2e-05 (was
#     0.0); qft@20 now 354.0s (was a 47+ min kill) with fidelity_final=6.7e-05.
#   - w_state, hw_efficient_ansatz (n=20): these stay cheap under mps (their low
#     entanglement is exactly what it's good at, per logs.txt's SCALING section) --
#     the problem here was purely shot-noise resolution, so raising samples/shots
#     (8->32, 128->2048, lowering the floor from 1/1024 to 1/65536) was the cheaper
#     fix vs. switching backends. Validated: w_state@20 now 226.3s (was ~60-70s)
#     with fidelity_final=9.2e-05 (was 0.000977, i.e. still floor); hw_efficient_
#     ansatz@20 now 204.6s (was ~50-60s) with fidelity_final=3.2e-04 (was 0.0).
# Neither fix has been tried at qft/qaoa_maxcut's OWN higher untested sizes (e.g.
# qft n=32, qaoa_maxcut n=20) -- extending CIRCUIT_QUBIT_SIZES further needs its own
# pilot, the same way this round did.
CIRCUIT_FIDELITY_SETTINGS = {
    "weak_random": {},
    "qaoa_maxcut": {"fidelity_approximate_backend": "statevector"},
    "w_state": {"fidelity_echo_samples": 32, "fidelity_echo_shots": 2048},
    "qft": {"fidelity_approximate_backend": "statevector"},
    "hw_efficient_ansatz": {"fidelity_echo_samples": 32, "fidelity_echo_shots": 2048},
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
            # Only apply CIRCUIT_FIDELITY_SETTINGS above the exact-fidelity tier --
            # these flags are unused no-ops at or below it (run_experiment.py's own
            # default fidelity_exact_threshold=13, or the CLI override if set), so
            # applying them at n=8/12 would only pollute run_id_for's suffix and
            # create a spurious duplicate of already-collected data under a new
            # run_id, not change any actual run behavior.
            exact_threshold = fidelity_exact_threshold if fidelity_exact_threshold is not None else 13
            fidelity_settings = CIRCUIT_FIDELITY_SETTINGS.get(circuit, {}) if n_qubits > exact_threshold else {}
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
                                    "fidelity_approximate_backend": fidelity_settings.get("fidelity_approximate_backend"),
                                    "fidelity_echo_samples": fidelity_settings.get("fidelity_echo_samples"),
                                    "fidelity_echo_shots": fidelity_settings.get("fidelity_echo_shots"),
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
    # Same "only appended when non-default" rule as fet/ifet above -- keeps existing
    # run_ids (n=8/12, weak_random's n=20) untouched, since CIRCUIT_FIDELITY_SETTINGS
    # only sets these for the newly-piloted large-n configs (see its comment).
    backend = cfg.get("fidelity_approximate_backend")
    samples = cfg.get("fidelity_echo_samples")
    shots = cfg.get("fidelity_echo_shots")
    if backend is not None and backend != "mps":
        run_id += f"_{backend}"
    if samples is not None and samples != 8:
        run_id += f"_es{samples}"
    if shots is not None and shots != 128:
        run_id += f"_sh{shots}"
    return run_id


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--circuits", choices=CIRCUITS, nargs="+", default=CIRCUITS)
    p.add_argument("--n-qubits", type=int, nargs="+", default=None,
                    help="Override qubit sizes for every selected circuit "
                         "(default: each circuit's own CIRCUIT_QUBIT_SIZES entry).")
    p.add_argument("--injection-methods", choices=["sa", "stochastic"], nargs="+",
                    default=INJECTION_METHODS)
    p.add_argument("--block-algorithms", choices=["nsga2", "smsemoa", "nsga3"], nargs="+",
                    default=BLOCK_ALGORITHMS,
                    help="Must stay in sync with M1_finale.final_m1_script.BLOCK_OPTIMIZERS' "
                         "keys -- this script deliberately doesn't import that module (it only "
                         "launches run_experiment.py as a subprocess), so a new algorithm added "
                         "there needs its name added here too.")
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
                         "use its own default (7, lowered from 12 on 2026-08-31 -- see logs.txt "
                         "'SCALING -- INJECTION_FIDELITY_EXACT_THRESHOLD RE-BENCHMARKED'). E.g. pass "
                         "13 to force the exact fast path up to 13-qubit circuits instead of the "
                         "(now cheaper) statevector one.")
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
        if cfg.get("fidelity_approximate_backend") is not None:
            cmd += ["--fidelity-approximate-backend", cfg["fidelity_approximate_backend"]]
        if cfg.get("fidelity_echo_samples") is not None:
            cmd += ["--fidelity-echo-samples", str(cfg["fidelity_echo_samples"])]
        if cfg.get("fidelity_echo_shots") is not None:
            cmd += ["--fidelity-echo-shots", str(cfg["fidelity_echo_shots"])]
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
