# Quantum Circuit Optimization with a Genetic Algorithm

## Description
This project implements a genetic algorithm for optimizing and approximating quantum circuits. The main goal is to use evolutionary techniques to generate quantum circuits that best approximate a target unitary matrix, or to optimize the parameters of existing circuits.

The project uses **Qiskit** to build and simulate quantum circuits, along with scientific computing tools such as **NumPy** and **SciPy**.

## Installation

Create and activate a virtual environment, then install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If a `.venv/` already exists (e.g. it came with the repo), just activate it:

```bash
source .venv/bin/activate
```

Once active, your prompt shows `(.venv)` and plain `python`/`pip`/`jupyter`/`pytest` resolve to this environment. To leave it: `deactivate`.

## Usage

The project is mainly organized around Jupyter notebooks. To explore and run the code:

1. Activate the virtual environment (see Installation above), then launch JupyterLab:
   ```bash
   source .venv/bin/activate
   jupyter lab
   ```
2. Open one of the main notebooks, for example:
   - `AG_mono/code_ag.ipynb`: Mono-objective genetic algorithm.
   - `NSGA-II/AG_multi_objectifs_VF.ipynb`: Multi-objective genetic algorithm (NSGA-II).
   - `final_test_AG/code_travaille copy 3.ipynb`: Final tests of the optimization algorithm.

## Running experiments

`M1_finale/` is the canonical, most-maintained pipeline (partitioning -> per-block NSGA-II -> inter-block injection -> compression). It's importable as a module and has a CLI layer on top for reproducible, structured experiment runs instead of ad-hoc notebook execution:

```bash
source .venv/bin/activate

# Smoke-test the pipeline (~10s, tiny 6-qubit circuit) before a long run
pytest tests/test_pipeline_smoke.py

# Run a single configuration; writes runs/<run_id>/{config.json,metrics.json,final_circuit.qpy}
python run_experiment.py --n-qubits 12 --seed 0 --generations 100 --pop-size 100

# --circuit selects the target circuit generator (default: weak_random):
#   weak_random           random weakly-connected circuit (any size, incl. 14/64-qubit cases)
#   qaoa_maxcut           QAOA on a ring+chord MaxCut graph (--qaoa-p, --qaoa-gammas, --qaoa-betas)
#   w_state               W-state preparation (ry+cx ladder)
#   qft                   Quantum Fourier Transform
#   hw_efficient_ansatz   generic hardware-efficient ansatz (H + ry/rz + linear cx; --ansatz-reps)
python run_experiment.py --circuit qaoa_maxcut --n-qubits 8 --qaoa-p 2 --generations 100 --pop-size 100

# --block-algorithm selects the per-block optimizer (default: nsga2):
#   nsga2      DEAP NSGA-II (3 objectives: fidelity, depth, cost)
#   smsemoa    hypervolume-driven SMS-EMOA (larger/more diverse Pareto fronts, similar
#              wall-clock; doesn't strictly beat nsga2 on peak fidelity, see logs.txt)
# --mutation-scheme (point / swap_add / swap_add_delete) and --hybrid-las (runs a
# finite-difference local angle search on the GA's winner) are independent axes on
# top of --block-algorithm; which combination wins depends on the pairing (see
# logs.txt's "RESEARCH ANGLE CHOSEN AND IMPLEMENTED" for the ablation results).
python run_experiment.py --circuit qft --n-qubits 8 --block-algorithm smsemoa \
    --mutation-scheme swap_add_delete --hybrid-las --generations 100 --pop-size 100

# Run a resumable multi-seed sweep over the fixed benchmark set (5 circuit families x
# CIRCUIT_QUBIT_SIZES x injection method x block algorithm x mutation scheme x
# hybrid-LAS option x seed; edit the constants at the top of the file, or override via
# flags). Re-running the same command skips any run whose runs/<run_id>/metrics.json
# already exists, so a laptop-scale sweep can be stopped and resumed without losing
# progress or duplicating work.
python run_sweep.py --dry-run              # preview the grid without running anything
python run_sweep.py --circuits weak_random qft --n-seeds 5

# Aggregate every completed run under runs/ into one table: runs/results_master.csv
python aggregate_results.py
```

Each run's `metrics.json` includes final fidelity/depth/cost, per-block MOO indicators (hypervolume, spread, spacing), a `stage_timings_s` breakdown (partitioning / block_optimization / injection / compression), which injection path actually produced the final circuit (`injection_path_used`, `fidelity_after_injection_method`, `fidelity_after_fidelity_driven_greedy` — the pipeline's greedy fallback pass often wins over the requested `--injection-method`, so these fields tell you which one you actually got), which fidelity backend was used at the reporting level (`fidelity_backend`: `"exact"`, `"echo_test_mc"`, or `"statevector_mc"`, see below), and which tier `fidelity_driven_injection`'s per-trial loop used (`fidelity_driven_tier`: `"exact"` / `"statevector"` / `"echo_test_mc"`).

**Fidelity at scale:** the injection stage's fidelity checks run on the full circuit, which used to mean an exact dense-`Operator` computation that became intractable past ~10-12 qubits (a 12-qubit run once took over 1h45m before being killed). Circuits above `--fidelity-exact-threshold` (default 13) now use an approximate Monte-Carlo estimate instead — by default the fidelity-echo estimator (`--fidelity-echo-samples`, default 8; `--fidelity-echo-shots`, default 128), a proxy metric over random product states rather than a full-Hilbert-space fidelity. This replaced an older SWAP-test formulation on 2026-08-27: same target quantity, but computed on `n` qubits instead of `2n+1` with no extra CSWAP-induced entanglement between two coupled registers, which used to make it exponentially expensive specifically for genuinely entangled circuit families (QAOA/QFT/W-state/hw-efficient-ansatz) regardless of qubit count. Benchmarked post-fix (isolated per-call cost only): `w_state`/`hw_efficient_ansatz` stay cheap (~0.05s/call) up to at least n=32; `qft` scales gently (~20s/call at n=32); `qaoa_maxcut` is the outlier and still gets expensive past ~n=16 (its ring+chord graph structure genuinely gets more entangled as n grows — not a software artifact). **Caveat found later (2026-08-28, see below): this per-call cost benchmark never checked whether the reported fidelity is actually above the estimator's resolution floor, nor exercised a real run's reinjection-gate count — both turned out to matter more than raw call cost for 3 of these 4 families.**

For `qaoa_maxcut`-like cases, pass `--fidelity-approximate-backend statevector` (default `mps`): an exact-per-sample, no-shot-noise estimator that's ~150x+ faster than the MPS-backed echo estimator for highly-entangled circuits at n=16, but 250-400x *slower* for low-entanglement families like `w_state`/`hw_efficient_ansatz` at n=24 — there's no auto-detection, pick the backend per circuit family. With it, a previously-timing-out n=20 `qaoa_maxcut` run completes in ~38s. `fidelity_driven_injection`'s own per-trial loop gets a similar exact-statevector fast path automatically up to `--fidelity-driven-statevector-threshold` (default 24) regardless of the reporting-level backend choice.

`--fidelity-driven-max-trials` (default 300, was hardcoded/not tunable before 2026-08-27) controls `fidelity_driven_injection`'s greedy pass, which runs unconditionally regardless of `--injection-method` and is now the dominant per-run cost for entangled families above `--injection-fidelity-exact-threshold` — lower it for large-n runs on entangled families if not also using the statevector fast path above. `--sa-iters` is likewise not auto-reduced above `--fidelity-exact-threshold` — lower it manually if using `--injection-method sa` (though `sa` now has its own closed-form energy fast path and is typically the *cheaper* of the two injection methods on measured wall-clock, not the more expensive one). See `logs.txt`'s "SCALING" entries (search for that word) for the full history, including the specific fixes and their verification.

**Practical qubit ceiling (2026-08-28, validated by real production pilot runs per family, not cost benchmarks alone):** raising `run_sweep.py`'s grid to each family's isolated-cost-benchmark "ceiling" above and actually inspecting the results found 3 of 4 non-`weak_random` families silently returning noise, not signal — `w_state`@20, `hw_efficient_ansatz`@20, and `qaoa_maxcut`@16 all completed quickly under plain defaults, but `fidelity_final` came back at or below the echo estimator's exact resolution floor (`1/(samples×shots)` = `1/1024` at defaults — `0.000977` or exactly `0.0`, indistinguishable from a genuinely-zero fidelity). `qft`@20 didn't even complete (killed after 47+ minutes): its Fourier-transform structure is inherently globally-entangling, leaving far more cross-block gates to reinject than the other families (180, vs. 3-20), which hit the same MPS/entanglement wall `qaoa_maxcut` needed the statevector backend for. Fixed per family and validated with real reruns: `qaoa_maxcut`/`qft` now use `--fidelity-approximate-backend statevector` (fixes both the wall-clock and, for `qaoa_maxcut`, the floor); `w_state`/`hw_efficient_ansatz` instead use `--fidelity-echo-samples 32 --fidelity-echo-shots 2048` (lowers the floor to `1/65536`; their low entanglement means MPS was never the actual problem there). Results: `qaoa_maxcut`@16 91.7s (was 236-256s) with `fidelity_final=2.2e-05` (was `0.0`); `qft`@20 354.0s (was a 47+ min kill); `w_state`@20 226.3s (was ~60-70s) with `fidelity_final=9.2e-05` (was `0.000977`, i.e. still floor); `hw_efficient_ansatz`@20 204.6s (was ~50-60s) with `fidelity_final=3.2e-04` (was `0.0`). `run_sweep.py` now applies all of this automatically per circuit (`CIRCUIT_FIDELITY_SETTINGS`, gated to apply only above `--fidelity-exact-threshold` so existing n=8/12 data and run_ids are untouched) — a plain `python run_sweep.py` sweeps `weak_random` to n=20, `qaoa_maxcut` to n=16, and the other three to n=20, each with the right fidelity settings already applied. None of the four has been tried past this newly-validated size (`qft` n=32, `qaoa_maxcut` n=20) — extending further needs its own pilot, not an assumption that the fix generalizes; `qaoa_maxcut` specifically becomes memory-bound under statevector past roughly n=24-28 regardless.

**Parallelism:** per-block NSGA-II fitness evaluation (`optimise_block_nsga2`) always uses `joblib.Parallel(-1)` — all available cores, with no CLI knob to cap it. This is intentional: the runtime hardware's own thermal safety mechanisms handle CPU protection, so this software does not need to throttle its own usage.

## Project Structure
- `AG_mono/`: Mono-objective implementations.
- `NSGA-II/`: Multi-objective implementation using DEAP's built-in NSGA-II.
- `NSGA-III/`: Independently hand-rolled NSGA-III (Deb & Jain 2014 reference-point method, not DEAP's `selNSGA3`) — exists only on this branch (`ULBS_qiea`).
- `Final_test/` & `final_test_AG/`: Validation scripts and performance tests.
- `m1*/`, `m2*/`: Test modules for different types of circuit blocks.
- `M1_finale/`: Canonical pipeline (partitioning, block-local NSGA-II/SMS-EMOA, inter-block injection, compression) — see "Running experiments" above.
- `run_experiment.py`, `run_sweep.py`, `aggregate_results.py`: CLI tooling for single runs, resumable multi-seed sweeps, and results aggregation.
- `runs/`: Structured output of experiment runs (gitignored) — one directory per run, aggregated by `aggregate_results.py`.
- `tests/`: Pytest smoke test for the pipeline (no broader test suite exists).
- `logs.txt`: Full narrative project log (scaling history, research findings, ablation results). `status.txt`: latest supervisor-facing status snapshot. `moo.txt`: web-research notes on the wider Pareto-dominance MOO algorithm landscape.