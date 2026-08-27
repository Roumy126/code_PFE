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

# Run a resumable multi-seed sweep over the fixed benchmark set (5 circuit families x
# CIRCUIT_QUBIT_SIZES x injection method x seed; edit those constants at the top of the
# file, or override via flags). Re-running the same command skips any run whose
# runs/<run_id>/metrics.json already exists, so a laptop-scale sweep can be stopped and
# resumed without losing progress or duplicating work.
python run_sweep.py --dry-run              # preview the grid without running anything
python run_sweep.py --circuits weak_random qft --n-seeds 5

# Aggregate every completed run under runs/ into one table: runs/results_master.csv
python aggregate_results.py
```

Each run's `metrics.json` includes final fidelity/depth/cost, per-block MOO indicators (hypervolume, spread, spacing), a `stage_timings_s` breakdown (partitioning / block_optimization / injection / compression), which injection path actually produced the final circuit (`injection_path_used`, `fidelity_after_injection_method`, `fidelity_after_fidelity_driven_greedy` — the pipeline's greedy fallback pass often wins over the requested `--injection-method`, so these fields tell you which one you actually got), and which fidelity backend was used (`fidelity_backend`: `"exact"` or `"echo_test_mc"`, see below).

**Fidelity at scale:** the injection stage's fidelity checks run on the full circuit, which used to mean an exact dense-`Operator` computation that became intractable past ~10-12 qubits (a 12-qubit run once took over 1h45m before being killed). Circuits above `--fidelity-exact-threshold` (default 13) now use an approximate Monte-Carlo fidelity-echo estimate instead (`--fidelity-echo-samples`, default 8; `--fidelity-echo-shots`, default 128) — a proxy metric over random product states, not a full-Hilbert-space fidelity. This replaced an older SWAP-test formulation on 2026-08-27: same target quantity, but computed on `n` qubits instead of `2n+1` with no extra CSWAP-induced entanglement between two coupled registers, which used to make it exponentially expensive specifically for genuinely entangled circuit families (QAOA/QFT/W-state/hw-efficient-ansatz) regardless of qubit count. Benchmarked post-fix: `w_state`/`hw_efficient_ansatz` stay cheap (~0.05s/call) up to at least n=32; `qft` scales gently (~20s/call at n=32); `qaoa_maxcut` is the outlier and still gets expensive past ~n=16 (its ring+chord graph structure genuinely gets more entangled as n grows — not a software artifact). `--fidelity-driven-max-trials` (default 300, was hardcoded/not tunable before 2026-08-27) controls `fidelity_driven_injection`'s greedy pass, which runs unconditionally regardless of `--injection-method` and is now the dominant per-run cost for entangled families above `--injection-fidelity-exact-threshold` (confirmed: a real n=16 `qaoa_maxcut` run did not finish within 280s at the default 300 trials; finished in ~204s once both this and `--fidelity-echo-samples`/`--fidelity-echo-shots` were lowered) — lower these for large-n runs on entangled families. `--sa-iters` is likewise not auto-reduced above `--fidelity-exact-threshold` — lower it manually if using `--injection-method sa`. See `logs.txt`'s "SCALING" entries (search for that word) for the full history, including the specific fixes and their verification.

**Parallelism:** per-block NSGA-II fitness evaluation (`optimise_block_nsga2`) always uses `joblib.Parallel(-1)` — all available cores, with no CLI knob to cap it. This is intentional: the runtime hardware's own thermal safety mechanisms handle CPU protection, so this software does not need to throttle its own usage.

## Project Structure
- `AG_mono/`: Mono-objective implementations.
- `NSGA-II/`: Multi-objective implementations.
- `Final_test/` & `final_test_AG/`: Validation scripts and performance tests.
- `m1*/`, `m2*/`: Test modules for different types of circuit blocks.
- `M1_finale/`: Canonical pipeline (partitioning, block-local NSGA-II, inter-block injection, compression) — see "Running experiments" above.
- `run_experiment.py`, `run_sweep.py`, `aggregate_results.py`: CLI tooling for single runs, resumable multi-seed sweeps, and results aggregation.
- `runs/`: Structured output of experiment runs (gitignored) — one directory per run, aggregated by `aggregate_results.py`.
- `tests/`: Pytest smoke test for the pipeline (no broader test suite exists).