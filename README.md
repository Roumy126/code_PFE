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

# Run a resumable multi-seed sweep (edit N_QUBITS/INJECTION_METHODS/N_SEEDS at the top of the
# file, or override via flags). Re-running the same command skips any run whose
# runs/<run_id>/metrics.json already exists, so a laptop-scale sweep can be stopped and
# resumed without losing progress or duplicating work.
python run_sweep.py --dry-run              # preview the grid without running anything
python run_sweep.py --n-seeds 5 --n-qubits 8 12 20

# Aggregate every completed run under runs/ into one table: runs/results_master.csv
python aggregate_results.py
```

Each run's `metrics.json` includes final fidelity/depth/cost, per-block MOO indicators (hypervolume, spread, spacing), and a `stage_timings_s` breakdown (partitioning / block_optimization / injection / compression) for tuning population size vs. generations on limited compute.

## Project Structure
- `AG_mono/`: Mono-objective implementations.
- `NSGA-II/`: Multi-objective implementations.
- `Final_test/` & `final_test_AG/`: Validation scripts and performance tests.
- `m1*/`, `m2*/`: Test modules for different types of circuit blocks.
- `M1_finale/`: Canonical pipeline (partitioning, block-local NSGA-II, inter-block injection, compression) — see "Running experiments" above.
- `run_experiment.py`, `run_sweep.py`, `aggregate_results.py`: CLI tooling for single runs, resumable multi-seed sweeps, and results aggregation.
- `runs/`: Structured output of experiment runs (gitignored) — one directory per run, aggregated by `aggregate_results.py`.
- `tests/`: Pytest smoke test for the pipeline (no broader test suite exists).