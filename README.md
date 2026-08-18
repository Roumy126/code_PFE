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

## Project Structure
- `AG_mono/`: Mono-objective implementations.
- `NSGA-II/`: Multi-objective implementations.
- `Final_test/` & `final_test_AG/`: Validation scripts and performance tests.
- `m1*/`, `m2*/`: Test modules for different types of circuit blocks.