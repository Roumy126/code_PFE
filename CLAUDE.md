# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Academic research project (French PFE — end-of-studies project) on **optimizing quantum circuits with genetic algorithms**. The goal is to take a target quantum circuit (or unitary), partition it, and evolve a cheaper/shallower circuit that still reproduces the target's behavior with high fidelity, using **Qiskit** for circuit simulation and **DEAP** for evolutionary computation (NSGA-II / NSGA-III style multi-objective optimization).

The codebase is a research sandbox, not a packaged library: it is organized as a progression of **standalone Jupyter notebooks and scripts**, each an iteration of the same core pipeline (mono-objective → multi-objective → block-decomposed/scalable). There is no shared package, no test suite, and no build step — each notebook/script is self-contained and re-defines the functions it needs.

## Setup and running code

```bash
pip install -r requirements.txt
```

`requirements.txt` at the repo root is incomplete for several notebooks/scripts — it's missing `deap`, `pandas`, and `pymoo`, which are imported throughout (e.g. `M1_finale/`, most `Final_test/*` notebooks). When working in those areas, also `pip install deap pandas pymoo`. `M1_finale/requirements_m1.txt` has the fuller list for that subproject (adds `nbconvert`, `pylatexenc`).

There is no CLI entry point for the project as a whole. Work happens by opening a notebook:

```bash
jupyter notebook
```

or running one of the few extracted `.py` scripts directly, e.g.:

```bash
python M1_finale/final_m1_script.py     # full pipeline on a generated 30-qubit circuit
python M1_finale/run_m1_test.py         # quick pipeline smoke test
python M1_finale/circuit_64qubits.py    # generates a 64-qubit weakly-connected circuit and runs the pipeline on it
```

No lint, format, or test commands exist in this repo — there is no test suite. Do not invent one unless asked.

Committed `.venv/`, `mon_env/`, `envq/` directories are local virtualenvs (already gitignored alongside `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`) — never edit files inside them.

## Repository structure — reading order

The directories represent **successive iterations** of the same idea, roughly in this order of sophistication:

1. **`AG_mono/`** — mono-objective genetic algorithm: evolve a chromosome (gate sequence) that approximates a target unitary, optimizing fidelity only.
2. **`NSGA-II/`**, **`NSGA-III/`** — multi-objective versions (fidelity, depth, gate cost) using DEAP's NSGA-II/NSGA-III selection.
3. **`m1/`, `m1_bloc_indep/`, `M1_indicateurs_version/`, `M1_finale/`** — "M1" line of work: adds circuit **partitioning** into qubit blocks (so NSGA-II runs per-block instead of on the whole circuit) plus quality indicators (hypervolume, IGD, spread, epsilon — see below). `M1_finale/` is the most complete/cleaned-up version and has extracted `.py` scripts alongside its notebook — read `M1_finale/final_m1_script.py` first to understand the full pipeline without wading through notebook cell markers.
4. **`m2_bloc_chevauche/`** — "block overlap" variant: blocks are allowed to share qubits at interfaces rather than being a strict partition.
5. **`Final_test/`, `final_test_AG/`, `test/`** — later validation/experiment notebooks applying the pipeline to specific target circuits (QAOA, MaxCut, W-state, QFT, VQE, 14/64-qubit cases, etc.), one subdirectory per experiment. These are throwaway experiment copies, not a distinct architecture — treat each as a snapshot of a notebook run against a specific test case rather than as unique code to maintain.

When asked to modify "the pipeline" without a specified location, assume `M1_finale/final_m1_script.py` — it is the canonical, most-maintained version. Notebooks with names like `code_travaille copy 3.ipynb` or `test3D(1) copy.ipynb` are duplicated experiment snapshots, not different components — don't assume divergent copies need to be kept in sync; check with the user before propagating a fix across the duplicates.

## Core pipeline architecture (`M1_finale/final_m1_script.py`)

The full optimization pipeline (`optimise_circuit_pipeline`) is the reference implementation others iterate on. It runs in stages, each a distinct concern worth knowing about before editing any one of them:

1. **Partitioning** — `louvain_partition` builds a qubit-interaction graph (`build_interaction_graph`, weighted by 2-qubit gate count) and applies Louvain community detection to split qubits into blocks. `multilevel_partition` offers a Metis/Kernighan-Lin recursive alternative for bounding block size. `extract_interblock_gates` captures gates that cross block boundaries so they can be reinjected later.
2. **Sub-circuit extraction / recomposition** — `extract_subcircuit` pulls out a self-contained, locally-reindexed circuit per block; `recompose_from_blocks` stitches optimized block circuits back into a global circuit in original gate order, preserving inter-block gates unchanged.
3. **Fidelity metric** — `compute_fidelity` computes operator-overlap fidelity `|Tr(U_circ · U_target^†)| / 2^n`. It short-circuits to `1.0` above 15 qubits to avoid dense-matrix blowup — this cap is why the pipeline is only exercised on small/weakly-connected circuits (see the 30q/64q generators) despite claiming to scale.
4. **Intra-block optimization** — `optimise_block_nsga2` runs DEAP NSGA-II per block over three objectives (maximize fidelity, minimize transpiled depth, minimize chromosome-length cost), followed by `update_rotation_angles` — a finite-difference local search ("LAS") that nudges rotation angles to squeeze out extra fidelity after the genetic search converges.
5. **Inter-block gate injection** — two interchangeable strategies to add back cross-block entanglement lost during block-local optimization: `sa_injection` (simulated annealing over an energy combining gate count, depth, crosstalk, fidelity penalty) or `stochastic_injection` (random trial-and-keep-if-fidelity-improves). `fidelity_driven_injection` is a simpler greedy variant used as a complementary final pass.
6. **Compression** — hand-rolled passes (`cancel_inverse_gates`, `merge_rotations`, `remove_negligible_rotations`, composed in `compress_custom`) plus a standard Qiskit `PassManager` pass (`qiskit_opt_pass` — `Optimize1qGates` + `CommutativeCancellation`) applied at the end.
7. **Multi-objective quality indicators** — `evaluate_run`/`compute_hv`/`spread_delta`/`compute_spacing`/`compute_epsilon` implement standard MOO metrics (hypervolume, spread, spacing, epsilon indicator, and IGD when a reference Pareto front `P_star` is supplied) used to track convergence quality across generations, not just best-fitness.

Later stages depend on earlier ones' output shape (block list, per-block optimized circuit, kept-injection list) — when changing one stage's output type, check `optimise_circuit_pipeline`'s call sequence for how it's consumed downstream.

## Conventions specific to this codebase

- **Chromosome encoding**: individuals are Python lists of gene tuples `(gate_name, target_qubit, ctrl_qubit_or_None, angle_or_None)`. Any new gate type added to a `gate_pool` needs matching branches in every `build`/`build_fn` closure that turns chromosomes into `QuantumCircuit`s (these closures are redefined locally inside each optimisation function rather than shared).
- **DEAP creator classes** (`creator.FitnessMulti`, `creator.Individual`) are registered guarded by `hasattr(creator, ...)` checks since DEAP raises on re-registration — preserve that guard if touching this code, especially across notebook re-runs.
- **Figures are a primary output**, not incidental logging: `save_plot`/`plot_convergence`/`plot_pareto`/`plot_3d_clusters`/`plot_moo_history`/`export_all_indicators` all write PNGs to an `out_figs/` directory (created on demand) alongside stdout progress prints (with emoji status markers ✅/❌/💰/📎). Keep this pattern when extending a pipeline stage — the notebooks are read by inspecting saved figures as much as by reading code.
- **Comments, docstrings, print messages, plot labels, and markdown prose are English** throughout the repo (originally French, translated in full) — write new comments/docstrings/strings in English too. Two large thesis PDFs at the repo root (`MST2606 BOUHADOUZA ET GHLIB.pdf`, `PST2339_BOUHADOUZA_ET_GHLIB.pdf`) remain in French; they're academic writeups, not code, and weren't part of the translation.
- Most heavy computation (chromosome fitness evaluation) is parallelized via `joblib.Parallel(n_jobs=...)` — respect this when modifying `eval_ind`-style functions, since they must stay picklable for multiprocessing.
