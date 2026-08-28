# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Academic research project (French PFE — end-of-studies project) on **optimizing quantum circuits with genetic algorithms**. The goal is to take a target quantum circuit (or unitary), partition it, and evolve a cheaper/shallower circuit that still reproduces the target's behavior with high fidelity, using **Qiskit** for circuit simulation and **DEAP** for evolutionary computation (NSGA-II / NSGA-III style multi-objective optimization).

Most of the tree is still a research sandbox of **standalone Jupyter notebooks and scripts** (mono-objective → multi-objective → block-decomposed/scalable iterations), each self-contained and re-defining the functions it needs, with no shared package between them. The exception is `M1_finale/`, which has grown a real, if small, tooling layer on top of `final_m1_script.py`: it's importable as a module, has a CLI (`run_experiment.py`, `run_sweep.py`, `aggregate_results.py`) for structured/reproducible runs instead of ad-hoc notebook execution, and a pytest smoke test (`tests/test_pipeline_smoke.py`). There is still no lint/format tooling and no broader test suite beyond that one smoke test.

## Setup and running code

```bash
pip install -r requirements.txt
```

`requirements.txt` at the repo root is now pinned to the working `.venv`'s exact versions and includes `deap`, `pandas`, `pymoo`, and `pytest` (previously missing). `M1_finale/requirements_m1.txt` has a couple of extras for that subproject specifically (`nbconvert`, `pylatexenc`).

Most notebooks are still opened directly:

```bash
jupyter lab
```

For the canonical pipeline (`M1_finale/`), prefer the CLI over opening the notebook:

```bash
# Smoke-test the pipeline (~10s, tiny 6-qubit circuit) before a long run
pytest tests/test_pipeline_smoke.py

# Run a single configuration; writes runs/<run_id>/{config.json,metrics.json,final_circuit.qpy}
python run_experiment.py --n-qubits 12 --seed 0 --generations 100 --pop-size 100

# --circuit selects the target circuit generator (default: weak_random):
#   weak_random, qaoa_maxcut, w_state, qft, hw_efficient_ansatz
python run_experiment.py --circuit qaoa_maxcut --n-qubits 8 --qaoa-p 2 --generations 100 --pop-size 100

# Resumable multi-seed sweep over the fixed benchmark set (skips runs whose
# runs/<run_id>/metrics.json already exists)
python run_sweep.py --dry-run
python run_sweep.py --circuits weak_random qft --n-seeds 5

# Aggregate every completed run under runs/ into runs/results_master.csv
python aggregate_results.py
```

See the README's "Running experiments" section for the full flag reference (fidelity backend selection, injection trial counts, parallelism notes) — it's kept current there rather than duplicated here. `logs.txt` has the full narrative history of how this tooling and the scaling/research work evolved; `status.txt` has the latest supervisor-facing status snapshot.

No lint or format commands exist in this repo. Do not invent a broader test suite beyond `tests/test_pipeline_smoke.py` unless asked.

Committed `.venv/`, `mon_env/`, `envq/` directories are local virtualenvs (already gitignored alongside `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`) — never edit files inside them. `runs/` (experiment output) is also gitignored.

## Repository structure — reading order

The directories represent **successive iterations** of the same idea, roughly in this order of sophistication:

1. **`AG_mono/`** — mono-objective genetic algorithm: evolve a chromosome (gate sequence) that approximates a target unitary, optimizing fidelity only.
2. **`NSGA-II/`**, **`NSGA-III/`** — multi-objective versions (fidelity, depth, gate cost) using DEAP's NSGA-II selection and a hand-rolled NSGA-III (Deb & Jain 2014 — not DEAP's built-in `selNSGA3`; `NSGA-III/AG_multi_objectifs_NSGA3.ipynb` exists only on this branch, `ULBS_qiea`).
3. **`m1/`, `m1_bloc_indep/`, `M1_indicateurs_version/`, `M1_finale/`** — "M1" line of work: adds circuit **partitioning** into qubit blocks (so NSGA-II runs per-block instead of on the whole circuit) plus quality indicators (hypervolume, IGD, spread, epsilon — see below). `M1_finale/` is the most complete/cleaned-up version, has extracted `.py` scripts alongside its notebook, and is the only part of the repo with a CLI/test layer on top (`run_experiment.py`, `run_sweep.py`, `aggregate_results.py`, `tests/test_pipeline_smoke.py`) — read `M1_finale/final_m1_script.py` first to understand the full pipeline without wading through notebook cell markers.
4. **`m2_bloc_chevauche/`** — "block overlap" variant: blocks are allowed to share qubits at interfaces rather than being a strict partition.
5. **`Final_test/`, `final_test_AG/`, `test/`** — later validation/experiment notebooks applying the pipeline to specific target circuits (QAOA, MaxCut, W-state, QFT, VQE, 14/64-qubit cases, etc.), one subdirectory per experiment. These are throwaway experiment copies, not a distinct architecture — treat each as a snapshot of a notebook run against a specific test case rather than as unique code to maintain.

When asked to modify "the pipeline" without a specified location, assume `M1_finale/final_m1_script.py` — it is the canonical, most-maintained version. Notebooks with names like `code_travaille copy 3.ipynb` or `test3D(1) copy.ipynb` are duplicated experiment snapshots, not different components — don't assume divergent copies need to be kept in sync; check with the user before propagating a fix across the duplicates.

## Core pipeline architecture (`M1_finale/final_m1_script.py`)

The full optimization pipeline (`optimise_circuit_pipeline`) is the reference implementation others iterate on. It runs in stages, each a distinct concern worth knowing about before editing any one of them:

1. **Partitioning** — `louvain_partition` builds a qubit-interaction graph (`build_interaction_graph`, weighted by 2-qubit gate count) and applies Louvain community detection to split qubits into blocks. `multilevel_partition` offers a Metis/Kernighan-Lin recursive alternative for bounding block size. `extract_interblock_gates` captures gates that cross block boundaries so they can be reinjected later.
2. **Sub-circuit extraction / recomposition** — `extract_subcircuit` pulls out a self-contained, locally-reindexed circuit per block; `recompose_from_blocks` stitches optimized block circuits back into a global circuit in original gate order, preserving inter-block gates unchanged.
3. **Fidelity metric** — `compute_fidelity` computes the exact operator-overlap fidelity `|Tr(U_circ · U_target^†)| / 2^n` (dense `Operator`, intractable much past ~13 qubits). `safe_fidelity_between_circuits` is the scale-aware version actually used through most of the pipeline: exact below `fidelity_exact_threshold` (default 13), otherwise an approximate Monte-Carlo estimate — either `approximate_gate_fidelity_echo_mc` ("echo-test": same overlap quantity as an older SWAP-test formulation but computed on `n` qubits instead of `2n+1`, no CSWAP-induced entanglement, `--fidelity-echo-*` flags) or, opt-in per circuit family via `approximate_backend`/`--fidelity-approximate-backend`, `approximate_gate_fidelity_statevector_mc` (exact per-sample, no shot noise; much faster for highly-entangled circuits like `qaoa_maxcut`, much slower than the MPS-based echo estimator for low-entanglement families). No backend is auto-selected — pick one per circuit family, same as `injection_method`/`block_algorithm`/`mutation_scheme` below. See `logs.txt`'s "SCALING" section for the full tuning history and current per-family qubit ceilings.
4. **Intra-block optimization** — pluggable per-block optimizer, selected via `block_algorithm` and dispatched through the `BLOCK_OPTIMIZERS` registry dict (keyed by algorithm name): `optimise_block_nsga2` (DEAP NSGA-II, three objectives: maximize fidelity, minimize transpiled depth, minimize chromosome-length cost), `optimise_block_smsemoa` (hypervolume-driven SMS-EMOA — binary tournament by Pareto rank, environmental selection trims the boundary front by least-hypervolume-loss; hand-rolled in the existing DEAP loop rather than pymoo's `SMSEMOA`, by deliberate choice), and `optimise_block_nsga3` (DEAP's built-in `tools.selNSGA3` reference-point selection, Deb & Jain 2014 — deliberately reuses DEAP's tested implementation rather than the independently hand-rolled version in `NSGA-III/AG_multi_objectifs_NSGA3.ipynb`, whose own point was a from-scratch correctness demonstration, not a block-optimizer candidate). Adding a further MOO algorithm here is: write a function matching the same signature and add one entry to `BLOCK_OPTIMIZERS` — `optimise_circuit_pipeline`'s dispatch, `run_experiment.py`'s `--block-algorithm` choices, and `run_sweep.py`'s `--block-algorithms` choices all key off that registry (the last one can't import it directly — see its own comment — so its choices list needs a matching one-line addition). All three take a `mutation_scheme` (`"point"` / `"swap_add"` / `"swap_add_delete"`, via `mut_swap`/`mut_addition`/`mut_deletion` — the swap schemes make chromosomes variable-length) and an optional `hybrid_las` flag that runs `update_rotation_angles` ("LAS", a finite-difference local search on rotation angles) on the GA's winner afterward. No optimizer dominates the others outright — see `logs.txt`'s "RESEARCH ANGLE CHOSEN AND IMPLEMENTED" section for the original NSGA-II-vs-SMS-EMOA / mutation-scheme / hybrid-LAS ablation results and their caveats, and its "NSGA-III ADDED AS THIRD PER-BLOCK OPTIMIZER" section for the same three sweeps extended with `nsga3` — headline: `nsga3` reaches close to `nsga2`'s fidelity at roughly half the wall-clock cost, and pairs best with `hybrid_las` of the three.
5. **Inter-block gate injection** — re-adds cross-block entanglement lost during block-local optimization. Two interchangeable strategies selected via `injection_method`: `sa_injection` (simulated annealing over an energy combining gate count, depth, crosstalk, fidelity penalty) or `stochastic_injection` (random trial-and-keep-if-fidelity-improves) — both have closed-form/incremental fidelity fast paths now (see `logs.txt` "SCALING"), so pick between them on other grounds; `sa` also happens to now be the cheaper of the two on measured wall-clock. `fidelity_driven_injection` (greedy complement pass, tunable via `--fidelity-driven-max-trials`) **always runs afterward regardless of `injection_method`**, and per benchmark results usually produces the final circuit — treat it as the dominant stage when reasoning about final fidelity, not the requested `injection_method`.
6. **Compression** — hand-rolled passes (`cancel_inverse_gates`, `merge_rotations`, `remove_negligible_rotations`, composed in `compress_custom`) plus a standard Qiskit `PassManager` pass (`qiskit_opt_pass` — `Optimize1qGates` + `CommutativeCancellation`) applied at the end.
7. **Multi-objective quality indicators** — `evaluate_run`/`compute_hv`/`spread_delta`/`compute_spacing`/`compute_epsilon` implement standard MOO metrics (hypervolume, spread, spacing, epsilon indicator, and IGD when a reference Pareto front `P_star` is supplied) used to track convergence quality across generations, not just best-fitness.

Later stages depend on earlier ones' output shape (block list, per-block optimized circuit, kept-injection list) — when changing one stage's output type, check `optimise_circuit_pipeline`'s call sequence for how it's consumed downstream. When running via the CLI, each run's `metrics.json` records which optimizer/injection path and fidelity backend were actually used (`injection_path_used`, `fidelity_backend`, `fidelity_driven_tier`) — don't assume the requested flags are what a given run's numbers reflect without checking those fields.

## Conventions specific to this codebase

- **Chromosome encoding**: individuals are Python lists of gene tuples `(gate_name, target_qubit, ctrl_qubit_or_None, angle_or_None)`. Any new gate type added to a `gate_pool` needs matching branches in every `build`/`build_fn` closure that turns chromosomes into `QuantumCircuit`s (these closures are redefined locally inside each optimisation function rather than shared). In `M1_finale/final_m1_script.py`, chromosomes are fixed-length only under `mutation_scheme="point"`; `"swap_add"`/`"swap_add_delete"` (via `mut_addition`/`mut_deletion`) make them variable-length, so don't assume a fixed chromosome length when touching that code path.
- **DEAP creator classes** (`creator.FitnessMulti`, `creator.Individual`) are registered guarded by `hasattr(creator, ...)` checks since DEAP raises on re-registration — preserve that guard if touching this code, especially across notebook re-runs.
- **Figures are a primary output**, not incidental logging: `save_plot`/`plot_convergence`/`plot_pareto`/`plot_3d_clusters`/`plot_moo_history`/`export_all_indicators` all write PNGs to an `out_figs/` directory (created on demand) alongside stdout progress prints (with emoji status markers ✅/❌/💰/📎). Keep this pattern when extending a pipeline stage — the notebooks are read by inspecting saved figures as much as by reading code.
- **Comments, docstrings, print messages, plot labels, and markdown prose are English** throughout the repo (originally French, translated in full) — write new comments/docstrings/strings in English too. Two large thesis PDFs at the repo root (`MST2606 BOUHADOUZA ET GHLIB.pdf`, `PST2339_BOUHADOUZA_ET_GHLIB.pdf`) remain in French; they're academic writeups, not code, and weren't part of the translation.
- Most heavy computation (chromosome fitness evaluation) is parallelized via `joblib.Parallel(n_jobs=...)` — respect this when modifying `eval_ind`-style functions, since they must stay picklable for multiprocessing. In `M1_finale/`, every function in `BLOCK_OPTIMIZERS` (`optimise_block_nsga2`/`optimise_block_smsemoa`/`optimise_block_nsga3`) always uses `joblib.Parallel(-1)` (all cores) with no CLI knob to cap it — this is intentional, not an oversight, so don't add throttling without checking with the user first.
- `run_experiment.py`/`run_sweep.py` seed Python's global `random`/`np.random` before calling the pipeline. `louvain_partition` has no seed of its own, so calling the pipeline directly without replicating that seeding will break reproducibility.
