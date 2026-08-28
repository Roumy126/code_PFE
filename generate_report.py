#!/usr/bin/env python
"""Generate paper-ready tables and figures from runs/results_master.csv.

Reads the aggregated results table (see aggregate_results.py) and produces,
under --out-dir (default report/): one CSV per analysis table in
<out-dir>/tables/, a combined <out-dir>/tables/summary.md with all tables
rendered as markdown, and one PNG per figure in <out-dir>/figures/. Re-run
any time results_master.csv is refreshed -- this is a pure re-derivation,
no incremental state to go stale.

The analyses mirror the sweeps documented in logs.txt's "RESEARCH ANGLE
CHOSEN AND IMPLEMENTED" and "SCALING" sections: block-algorithm comparison
(NSGA-II vs SMS-EMOA), mutation-scheme ablation, hybrid-LAS ablation,
injection-method comparison (sa vs stochastic), and per-circuit-family
qubit scaling.

results_master.csv mixes two schema eras: current rows have block_algorithm/
mutation_scheme/hybrid_las populated; a 50-row legacy baseline (the original
Phase 2 8-qubit sa-vs-stochastic comparison, pre-dating those columns) has
them as NaN. The ablation tables use only current-schema rows; the legacy
baseline is reported separately for historical reference.

Example:
    python generate_report.py
    python generate_report.py --input runs/results_master.csv --out-dir report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.indicators.hv import HV as _PymooHV

# Fixed categorical order (dataviz skill reference palette) -- assign by
# identity/order, never re-cycle or reassign per chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown-table renderer (avoids adding a tabulate dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["is_legacy_schema"] = df["block_algorithm"].isna()
    return df


# The fixed 5-circuit x 5-seed benchmark set the algorithm/mutation/hybrid-LAS/
# injection-method ablations were run on (see logs.txt's "RESEARCH ANGLE CHOSEN
# AND IMPLEMENTED" and Phase 2 baseline). n_qubits also has 12/20-qubit scaling
# rows at some of the same setting combinations (e.g. nsga2/point/no-LAS/
# stochastic) -- excluding them here is required, not just a default, or the
# ablation tables silently pool two different qubit counts into one mean.
BASELINE_N_QUBITS = 8


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_algorithm_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """NSGA-II vs SMS-EMOA at baseline settings (point mutation, no hybrid LAS)."""
    baseline = df[
        (df["n_qubits"] == BASELINE_N_QUBITS)
        & (~df["is_legacy_schema"])
        & (df["mutation_scheme"] == "point")
        & (df["hybrid_las"] == False)
        & (df["injection_method"] == "stochastic")
    ]
    g = baseline.groupby("block_algorithm").agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        depth_after=("depth_after", "mean"),
        cost_after=("cost_after", "mean"),
        mean_hv=("mean_hv", "mean"),
        total_n_pareto=("total_n_pareto", "mean"),
        wall_clock_s=("wall_clock_s", "mean"),
    ).reset_index()
    return g


# The algorithm-comparison baseline's own circuits/algorithms/seed count (must match
# table_algorithm_comparison's filter above) -- used to locate each run's metrics.json
# directly, since raw Pareto-front points aren't (and shouldn't be) flattened into
# results_master.csv's per-run summary row.
FAIR_HV_CIRCUITS = ["weak_random", "qaoa_maxcut", "w_state", "qft", "hw_efficient_ansatz"]
FAIR_HV_ALGORITHMS = ["nsga2", "smsemoa", "nsga3"]
FAIR_HV_N_SEEDS = 5
# Shared, FIXED reference point in normalized objective space -- the whole point of this
# table. Matches this codebase's existing "+0.1 margin past the worst point" convention
# (see evaluate_run in final_m1_script.py), just made shared across algorithms instead of
# adaptively re-derived per run, which is what makes results_master.csv's own mean_hv
# non-comparable across algorithms (see logs.txt's "HYPERVOLUME COMPARABILITY" entry).
FAIR_HV_REF_POINT = np.array([1.1, 1.1, 1.1])


def _normalize_shared(costs: np.ndarray, f_min: np.ndarray, f_max: np.ndarray) -> np.ndarray:
    denom = f_max - f_min
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (costs - f_min) / denom


def table_fair_hv_comparison(runs_dir: Path) -> pd.DataFrame:
    """Cross-algorithm-comparable hypervolume, computed from raw Pareto-front points
    (metrics.json's front_raw field, added 2026-08-28 specifically to make this table
    possible) instead of results_master.csv's mean_hv, which uses a private per-run
    reference point and normalization -- valid for tracking one run's own convergence,
    NOT for ranking algorithms against each other (verified: nsga2 ranks above nsga3
    under the adaptive scheme on real data, but nsga3 ranks above nsga2 on the same
    matched front under this fixed-reference scheme -- see logs.txt).

    For each (circuit, seed) with front_raw data from >=2 algorithms, pools all present
    algorithms' front points PER BLOCK INDEX (blocks are identical across algorithms for
    a given circuit+seed, since partitioning happens before block_algorithm is chosen),
    derives ONE shared min/max normalization from that pooled set, and evaluates each
    algorithm's HV against the one shared FAIR_HV_REF_POINT.
    """
    rows = []
    n_mismatched = 0
    for circuit in FAIR_HV_CIRCUITS:
        for seed in range(FAIR_HV_N_SEEDS):
            per_algo_blocks = {}
            for algo in FAIR_HV_ALGORITHMS:
                run_id = f"{circuit}_8q_stochastic_{algo}_point_las0_g100_p100_seed{seed}"
                metrics_path = runs_dir / run_id / "metrics.json"
                if not metrics_path.exists():
                    continue
                blocks = json.loads(metrics_path.read_text()).get("moo_metrics_per_block", [])
                if blocks and all("front_raw" in b and b["front_raw"] for b in blocks):
                    per_algo_blocks[algo] = blocks
            if len(per_algo_blocks) < 2:
                continue
            n_blocks_seen = {len(v) for v in per_algo_blocks.values()}
            if len(n_blocks_seen) != 1:
                n_mismatched += 1
                continue
            for b in range(n_blocks_seen.pop()):
                per_algo_costs = {}
                for algo, blocks in per_algo_blocks.items():
                    front = np.array(blocks[b]["front_raw"])
                    costs = front.copy(); costs[:, 0] = 1.0 - costs[:, 0]  # minimize convention, matches evaluate_run
                    per_algo_costs[algo] = costs
                pooled = np.vstack(list(per_algo_costs.values()))
                f_min, f_max = pooled.min(axis=0), pooled.max(axis=0)
                for algo, costs in per_algo_costs.items():
                    hv = _PymooHV(ref_point=FAIR_HV_REF_POINT)(_normalize_shared(costs, f_min, f_max))
                    rows.append({"circuit": circuit, "seed": seed, "block": b, "block_algorithm": algo, "fair_hv": hv})
    if n_mismatched:
        print(f"⚠️ table_fair_hv_comparison: skipped {n_mismatched} (circuit, seed) groups "
              f"with mismatched block counts across algorithms")
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["block_algorithm", "n_runs", "fair_mean_hv"])
    per_run = detail.groupby(["circuit", "seed", "block_algorithm"])["fair_hv"].mean().reset_index()
    return per_run.groupby("block_algorithm").agg(
        n_runs=("fair_hv", "count"),
        fair_mean_hv=("fair_hv", "mean"),
    ).reset_index()


def table_mutation_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Mutation-scheme ablation, split by block algorithm (no hybrid LAS)."""
    subset = df[
        (df["n_qubits"] == BASELINE_N_QUBITS)
        & (~df["is_legacy_schema"])
        & (df["hybrid_las"] == False)
        & (df["injection_method"] == "stochastic")
    ]
    g = subset.groupby(["block_algorithm", "mutation_scheme"]).agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        depth_after=("depth_after", "mean"),
        cost_after=("cost_after", "mean"),
        mean_hv=("mean_hv", "mean"),
    ).reset_index()
    return g


def table_hybrid_las_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Hybrid GA+LAS ablation at point mutation, split by block algorithm.

    Includes a paired improved-fraction: the share of matching
    (circuit, seed) pairs where fidelity_final was higher with hybrid_las=True
    than with hybrid_las=False, matching how logs.txt reports this ablation
    ("16/25 improved" style) rather than only a mean delta.
    """
    subset = df[
        (df["n_qubits"] == BASELINE_N_QUBITS)
        & (~df["is_legacy_schema"])
        & (df["mutation_scheme"] == "point")
        & (df["injection_method"] == "stochastic")
    ]
    g = subset.groupby(["block_algorithm", "hybrid_las"]).agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        mean_fidelity_before_las=("mean_fidelity_before_las", "mean"),
        mean_fidelity_after_las=("mean_fidelity_after_las", "mean"),
    ).reset_index()

    rows = []
    for algo, algo_df in subset.groupby("block_algorithm"):
        off = algo_df[algo_df["hybrid_las"] == False].set_index(["circuit", "seed"])["fidelity_final"]
        on = algo_df[algo_df["hybrid_las"] == True].set_index(["circuit", "seed"])["fidelity_final"]
        paired = off.to_frame("off").join(on.to_frame("on"), how="inner")
        if len(paired):
            improved_frac = (paired["on"] > paired["off"]).mean()
            mean_delta = (paired["on"] - paired["off"]).mean()
        else:
            improved_frac = mean_delta = None
        rows.append({
            "block_algorithm": algo, "n_paired": len(paired),
            "frac_improved": improved_frac, "mean_delta_fidelity_final": mean_delta,
        })
    pairing = pd.DataFrame(rows)
    return g.merge(pairing, on="block_algorithm", how="left")


def table_injection_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """sa vs stochastic injection, current-schema baseline (post fast-path fix), per
    block algorithm -- this axis started nsga2-only; extend as other algorithms grow
    sa-injection data (see logs.txt's "NSGA-III ADDED AS THIRD PER-BLOCK OPTIMIZER")."""
    subset = df[
        (df["n_qubits"] == BASELINE_N_QUBITS)
        & (~df["is_legacy_schema"])
        & (df["mutation_scheme"] == "point")
        & (df["hybrid_las"] == False)
    ]
    return subset.groupby(["block_algorithm", "injection_method"]).agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        wall_clock_s=("wall_clock_s", "mean"),
    ).reset_index()


def table_injection_method_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """The original Phase 2 baseline (pre sa_injection fast-path fix), for
    historical comparison against table_injection_method_comparison."""
    subset = df[(df["n_qubits"] == BASELINE_N_QUBITS) & (df["is_legacy_schema"])]
    return subset.groupby("injection_method").agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        wall_clock_s=("wall_clock_s", "mean"),
    ).reset_index()


def table_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit-family qubit scaling: cost and fidelity vs n_qubits."""
    g = df.groupby(["circuit", "n_qubits"]).agg(
        n_runs=("run_id", "count"),
        fidelity_final=("fidelity_final", "mean"),
        wall_clock_s=("wall_clock_s", "mean"),
    ).reset_index()
    return g.sort_values(["circuit", "n_qubits"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bar(series: pd.Series, *, title: str, ylabel: str, out_path: Path, log_y: bool = False):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(series))]
    ax.bar([str(i) for i in series.index], series.values, color=colors, width=0.55)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    if log_y:
        ax.set_yscale("log")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grouped_bar(df: pd.DataFrame, *, x: str, hue: str, y: str, title: str, ylabel: str, out_path: Path):
    x_vals = list(dict.fromkeys(df[x]))
    hue_vals = list(dict.fromkeys(df[hue]))
    n_hue = len(hue_vals)
    width = 0.8 / n_hue
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, hv in enumerate(hue_vals):
        sub = df[df[hue] == hv].set_index(x)[y]
        heights = [sub.get(xv, float("nan")) for xv in x_vals]
        offsets = [j + (i - (n_hue - 1) / 2) * width for j in range(len(x_vals))]
        ax.bar(offsets, heights, width=width * 0.9, label=str(hv), color=CATEGORICAL[i % len(CATEGORICAL)])
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_scaling_lines(df: pd.DataFrame, *, y: str, title: str, ylabel: str, out_path: Path, log_y: bool = False):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, (circuit, sub) in enumerate(df.groupby("circuit")):
        sub = sub.sort_values("n_qubits")
        ax.plot(sub["n_qubits"], sub[y], marker="o", markersize=5, linewidth=2,
                label=circuit, color=CATEGORICAL[i % len(CATEGORICAL)])
    ax.set_xlabel("n_qubits", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.set_title(title, fontsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="runs/results_master.csv", type=Path)
    p.add_argument("--out-dir", default="report", type=Path)
    p.add_argument("--runs-dir", default="runs", type=Path,
                    help="Used only by table_fair_hv_comparison, which reads metrics.json "
                         "files directly for their raw Pareto-front points.")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"❌ No such file: {args.input} (run aggregate_results.py first)")
        return 1

    df = load_results(args.input)
    tables_dir = args.out_dir / "tables"
    figures_dir = args.out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "algorithm_comparison": table_algorithm_comparison(df),
        "fair_hv_comparison": table_fair_hv_comparison(args.runs_dir),
        "mutation_ablation": table_mutation_ablation(df),
        "hybrid_las_ablation": table_hybrid_las_ablation(df),
        "injection_method_comparison": table_injection_method_comparison(df),
        "injection_method_legacy": table_injection_method_legacy(df),
        "scaling": table_scaling(df),
    }
    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)

    summary_md = "\n\n".join(
        f"## {name}\n\n{_df_to_markdown(table)}" for name, table in tables.items()
    )
    (tables_dir / "summary.md").write_text(f"# Results summary\n\n{summary_md}\n")

    algo = tables["algorithm_comparison"].set_index("block_algorithm")
    fig_bar(algo["fidelity_final"], title="Fidelity by block algorithm", ylabel="fidelity_final",
            out_path=figures_dir / "algorithm_fidelity.png")

    # Two SEPARATE figures, not one combined chart -- adaptive HV (~0.05) and fair HV
    # (~0.9-1.0, a shared-reference scale) differ by ~20x in magnitude for unrelated
    # reasons (see table_fair_hv_comparison's docstring), so plotting them on one shared
    # axis would visually imply "fair is bigger/better", which is not a real comparison
    # -- only each figure's OWN cross-algorithm ranking is meaningful.
    fig_bar(algo["mean_hv"], title="Hypervolume (adaptive per-run reference -- NOT\ncomparable across algorithms, see logs.txt)",
            ylabel="mean_hv", out_path=figures_dir / "algorithm_hypervolume.png")
    if not tables["fair_hv_comparison"].empty:
        fair_hv = tables["fair_hv_comparison"].set_index("block_algorithm")["fair_mean_hv"]
        fig_bar(fair_hv, title="Hypervolume (fair: shared fixed reference point)",
                ylabel="fair_mean_hv", out_path=figures_dir / "algorithm_hypervolume_fair.png")

    fig_grouped_bar(tables["mutation_ablation"], x="mutation_scheme", hue="block_algorithm",
                     y="fidelity_final", title="Mutation-scheme ablation",
                     ylabel="fidelity_final", out_path=figures_dir / "mutation_ablation.png")

    fig_grouped_bar(tables["hybrid_las_ablation"], x="block_algorithm", hue="hybrid_las",
                     y="fidelity_final", title="Hybrid GA+LAS ablation",
                     ylabel="fidelity_final", out_path=figures_dir / "hybrid_las_ablation.png")

    fig_grouped_bar(tables["injection_method_comparison"], x="block_algorithm", hue="injection_method",
                     y="fidelity_final", title="Injection method: fidelity",
                     ylabel="fidelity_final", out_path=figures_dir / "injection_method_fidelity.png")
    fig_grouped_bar(tables["injection_method_comparison"], x="block_algorithm", hue="injection_method",
                     y="wall_clock_s", title="Injection method: wall-clock cost",
                     ylabel="wall_clock_s", out_path=figures_dir / "injection_method_wallclock.png")

    fig_scaling_lines(tables["scaling"], y="wall_clock_s", title="Wall-clock cost vs n_qubits",
                       ylabel="wall_clock_s (log)", out_path=figures_dir / "scaling_wallclock.png", log_y=True)
    fig_scaling_lines(tables["scaling"], y="fidelity_final", title="Fidelity vs n_qubits",
                       ylabel="fidelity_final", out_path=figures_dir / "scaling_fidelity.png")

    print(f"✅ Wrote {len(tables)} tables -> {tables_dir}/ (+ summary.md)")
    print(f"✅ Wrote {len(list(figures_dir.glob('*.png')))} figures -> {figures_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
