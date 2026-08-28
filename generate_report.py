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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
    """sa vs stochastic injection, current-schema baseline (post fast-path fix)."""
    subset = df[
        (df["n_qubits"] == BASELINE_N_QUBITS)
        & (~df["is_legacy_schema"])
        & (df["block_algorithm"] == "nsga2")
        & (df["mutation_scheme"] == "point")
        & (df["hybrid_las"] == False)
    ]
    return subset.groupby("injection_method").agg(
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
    fig_bar(algo["mean_hv"], title="Hypervolume by block algorithm", ylabel="mean_hv",
            out_path=figures_dir / "algorithm_hypervolume.png")

    fig_grouped_bar(tables["mutation_ablation"], x="mutation_scheme", hue="block_algorithm",
                     y="fidelity_final", title="Mutation-scheme ablation",
                     ylabel="fidelity_final", out_path=figures_dir / "mutation_ablation.png")

    fig_grouped_bar(tables["hybrid_las_ablation"], x="block_algorithm", hue="hybrid_las",
                     y="fidelity_final", title="Hybrid GA+LAS ablation",
                     ylabel="fidelity_final", out_path=figures_dir / "hybrid_las_ablation.png")

    inj = tables["injection_method_comparison"].set_index("injection_method")
    fig_bar(inj["fidelity_final"], title="Injection method: fidelity", ylabel="fidelity_final",
            out_path=figures_dir / "injection_method_fidelity.png")
    fig_bar(inj["wall_clock_s"], title="Injection method: wall-clock cost", ylabel="wall_clock_s",
            out_path=figures_dir / "injection_method_wallclock.png")

    fig_scaling_lines(tables["scaling"], y="wall_clock_s", title="Wall-clock cost vs n_qubits",
                       ylabel="wall_clock_s (log)", out_path=figures_dir / "scaling_wallclock.png", log_y=True)
    fig_scaling_lines(tables["scaling"], y="fidelity_final", title="Fidelity vs n_qubits",
                       ylabel="fidelity_final", out_path=figures_dir / "scaling_fidelity.png")

    print(f"✅ Wrote {len(tables)} tables -> {tables_dir}/ (+ summary.md)")
    print(f"✅ Wrote 8 figures -> {figures_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
