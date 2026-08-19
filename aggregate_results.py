#!/usr/bin/env python
"""Aggregate runs/*/{config.json,metrics.json} into one master results table.

Scans the runs directory produced by run_experiment.py / run_sweep.py and
builds a single pandas DataFrame -- one row per completed run -- written to
CSV. This is the direct source for paper tables/figures once enough runs
exist; re-run any time to refresh it, since it's a pure re-scan (no
incremental state to go stale).

Example:
    python aggregate_results.py
    python aggregate_results.py --runs-dir runs --out runs/results_master.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CONFIG_FIELDS = [
    "run_id", "circuit", "n_qubits", "depth", "twoq_gates_total",
    "connectivity_edges", "injection_method", "fid_threshold", "sa_iters",
    "generations", "pop_size", "qubit_duplication_threshold", "seed",
]

METRIC_FIELDS = [
    "fidelity_final", "depth_before", "depth_after", "cost_before",
    "cost_after", "wall_clock_s", "original_num_qubits", "final_num_qubits",
    "injection_path_used", "fidelity_after_injection_method",
    "fidelity_after_fidelity_driven_greedy", "fidelity_backend",
    "fidelity_exact_threshold", "fidelity_samples", "fidelity_shots",
]

MOO_METRICS = ["HV", "Spread", "Spacing"]


def _moo_aggregates(moo_metrics_per_block):
    row = {}
    for metric in MOO_METRICS:
        values = [b[metric] for b in moo_metrics_per_block if b.get(metric) is not None]
        row[f"mean_{metric.lower()}"] = sum(values) / len(values) if values else None
    n_pareto = [b["n_pareto"] for b in moo_metrics_per_block if b.get("n_pareto") is not None]
    row["total_n_pareto"] = sum(n_pareto) if n_pareto else None
    return row


def collect_run(run_dir: Path) -> dict | None:
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"⚠️ skipping {run_dir.name}: no metrics.json (still running or failed)", file=sys.stderr)
        return None
    if not config_path.exists():
        print(f"⚠️ skipping {run_dir.name}: no config.json", file=sys.stderr)
        return None

    config = json.loads(config_path.read_text())
    metrics = json.loads(metrics_path.read_text())

    row = {field: config.get(field) for field in CONFIG_FIELDS}
    row.update({field: metrics.get(field) for field in METRIC_FIELDS})
    row["n_blocks"] = len(metrics.get("blocks", []))
    row["n_kept_injections"] = len(metrics.get("kept_injections", []))
    for stage, t in metrics.get("stage_timings_s", {}).items():
        row[f"stage_{stage}_s"] = t
    row.update(_moo_aggregates(metrics.get("moo_metrics_per_block", [])))
    return row


def aggregate(runs_dir: Path) -> pd.DataFrame:
    rows = [
        row for run_dir in sorted(runs_dir.iterdir())
        if run_dir.is_dir() and (row := collect_run(run_dir)) is not None
    ]
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs", type=Path)
    p.add_argument("--out", default=None, type=Path,
                    help="Defaults to <runs-dir>/results_master.csv")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or (args.runs_dir / "results_master.csv")

    if not args.runs_dir.exists():
        print(f"No such runs directory: {args.runs_dir}", file=sys.stderr)
        return 1

    df = aggregate(args.runs_dir)
    if df.empty:
        print("No completed runs found (no runs/*/metrics.json).", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {out}")
    print("Columns:", ", ".join(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
