import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RATIO_ORDER = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0)
METRICS = (
    "success_rate",
    "discounted_return",
    "undiscounted_return",
    "final_distance",
)
AXIS_LABELS = {
    "Axis R (Reachability)": "Reachability",
    "Axis C (Coverage)": "Coverage",
    "Axis B (Behavior)": "Behavioral tendency",
}
METRIC_LABELS = {
    "success_rate": "Success rate",
    "discounted_return": "Discounted return",
    "undiscounted_return": "Undiscounted return",
    "final_distance": "Final distance (lower is better)",
}
ENVIRONMENT_ORDER = ("PointMaze", "FetchPush")
AXIS_ORDER = tuple(AXIS_LABELS)
METHOD_ORDER = ("GCSL", "WGCSL", "GoFAR", "HER+DDPG")


def _ratio_string(values):
    formatted = []
    for ratio in sorted(values.dropna().unique()):
        formatted.append(str(int(ratio)) if float(ratio).is_integer() else str(ratio))
    return ", ".join(formatted)


def _write_tables(data, table_dir):
    data.to_csv(table_dir / "Table_A0_cleaned_raw_results.csv", index=False)
    baseline = (
        data[data["Ratio"] == 1.0]
        .groupby(["environment", "method", "method_label"], as_index=False)[
            list(METRICS)
        ]
        .mean()
        .rename(columns={metric: f"{metric}_baseline_r1_mean" for metric in METRICS})
    )
    baseline.to_csv(table_dir / "Table_A2_averaged_r1_baselines.csv", index=False)

    merged = data.merge(
        baseline,
        on=["environment", "method", "method_label"],
        how="left",
    )
    for metric in METRICS:
        merged[f"delta_{metric}"] = (
            merged[metric] - merged[f"{metric}_baseline_r1_mean"]
        )
    merged.to_csv(table_dir / "Table_A3_full_results_with_deltas.csv", index=False)

    group_columns = ["environment", "Imbalance_Axis", "subtype", "method_label"]
    matrix = (
        merged.groupby(group_columns, as_index=False)["Ratio"]
        .agg(_ratio_string)
        .rename(columns={"Ratio": "available_ratios"})
    )
    matrix["n_ratios"] = matrix["available_ratios"].str.split(", ").str.len()
    matrix.to_csv(table_dir / "Table_A1_experimental_matrix.csv", index=False)

    records = []
    for keys, group in merged.groupby(group_columns):
        environment, axis, subtype, method = keys
        record = {
            "environment": environment,
            "axis": axis,
            "subtype": subtype,
            "method": method,
        }
        for metric in ("discounted_return", "success_rate"):
            best = merged.loc[group[f"delta_{metric}"].idxmax()]
            worst = merged.loc[group[f"delta_{metric}"].idxmin()]
            record.update(
                {
                    f"best_{metric}_ratio": best["Ratio"],
                    f"best_{metric}": best[metric],
                    f"best_delta_{metric}": best[f"delta_{metric}"],
                    f"worst_{metric}_ratio": worst["Ratio"],
                    f"worst_{metric}": worst[metric],
                    f"worst_delta_{metric}": worst[f"delta_{metric}"],
                }
            )
        records.append(record)
    best_worst = pd.DataFrame(records)
    best_worst.to_csv(table_dir / "Table_A4_best_worst_conditions.csv", index=False)
    return baseline, merged, matrix, best_worst


def _write_figures(merged, figure_dir, table_dir):
    figure_index = []
    figure_number = 1
    tick_labels = [
        str(int(ratio)) if float(ratio).is_integer() else str(ratio)
        for ratio in RATIO_ORDER
    ]
    for environment in ENVIRONMENT_ORDER:
        environment_data = merged[merged["environment"] == environment]
        for axis in AXIS_ORDER:
            axis_data = environment_data[environment_data["Imbalance_Axis"] == axis]
            if axis_data.empty:
                continue
            for method in METHOD_ORDER:
                method_data = axis_data[axis_data["method_label"] == method]
                if method_data.empty:
                    continue
                for metric in METRICS:
                    figure, axes = plt.subplots(figsize=(7.2, 4.8))
                    for subtype, subtype_data in method_data.groupby("subtype"):
                        subtype_data = subtype_data.sort_values("Ratio")
                        x_values = [
                            RATIO_ORDER.index(float(ratio))
                            for ratio in subtype_data["Ratio"]
                            if float(ratio) in RATIO_ORDER
                        ]
                        y_values = [
                            subtype_data.loc[
                                subtype_data["Ratio"] == RATIO_ORDER[index], metric
                            ].iloc[0]
                            for index in x_values
                        ]
                        axes.plot(x_values, y_values, marker="o", label=subtype)
                    baseline = method_data[f"{metric}_baseline_r1_mean"].iloc[0]
                    axes.axhline(
                        baseline,
                        linestyle="--",
                        linewidth=1,
                        color="black",
                        label="averaged r = 1 baseline",
                    )
                    axes.set_xticks(range(len(RATIO_ORDER)), tick_labels)
                    axes.set_xlabel("Trajectory-count multiplier r")
                    axes.set_ylabel(METRIC_LABELS[metric])
                    axes.set_title(f"{environment} | {AXIS_LABELS[axis]} | {method}")
                    axes.grid(True, linewidth=0.5, alpha=0.4)
                    axes.legend(fontsize=8)
                    figure.tight_layout()
                    filename = (
                        f"Figure_A{figure_number:02d}_{environment}_"
                        f"{AXIS_LABELS[axis].replace(' ', '_')}_"
                        f"{method.replace('+', 'plus')}_{metric}.png"
                    )
                    figure.savefig(figure_dir / filename, dpi=300)
                    plt.close(figure)
                    figure_index.append(
                        {
                            "figure_no": f"A{figure_number:02d}",
                            "filename": filename,
                            "environment": environment,
                            "axis": AXIS_LABELS[axis],
                            "method": method,
                            "metric": METRIC_LABELS[metric],
                        }
                    )
                    figure_number += 1
    pd.DataFrame(figure_index).to_csv(
        table_dir / "Appendix_Figure_Index.csv", index=False
    )
    return len(figure_index)


def build_analysis(input_path, output_root, *, generate_figures=True):
    """Rebuild appendix tables and response-curve figures from cleaned rows."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_path)
    required = {
        "environment",
        "Imbalance_Axis",
        "subtype",
        "method",
        "method_label",
        "Ratio",
        *METRICS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"processed results are missing columns: {', '.join(missing)}")
    data["Ratio"] = pd.to_numeric(data["Ratio"], errors="raise")
    baseline, merged, matrix, best_worst = _write_tables(data, table_dir)
    figure_count = (
        _write_figures(merged, figure_dir, table_dir) if generate_figures else 0
    )
    return {
        "records": len(data),
        "baselines": len(baseline),
        "matrix_rows": len(matrix),
        "best_worst_rows": len(best_worst),
        "figures": figure_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the manuscript appendix tables and response curves."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/processed/Table_A0_cleaned_raw_results.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/appendix"))
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    summary = build_analysis(
        args.input,
        args.output_root,
        generate_figures=not args.tables_only,
    )
    print(summary)


if __name__ == "__main__":
    main()
