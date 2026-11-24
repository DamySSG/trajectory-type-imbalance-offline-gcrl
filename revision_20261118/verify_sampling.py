"""Check the published sampling probabilities (requires openpyxl)."""

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import load_workbook


SOURCE = Path(__file__).resolve().parents[1] / "figure_data" / "Sampling_cross_tabs.xlsx"
EXPECTED_MODES = {"quality_cross_axis": 441, "source_only": 81}
TOLERANCE_PP = 1e-8  # Percentage points, allowing for rounding in workbook cells.


def records(sheet):
    rows = iter(sheet.values)
    columns = next(rows)
    for row in rows:
        yield dict(zip(columns, row))


def verify(book):
    controls = {}
    modes = {}
    audit_max = defaultdict(float)
    for row in records(book["Views"]):
        view, mode = row["view_id"], row["balancing"]
        if view in modes or mode not in EXPECTED_MODES:
            raise ValueError(f"Duplicate view or unexpected matching mode: {view}, {mode}")
        modes[view] = mode
        controls[view] = {"source"}
        if mode == "quality_cross_axis":
            direction = "motion_direction" if row["task"] == "AntMazeLarge" else "action_direction"
            descriptors = {"goal_separation", "spatial_support", direction}
            if row["target"] not in descriptors:
                raise ValueError(f"Unknown target {row['target']} in {view}")
            controls[view].update({"final_success", "return_group"}, descriptors - {row["target"]})
        # This stored audit covers all controlled variables jointly.
        change = row["max_controlled_joint_change_pp"]
        if change is None or not math.isfinite(change) or not 0 <= change <= TOLERANCE_PP:
            raise ValueError(f"Invalid maximum change in controlled strata for {view}: {change}")
        audit_max[mode] = max(audit_max[mode], change)
    counts = Counter(modes.values())
    if counts != EXPECTED_MODES:
        raise ValueError(f"Unexpected number of settings: {counts}")

    # Independently reconstruct each controlled marginal from one source-by-variable table.
    # Two-variable tables cannot reconstruct the full five-variable joint distribution.
    marginals = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for row in records(book["Cross-tabs"]):
        if row["variable_1"] != "source":
            continue
        view, variable = row["view_id"], row["variable_2"]
        before, after = row["before_share"], row["after_share"]
        if not (0 <= before <= 1 and 0 <= after <= 1):
            raise ValueError(f"Invalid sampling probability in {view}, {variable}")
        cells = []
        if variable == "final_success":
            cells.append(("source", row["level_1"]))
        if variable in controls[view]:
            cells.append((variable, row["level_2"]))
        for name, level in cells:
            shares = marginals[view, name][level]
            shares[0] += before
            shares[1] += after

    marginal_max = defaultdict(float)
    for view, variables in controls.items():
        for variable in variables:
            levels = marginals[view, variable].values()
            if not levels:
                raise ValueError(f"Missing marginal for {view}, {variable}")
            before, after = zip(*levels)
            if any(abs(sum(side) - 1) * 100 > TOLERANCE_PP for side in (before, after)):
                raise ValueError(f"Shares do not sum to one for {view}, {variable}")
            change = max(abs(a - b) * 100 for a, b in levels)
            if change > TOLERANCE_PP:
                raise ValueError(f"Controlled {variable} changes by {change} pp in {view}")
            mode = modes[view]
            marginal_max[mode] = max(marginal_max[mode], change)
    return counts, audit_max, marginal_max


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="sampling-distributions workbook")
    args = parser.parse_args()
    book = load_workbook(args.source, read_only=True, data_only=True)
    try:
        counts, audit_max, marginal_max = verify(book)
    finally:
        book.close()
    print(f"{sum(counts.values())} sampling settings.")
    for mode, count in counts.items():
        print(f"{mode} ({count}): saved maximum controlled-stratum change "
              f"{audit_max[mode]:.6g} pp; largest controlled marginal change "
              f"recalculated from cross-tabs {marginal_max[mode]:.6g} pp.")


if __name__ == "__main__":
    main()
