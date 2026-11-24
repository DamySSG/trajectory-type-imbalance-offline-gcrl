"""Reproduce Figure 3 cells and count changes in direction across sampling levels."""

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "figure_data" / "Dose_profiles_source.csv"
METHODS = ("gcsl", "wgcsl", "ddpg", "gofar", "smore", "hiql", "crl")
DIRECTION = {
    "FetchPush": "action_direction",
    "PointMaze": "action_direction",
    "AntMazeLarge": "motion_direction",
}
SHIFTS = (-10, -5, -2.5, 0, 2.5, 5, 10)
SEEDS = ((11, 1011), (29, 1029), (47, 1047))
HEADER = ("task", "method", "target", "shift_pp", "n_pairs", "mean_success_percent",
          "sample_sd_success_pp", "mean_paired_change_pp", "sample_sd_paired_change_pp")


def summarize(source):
    cells = defaultdict(dict)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["task"], row["method"], row["target"], float(row["shift_pp"]))
            seed = (int(row["learner_seed"]), int(row["data_seed"]))
            if seed in cells[key]:
                raise ValueError(f"Repeated seed pair {seed} in {key}")
            success = float(row["success_percent"])
            baseline = float(row["paired_natural_percent"])
            change = float(row["paired_change_pp"])
            if not (0 <= success <= 100 and 0 <= baseline <= 100
                    and math.isclose(success - baseline, change, rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"Inconsistent success or change in {key}, seed pair {seed}")
            cells[key][seed] = (success, baseline, change)

    rows = []
    reversing = small_reversals = flat_zero = 0
    for task, method in itertools.product(DIRECTION, METHODS):
        for target in ("goal_separation", "spatial_support", DIRECTION[task]):
            profile = (task, method, target)
            means = []
            for shift in SHIFTS:
                key = (*profile, shift)
                observations = cells.get(key, {})
                if set(observations) != set(SEEDS):
                    raise ValueError(f"Expected seed pairs {SEEDS} for {key}, found {sorted(observations)}")
                natural = cells.get((*profile, 0), {})
                for seed, (_, baseline, _) in observations.items():
                    if seed not in natural or natural[seed][0] != baseline:
                        raise ValueError(f"Natural-sampling baseline mismatch in {key}, pair {seed}")
                success, _, changes = zip(*observations.values())
                means.append(statistics.mean(success))
                rows.append((*key, len(observations), means[-1], statistics.stdev(success),
                             statistics.mean(changes), statistics.stdev(changes)))
            # Traverse the seven shifts from -10 to +10. Ties add no change of direction.
            steps = [b - a for a, b in zip(means, means[1:])]
            if min(steps) < 0 < max(steps):
                reversing += 1
                if max(means) - min(means) < 1:
                    small_reversals += 1
            if all(mean == 0 for mean in means):
                flat_zero += 1
    if len(rows) != len(cells):
        raise ValueError("Source contains cells outside the Figure 3 design")
    return rows, reversing, small_reversals, flat_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="input Figure 3 CSV")
    parser.add_argument("--output", type=Path, help="write one summary row per Figure 3 cell")
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.source.resolve():
        parser.error("Output and source must have different paths")
    try:
        rows, reversing, small, flat = summarize(args.source)
        if args.output:
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADER)
                writer.writerows(rows)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f"Figure 3 check failed: {error}\n")
    print(f"{len(rows)} cells, {len(SEEDS)} paired seeds per cell, "
          f"{sum(row[4] for row in rows)} observations.")
    print(f"Mean success rises and falls across the seven sampling levels in "
          f"{reversing}/{len(rows) // len(SHIFTS)} profiles; {small} of these span "
          f"less than one percentage point. {flat} profiles are flat at zero success.")


if __name__ == "__main__":
    main()
