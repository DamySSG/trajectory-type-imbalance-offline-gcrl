"""Reproduce Table 4 of the FetchPush replication using only the Python standard library.

Each numeric column in table4_paired_counts.csv is a training seed. Its dataset
seed is the column value plus 1,000. An entry counts how many more (or fewer)
of the same 1,024 held-out episodes succeeded under the intervention than under
natural sampling. The between-algorithm contrast is calculated within each
seed pair. The confidence intervals apply individually to each mean. The Holm
adjustment covers the nine two-sided permutation tests together.

Run: python revision_20261118/reproduce_table4.py
Output: CSV on standard output.
"""

import csv
from itertools import product
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import sys


TRAINING_SEEDS = (61, 73, 89, 101, 113, 127, 139, 151, 163, 179)
EPISODES = 1024
T_CRITICAL = 2.2621571628540993  # Two-sided 95% interval with 9 degrees of freedom.
INTERVENTIONS = ("S-high +10 pp", "Remove G-high", "Remove S-high", "Remove A-pos")


def read_contrasts():
    path = Path(__file__).with_name("table4_paired_counts.csv")
    seed_columns = [str(seed) for seed in TRAINING_SEEDS]
    values = {}
    with path.open(newline="", encoding="ascii") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["algorithm", "intervention", *seed_columns]:
            raise ValueError("Unexpected seed columns in Table 4 data")
        for row in rows:
            key = (row["algorithm"], row["intervention"])
            if key in values:
                raise ValueError(f"Duplicate Table 4 contrast: {key}")
            values[key] = [int(row[seed]) for seed in seed_columns]

    expected = {(name, condition) for name in ("GoFAR", "SMORe") for condition in INTERVENTIONS}
    if set(values) != expected:
        raise ValueError("Table 4 requires two algorithms and four interventions")
    for condition in INTERVENTIONS:
        gofar = values["GoFAR", condition]
        smore = values["SMORe", condition]
        yield "GoFAR", condition, gofar
        yield "SMORe", condition, smore
        if condition == "S-high +10 pp":
            yield "SMORe - GoFAR", condition, [s - g for s, g in zip(smore, gofar)]


def permutation_p(counts):
    """Evaluate both tails by swapping the sign independently within each pair."""
    observed = abs(sum(counts))
    extremes = 0
    for signs in product((-1, 1), repeat=len(counts)):
        signed_sum = sum(sign * count for sign, count in zip(signs, counts))
        if abs(signed_sum) >= observed:
            extremes += 1
    return extremes / (2 ** len(counts))


def holm_adjust(p_values):
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        multiplier = len(p_values) - rank
        running_max = max(running_max, multiplier * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def main():
    contrasts = list(read_contrasts())
    p_values = [permutation_p(counts) for _, _, counts in contrasts]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("algorithm", "intervention", "n_pairs", "mean_pp", "sd_pp",
                     "ci95_low_pp", "ci95_high_pp", "p_exact_two_sided", "p_holm"))
    for (algorithm, condition, counts), raw, adjusted in zip(contrasts, p_values, holm_adjust(p_values)):
        effects = [100 * count / EPISODES for count in counts]
        mu, sd = mean(effects), stdev(effects)
        half_width = T_CRITICAL * sd / sqrt(len(effects))
        writer.writerow((algorithm, condition, len(effects), f"{mu:.2f}", f"{sd:.2f}",
                         f"{mu - half_width:.2f}", f"{mu + half_width:.2f}", raw, adjusted))


if __name__ == "__main__":
    main()
