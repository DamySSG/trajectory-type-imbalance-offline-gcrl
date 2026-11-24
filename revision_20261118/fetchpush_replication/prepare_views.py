"""Rebuild the five weighted FetchPush training views for each replication seed."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


LABELS_SHA256 = "96ced21c25447d8a6c50c3d5e4da0e705f993c46cf03b11cfa53deb9ddc01b7a"
SEEDS = (61, 73, 89, 101, 113, 127, 139, 151, 163, 179)
REMOVAL_AXES = ("goal_separation", "spatial_support", "action_direction")


def prepare(labels, implementation, output):
    if hashlib.sha256(labels.read_bytes()).hexdigest() != LABELS_SHA256:
        raise ValueError("The label file differs from the one used for the replication.")
    if not (implementation / "revision/experimental_controls.py").is_file():
        raise ValueError("--implementation must point to the extracted implementation snapshot.")
    sys.path.insert(0, str(implementation))
    from revision.experimental_controls import build_cohort
    from revision.secondary_controls import removal_cohort

    blocks = json.loads(Path(__file__).with_name("cohorts_verified.json").read_text())
    if [(s, d) for s, d, _ in blocks] != [(s, s + 1000) for s in SEEDS]:
        raise ValueError("The seed list differs from the ten registered pairs.")
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=output.parent) as work:
        root = Path(work)
        results = root / "weights"
        results.mkdir()
        with np.load(labels, allow_pickle=False) as lab:
            for seed, data_seed, cohort_hash in blocks:
                primary = root / str(seed) / "primary"
                removal = primary.parent / "removal"
                build_cohort(labels, primary, "FetchPush", data_seed, changes=(0.0, 0.10))
                removal_cohort(labels, primary, removal, "FetchPush", data_seed)
                views = {
                    "natural": primary / "goal_separation_level0_quality_cross_axis_delta+0.00.npz",
                    "plus": primary / "spatial_support_level2_quality_cross_axis_delta+0.10.npz",
                }
                for axis in REMOVAL_AXES:
                    views[axis] = removal / f"{axis}_level2_source_only_removed.npz"

                with np.load(primary / "cohort.npz") as cohort, np.load(removal / "cohort.npz") as other:
                    if any(not np.array_equal(cohort[k], other[k]) for k in cohort.files):
                        raise ValueError(f"Primary and removal cohorts differ for seed {seed}.")
                    ids = cohort["global_ids"]
                    source = cohort["source"]
                    if hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest() != cohort_hash:
                        raise ValueError(f"The ordered cohort differs from the original for seed {seed}.")
                    if not np.array_equal(np.bincount(source), [5000, 5000]):
                        raise ValueError(f"Unexpected source counts for seed {seed}.")

                    arrays = {k: cohort[k] for k in cohort.files}
                    for name, filename in views.items():
                        with np.load(filename) as condition:
                            weights = condition["episode_probabilities"]
                        if (weights.shape != (10000,) or not np.all(np.isfinite(weights))
                                or np.any(weights < 0) or not np.allclose(
                                    np.bincount(source, weights=weights), [0.5, 0.5])):
                            raise ValueError(f"Invalid weights for {seed}/{name}.")
                        if name == "natural":
                            np.testing.assert_allclose(weights, 1.0 / len(ids))
                        elif name == "plus":
                            sparse = lab["label_spatial_support"][ids] == 2
                            np.testing.assert_allclose(weights[sparse].sum() - sparse.mean(), 0.10)
                        else:
                            removed = lab["label_" + name][ids] == 2
                            if np.any(weights[removed] != 0):
                                raise ValueError(f"Removed category has positive weight for {seed}/{name}.")
                        arrays[name] = weights
                    np.savez_compressed(results / f"seed_{seed}_data_{data_seed}.npz", **arrays)
        results.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path, help="Original FetchPush trajectory labels (.npz)")
    parser.add_argument("--implementation", required=True, type=Path, help="Extracted implementation_snapshot.zip")
    parser.add_argument("--output", required=True, type=Path, help="New directory for the ten output files")
    args = parser.parse_args()
    prepare(args.labels.resolve(), args.implementation.resolve(), args.output.resolve())
