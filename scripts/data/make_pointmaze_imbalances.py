import argparse
import csv
from pathlib import Path
import pickle

import numpy as np

from scripts.data.interventions import intervene
from scripts.data.pointmaze import behavior_masks, coverage_masks, reachability_masks


DEFAULT_RATIOS = (0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0)
MANIFEST_FIELDS = (
    "dimension",
    "subtype",
    "ratio",
    "seed",
    "episodes_before",
    "episodes_after",
    "target_before",
    "target_after",
    "other_before",
    "other_after",
    "realized_ratio",
    "path",
)


def _select_base(buffer, episode_count, rng):
    available = int(np.asarray(buffer["o"]).shape[0])
    if episode_count is None or episode_count == available:
        return buffer
    if episode_count <= 0 or episode_count > available:
        raise ValueError("base_episodes must lie between 1 and the available count")
    indices = np.sort(rng.permutation(available)[:episode_count])
    return {
        key: (np.asarray(value)[indices].copy() if np.asarray(value).shape[0] == available else value)
        for key, value in buffer.items()
    }


def _save_buffer(buffer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_manifest(rows, output_root):
    with output_root.joinpath("manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_variants(
    buffer,
    output_root,
    *,
    ratios=DEFAULT_RATIOS,
    seed=0,
    base_episodes=10000,
    behavior_eps=1e-3,
):
    """Generate all PointMaze-Medium-v3 trajectory-type intervention datasets."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = _select_base(buffer, base_episodes, rng)
    achieved_goals = np.asarray(base["ag"])
    coverage, _ = coverage_masks(achieved_goals)
    behavior = behavior_masks(achieved_goals, eps=behavior_eps)
    reachability, _ = reachability_masks(achieved_goals)
    conditions = (
        ("coverage", "tl", coverage["top_left"]),
        ("coverage", "tr", coverage["top_right"]),
        ("coverage", "bl", coverage["bottom_left"]),
        ("coverage", "br", coverage["bottom_right"]),
        ("coverage", "center", coverage["center"]),
        ("bias", "left", behavior["left"]),
        ("bias", "right", behavior["right"]),
        ("reachability", "short", reachability["short"]),
        ("reachability", "long", reachability["long"]),
    )

    manifest = []
    for ratio in ratios:
        for dimension, subtype, mask in conditions:
            variant, report = intervene(
                base,
                mask,
                ratio,
                rng,
                rounding="round",
                min_target=1,
                upsampling="resample",
            )
            prefix = {"coverage": "cov", "bias": "beh", "reachability": "reach"}[dimension]
            filename = f"buffer_{prefix}_{subtype}_r{ratio:g}_seed{seed}.pkl"
            relative_path = Path(dimension, filename)
            _save_buffer(variant, output_root / relative_path)
            manifest.append(
                {
                    "dimension": dimension,
                    "subtype": subtype,
                    "ratio": f"{ratio:g}",
                    "seed": seed,
                    **{key: report[key] for key in MANIFEST_FIELDS if key in report},
                    "realized_ratio": f"{report['realized_ratio']:.8g}",
                    "path": relative_path.as_posix(),
                }
            )
    _write_manifest(manifest, output_root)
    return manifest


def _parse_ratios(value):
    return tuple(float(item) for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled PointMaze-Medium-v3 trajectory-type datasets."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratios", type=_parse_ratios, default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-episodes", type=int, default=10000)
    parser.add_argument("--behavior-eps", type=float, default=1e-3)
    args = parser.parse_args()

    with args.input.open("rb") as handle:
        buffer = pickle.load(handle)
    build_variants(
        buffer,
        args.output_root,
        ratios=args.ratios,
        seed=args.seed,
        base_episodes=args.base_episodes,
        behavior_eps=args.behavior_eps,
    )


if __name__ == "__main__":
    main()
