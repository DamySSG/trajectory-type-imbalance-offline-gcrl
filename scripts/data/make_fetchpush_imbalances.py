import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from scripts.data.fetchpush import (
    behavior_masks,
    coverage_masks,
    initial_geometry,
    reachability_masks,
)
from scripts.data.interventions import intervene

DEFAULT_RATIOS = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0)
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
    grip_slice=slice(0, 3),
    behavior_eps=1e-8,
    coverage_quantile=0.01,
    alpha=1.0,
    beta=1.0,
    gamma=0.5,
    reachability_threshold=None,
    max_total_multiplier=12.0,
):
    """Generate all FetchPush trajectory-type intervention datasets."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gripper, object_pos, goal = initial_geometry(buffer, grip_slice)
    behavior = behavior_masks(gripper, object_pos, goal, eps=behavior_eps)
    coverage, _ = coverage_masks(gripper, object_pos, goal, quantile=coverage_quantile)
    reachability, _ = reachability_masks(
        gripper,
        object_pos,
        goal,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold=reachability_threshold,
    )
    conditions = (
        ("bias", "left", behavior["left"]),
        ("bias", "right", behavior["right"]),
        ("coverage", "inner", coverage["inner"]),
        ("coverage", "outer", coverage["outer"]),
        ("reachability", "short", reachability["short"]),
        ("reachability", "long", reachability["long"]),
    )

    manifest = []
    for dimension, subtype, mask in conditions:
        for ratio in ratios:
            rounding = "ceil" if ratio >= 1.0 else "floor"
            variant, report = intervene(
                buffer,
                mask,
                ratio,
                np.random.default_rng(seed),
                rounding=rounding,
                min_target=0,
                upsampling="append",
                max_total_multiplier=max_total_multiplier,
            )
            filename = f"buffer_{dimension}_{subtype}_r{ratio:g}_seed{seed}.pkl"
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
        description="Generate controlled FetchPush-v1 trajectory-type datasets."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratios", type=_parse_ratios, default=DEFAULT_RATIOS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grip-slice", default="0:3")
    parser.add_argument("--behavior-eps", type=float, default=1e-8)
    parser.add_argument("--coverage-quantile", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--reachability-threshold", type=float)
    parser.add_argument("--max-total-multiplier", type=float, default=12.0)
    args = parser.parse_args()

    start, stop = (int(item) for item in args.grip_slice.split(":"))
    with args.input.open("rb") as handle:
        buffer = pickle.load(handle)
    build_variants(
        buffer,
        args.output_root,
        ratios=args.ratios,
        seed=args.seed,
        grip_slice=slice(start, stop),
        behavior_eps=args.behavior_eps,
        coverage_quantile=args.coverage_quantile,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        reachability_threshold=args.reachability_threshold,
        max_total_multiplier=args.max_total_multiplier,
    )


if __name__ == "__main__":
    main()
