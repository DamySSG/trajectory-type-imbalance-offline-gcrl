import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

METHODS = ("ddpg", "gcsl", "wgcsl", "gofar")
RATIOS = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0)
CONDITIONS = {
    "FetchPush": {
        "bias": ("left", "right"),
        "coverage": ("inner", "outer"),
        "reachability": ("short", "long"),
    },
    "PointMaze": {
        "bias": ("left", "right"),
        "coverage": ("tl", "tr", "bl", "br", "center"),
        "reachability": ("short", "long"),
    },
}


@dataclass(frozen=True)
class Experiment:
    environment: str
    dimension: str
    subtype: str
    ratio: float
    method: str
    seed: int
    data_root: Path


def _ratios_for(environment, dimension):
    if environment == "PointMaze" and dimension == "reachability":
        return RATIOS[1:]
    return RATIOS


def build_matrix(data_root, seed=0):
    """Return the 532 environment-condition-method runs reported in the paper."""
    runs = []
    for environment, dimensions in CONDITIONS.items():
        for method in METHODS:
            for dimension, subtypes in dimensions.items():
                for subtype in subtypes:
                    for ratio in _ratios_for(environment, dimension):
                        runs.append(
                            Experiment(
                                environment=environment,
                                dimension=dimension,
                                subtype=subtype,
                                ratio=ratio,
                                method=method,
                                seed=seed,
                                data_root=Path(data_root),
                            )
                        )
    return runs


def _buffer_filename(run):
    if run.environment == "FetchPush":
        prefix = run.dimension
    else:
        prefix = {
            "bias": "beh",
            "coverage": "cov",
            "reachability": "reach",
        }[run.dimension]
    return f"buffer_{prefix}_{run.subtype}_r{run.ratio:g}_seed{run.seed}.pkl"


def _buffer_path(run, source):
    environment_dir = {
        "FetchPush": "fetchpush",
        "PointMaze": "pointmaze",
    }[run.environment]
    return (
        run.data_root / environment_dir / source / run.dimension / _buffer_filename(run)
    )


def command_for(run, python_executable="python"):
    """Render one fully specified training command."""
    command = [
        python_executable,
        "train.py",
        "--env",
        run.environment,
        "--n-epochs",
        "800",
        "--seed",
        str(run.seed),
        "--expert_percent",
        "0.5",
        "--random_percent",
        "0.5",
        "--method",
        run.method,
        "--use-mixture",
        "--buffer-expert",
        _buffer_path(run, "expert").as_posix(),
        "--buffer-random",
        _buffer_path(run, "random").as_posix(),
        "--imbalance-type",
        run.dimension,
        "--imbalance-target",
        run.subtype,
        "--imbalance-ratio",
        f"{run.ratio:g}",
    ]
    if run.method == "ddpg":
        command.extend(
            ["--relabel", "true", "--relabel_percent", "0.5", "--reward_type", "binary"]
        )
    elif run.method in ("gcsl", "wgcsl"):
        command.extend(
            ["--relabel", "true", "--relabel_percent", "1.0", "--reward_type", "binary"]
        )
    else:
        command.extend(
            [
                "--relabel",
                "false",
                "--relabel_percent",
                "0.0",
                "--reward_type",
                "disc",
                "--disc_iter",
                "20",
                "--disc_lambda",
                "0.01",
            ]
        )
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Generate or execute the paper's complete experiment matrix."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default="python")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    runs = build_matrix(args.data_root, seed=args.seed)
    records = []
    for index, run in enumerate(runs, start=1):
        command = command_for(run, python_executable=args.python)
        record = {
            "index": index,
            "environment": run.environment,
            "dimension": run.dimension,
            "subtype": run.subtype,
            "ratio": run.ratio,
            "method": run.method,
            "seed": run.seed,
            "command": shlex.join(command),
        }
        records.append(record)
        if args.execute:
            subprocess.run(command, check=True)

    output = "\n".join(json.dumps(record) for record in records) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
