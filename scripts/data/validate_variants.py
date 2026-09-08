import argparse
import csv
import json
from pathlib import Path
import pickle

import numpy as np


REQUIRED_KEYS = ("o", "u", "g", "ag")


def validate_manifest(output_root):
    """Check that every manifest entry exists and has the reported dimensions."""
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "manifest.csv"
    errors = []
    checked = 0
    if not manifest_path.is_file():
        return {"checked": 0, "errors": [f"missing manifest: {manifest_path}"]}

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for line_number, row in enumerate(rows, start=2):
        variant_path = (output_root / row["path"]).resolve()
        if output_root != variant_path and output_root not in variant_path.parents:
            errors.append(f"line {line_number}: path escapes output root")
            continue
        if not variant_path.is_file():
            errors.append(f"line {line_number}: missing variant {row['path']}")
            continue
        try:
            with variant_path.open("rb") as handle:
                buffer = pickle.load(handle)
        except Exception as error:
            errors.append(f"line {line_number}: cannot read {row['path']}: {error}")
            continue

        missing = [key for key in REQUIRED_KEYS if key not in buffer]
        if missing:
            errors.append(
                f"line {line_number}: {row['path']} missing keys {', '.join(missing)}"
            )
            continue
        episode_count = int(np.asarray(buffer["o"]).shape[0])
        expected_count = int(row["episodes_after"])
        if episode_count != expected_count:
            errors.append(
                f"line {line_number}: {row['path']} has {episode_count} episodes; "
                f"manifest reports {expected_count}"
            )
            continue
        inconsistent = [
            key
            for key in REQUIRED_KEYS
            if np.asarray(buffer[key]).shape[0] != episode_count
        ]
        if inconsistent:
            errors.append(
                f"line {line_number}: inconsistent first dimension for "
                f"{', '.join(inconsistent)}"
            )
            continue
        if int(row["other_before"]) != int(row["other_after"]):
            errors.append(f"line {line_number}: non-target count changed")
            continue
        checked += 1
    return {"checked": checked, "errors": errors}


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated trajectory-type dataset files."
    )
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    report = validate_manifest(args.output_root)
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
