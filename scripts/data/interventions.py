import math

import numpy as np

REQUIRED_KEYS = ("o", "u", "g", "ag")


def _episode_count(buffer):
    missing = [key for key in REQUIRED_KEYS if key not in buffer]
    if missing:
        raise KeyError(f"buffer is missing required keys: {', '.join(missing)}")
    count = int(np.asarray(buffer["o"]).shape[0])
    for key in REQUIRED_KEYS:
        if np.asarray(buffer[key]).shape[0] != count:
            raise ValueError(f"buffer key {key!r} has an inconsistent episode count")
    return count


def _select_episodes(buffer, indices, episode_count):
    selected = {}
    for key, value in buffer.items():
        array = np.asarray(value)
        selected[key] = (
            array[indices].copy() if array.shape[0] == episode_count else value
        )
    return selected


def _rounded_count(value, rounding):
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    if rounding == "round":
        return round(value)
    raise ValueError("rounding must be 'floor', 'ceil', or 'round'")


def intervene(
    buffer,
    target_mask,
    ratio,
    rng,
    *,
    rounding="floor",
    min_target=0,
    upsampling="append",
    max_total_multiplier=None,
):
    """Change one target subtype while retaining every non-target episode.

    ``ratio`` is the requested target count divided by its count in the input
    buffer. Downsampling is without replacement. With ``upsampling='append'``,
    all original target episodes are retained and additional target episodes
    are sampled with replacement. ``upsampling='resample'`` samples the full
    enlarged target set with replacement, matching the PointMaze source script.
    """
    episode_count = _episode_count(buffer)
    mask = np.asarray(target_mask, dtype=bool)
    if mask.shape != (episode_count,):
        raise ValueError("target_mask must contain one Boolean per episode")
    if ratio < 0:
        raise ValueError("ratio must be non-negative")

    target_indices = np.flatnonzero(mask)
    other_indices = np.flatnonzero(~mask)
    if not len(target_indices) or not len(other_indices):
        raise ValueError("both target and non-target subsets must be non-empty")

    requested = max(min_target, _rounded_count(len(target_indices) * ratio, rounding))
    if max_total_multiplier is not None:
        total_multiplier = (requested + len(other_indices)) / episode_count
        if total_multiplier > max_total_multiplier:
            raise ValueError(
                f"requested dataset multiplier {total_multiplier:.3f} exceeds "
                f"the configured maximum {max_total_multiplier:.3f}"
            )

    if requested < len(target_indices):
        selected_targets = rng.choice(target_indices, size=requested, replace=False)
    elif requested == len(target_indices):
        selected_targets = target_indices.copy()
    elif upsampling == "append":
        extra = rng.choice(
            target_indices,
            size=requested - len(target_indices),
            replace=True,
        )
        selected_targets = np.concatenate([target_indices, extra])
    elif upsampling == "resample":
        selected_targets = rng.choice(target_indices, size=requested, replace=True)
    else:
        raise ValueError("upsampling must be 'append' or 'resample'")

    selected_indices = np.concatenate([selected_targets, other_indices])
    selected_indices = rng.permutation(selected_indices)
    result = _select_episodes(buffer, selected_indices, episode_count)
    report = {
        "episodes_before": episode_count,
        "episodes_after": len(selected_indices),
        "target_before": len(target_indices),
        "target_after": len(selected_targets),
        "other_before": len(other_indices),
        "other_after": len(other_indices),
        "requested_ratio": float(ratio),
        "realized_ratio": len(selected_targets) / len(target_indices),
    }
    return result, report
