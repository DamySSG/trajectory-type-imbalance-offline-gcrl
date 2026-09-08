import numpy as np


def behavior_masks(achieved_goals, eps=1e-3):
    """Classify left/right motion using net horizontal displacement."""
    horizontal_displacement = (
        achieved_goals[:, -1, 0] - achieved_goals[:, 0, 0]
    )
    left = horizontal_displacement < -eps
    right = horizontal_displacement > eps
    return {
        "left": left,
        "right": right,
        "neutral": ~(left | right),
    }


def path_lengths(achieved_goals):
    """Return the sum of Euclidean step lengths for each trajectory."""
    displacement = np.diff(achieved_goals, axis=1)
    return np.linalg.norm(displacement, axis=-1).sum(axis=1)


def coverage_masks(achieved_goals, quartiles=None):
    """Classify initial positions into four corner regions and a central region."""
    starts = achieved_goals[:, 0, :2]
    x_coordinate = starts[:, 0]
    y_coordinate = starts[:, 1]
    if quartiles is None:
        x_lower, x_upper = np.quantile(x_coordinate, [0.25, 0.75])
        y_lower, y_upper = np.quantile(y_coordinate, [0.25, 0.75])
    else:
        x_lower, x_upper, y_lower, y_upper = quartiles

    top_left = (x_coordinate <= x_lower) & (y_coordinate >= y_upper)
    top_right = (x_coordinate >= x_upper) & (y_coordinate >= y_upper)
    bottom_left = (x_coordinate <= x_lower) & (y_coordinate <= y_lower)
    bottom_right = (x_coordinate >= x_upper) & (y_coordinate <= y_lower)
    center = (
        (x_coordinate >= x_lower)
        & (x_coordinate <= x_upper)
        & (y_coordinate >= y_lower)
        & (y_coordinate <= y_upper)
    )
    thresholds = np.array([x_lower, x_upper, y_lower, y_upper])
    return (
        {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right,
            "center": center,
        },
        {"quartiles": thresholds},
    )


def reachability_masks(achieved_goals, thresholds=None):
    """Classify trajectories using lower and upper path-length tertiles."""
    lengths = path_lengths(achieved_goals)
    if thresholds is None:
        lower, upper = np.quantile(lengths, [1.0 / 3.0, 2.0 / 3.0])
    else:
        lower, upper = thresholds
    short = lengths <= lower
    long = lengths >= upper
    middle = (lengths > lower) & (lengths < upper)
    return (
        {"short": short, "middle": middle, "long": long},
        {"lengths": lengths, "thresholds": np.array([lower, upper])},
    )
