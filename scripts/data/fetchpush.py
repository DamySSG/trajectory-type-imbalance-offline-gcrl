import numpy as np


def initial_geometry(buffer, grip_slice=slice(0, 3)):
    """Extract initial planar gripper, object, and desired-goal positions."""
    observations = np.asarray(buffer["o"])
    achieved_goals = np.asarray(buffer["ag"])
    desired_goals = np.asarray(buffer["g"])
    gripper = observations[:, 0, grip_slice][:, :2]
    object_pos = achieved_goals[:, 0, :2]
    goal = desired_goals[:, 0, :2]
    return gripper, object_pos, goal


def behavior_masks(gripper, object_pos, goal, eps=1e-8):
    """Classify left/right contact geometry using a signed planar cross product."""
    gripper_to_object = object_pos - gripper
    object_to_goal = goal - object_pos
    first_norm = np.linalg.norm(gripper_to_object, axis=1)
    second_norm = np.linalg.norm(object_to_goal, axis=1)
    valid = (first_norm > eps) & (second_norm > eps)
    cross = (
        gripper_to_object[:, 0] * object_to_goal[:, 1]
        - gripper_to_object[:, 1] * object_to_goal[:, 0]
    )
    left = (cross > eps) & valid
    right = (cross < -eps) & valid
    return {
        "left": left,
        "right": right,
        "neutral": valid & ~(left | right),
    }


def coverage_masks(gripper, object_pos, goal, *, quantile=0.01):
    """Classify desired goals relative to the inferred central exclusion disk."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    center = np.mean(gripper, axis=0)
    initial_object_distance = np.linalg.norm(object_pos - gripper, axis=1)
    radius = float(np.quantile(initial_object_distance, quantile))
    goal_distance = np.linalg.norm(goal - center[None, :], axis=1)
    inner = goal_distance < radius
    return (
        {"inner": inner, "outer": ~inner},
        {
            "center": center,
            "radius": radius,
            "quantile": float(quantile),
        },
    )


def reachability_masks(
    gripper,
    object_pos,
    goal,
    *,
    alpha=1.0,
    beta=1.0,
    gamma=0.5,
    eps=1e-9,
    threshold=None,
):
    """Classify trajectories by the initial weighted geometric effort score."""
    gripper_to_object = object_pos - gripper
    object_to_goal = goal - object_pos
    gripper_object_distance = np.linalg.norm(gripper_to_object, axis=1)
    object_goal_distance = np.linalg.norm(object_to_goal, axis=1)
    dot_product = np.sum(gripper_to_object * object_to_goal, axis=1)
    cosine = dot_product / (gripper_object_distance * object_goal_distance + eps)
    misalignment = 1.0 - np.clip(cosine, -1.0, 1.0)
    scores = (
        alpha * gripper_object_distance
        + beta * object_goal_distance
        + gamma * misalignment
    )
    split = float(np.median(scores) if threshold is None else threshold)
    short = scores <= split
    return (
        {"short": short, "long": ~short},
        {"scores": scores, "threshold": split},
    )
