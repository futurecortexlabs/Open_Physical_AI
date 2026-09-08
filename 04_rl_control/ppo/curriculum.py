"""Training-only task curriculum. Never supplies policy actions or demonstrations."""

import torch


def next_stage(stage, difficulty, completed, objective_rate, threshold=.75, elapsed_steps=None, increment=.25):
    """Advance only after enough completed episodes at the current task level."""
    # Fast successes must not hide unfinished failures during a new level's first
    # rollout. Wait at least one complete episode-length window before promotion.
    if elapsed_steps is not None and elapsed_steps < (180 if stage == "lift" else 450):
        return stage, difficulty
    if completed < 512 or objective_rate < threshold:
        return stage, difficulty
    if difficulty < 1.0:
        return stage, round(min(1.0, difficulty + increment), 8)
    if stage == "lift":
        return "full", 0.0
    return stage, difficulty


def lift_completion(metrics, velocity, previous_count):
    # Require a sustained, slow lift, not a single airborne frame.
    held = metrics["lifted_now"] & (metrics["grip_distance"] < .07)
    held &= velocity[:, :3].norm(dim=-1) < .15
    counter = torch.where(held, previous_count + 1, torch.zeros_like(previous_count))
    return counter >= 10, counter


def staged_potential(stage, metrics, cube_pos, finger, velocity=None):
    reach = 1 - torch.tanh(8 * metrics["grip_distance"])
    close = (finger / .55).clamp(0, 1)
    enclosure = reach * close * (metrics["grip_distance"] < .06)
    height = ((cube_pos[:, 2] - .03) / .10).clamp(0, 1)
    lift = height * (metrics["grip_distance"] < .10)
    if stage == "lift":
        if velocity is not None:
            # v4: an airborne cube moving slowly receives a higher state value.
            # This is still a potential difference, not a reward paid per hold frame.
            slow_hold = lift * torch.exp(-4 * velocity[:, :3].norm(dim=-1))
            return 20 * (.5 * reach + .25 * enclosure + lift + slow_hold)
        return 20 * (.5 * reach + .25 * enclosure + 2 * lift)
    goal = (1 - torch.tanh(5 * metrics["goal_distance"])) * metrics["ever_lifted"]
    near_goal = 1 - torch.tanh(12 * metrics["goal_distance"])
    on_floor = torch.exp(-((cube_pos[:, 2] - .03) / .035).square())
    release = near_goal * metrics["ever_lifted"] * on_floor * (1 - close)
    retreat = release * (metrics["grip_distance"] / .12).clamp(0, 1)
    return 20 * (.3 * reach + .3 * enclosure + lift + 3 * goal + release + retreat)


def precision_potential(stage, metrics, cube_pos, finger, velocity):
    """v5: avoid closing above the cube, without issuing a gripper action.

    The extra state potential disappears in the air. Near the cube the existing
    enclosure potential still rewards closing; final success is unchanged.
    """
    base = staged_potential(stage, metrics, cube_pos, finger, velocity)
    near_grasp = torch.sigmoid(160 * (.045 - metrics["grip_distance"]))
    on_floor = torch.exp(-((cube_pos[:, 2] - .03) / .03).square())
    open_finger = 1 - (finger / .55).clamp(0, 1)
    return base + 15 * on_floor * (1 - near_grasp) * open_finger
