"""Task metrics shared by training and evaluation; no motion/state-machine control."""

import torch


def task_metrics(cube_pos, cube_velocity, grip_pos, finger, goal, ever_lifted, stable_steps):
    grip_distance = (cube_pos - grip_pos).norm(dim=-1)
    goal_distance = (cube_pos - goal).norm(dim=-1)
    lifted_now = (cube_pos[:, 2] > 0.10) & (grip_distance < 0.09) & (finger > 0.1)
    lifted = ever_lifted | lifted_now
    at_goal = goal_distance < 0.035
    quiet = (cube_velocity[:, :3].norm(dim=-1) < 0.05) & (cube_velocity[:, 3:].norm(dim=-1) < 0.5)
    released = (finger < 0.12) & (grip_distance > 0.09)
    stable = lifted & at_goal & quiet & released
    count = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))
    return {"grip_distance": grip_distance, "goal_distance": goal_distance,
            "lifted_now": lifted_now, "ever_lifted": lifted, "at_goal": at_goal,
            "quiet": quiet, "released": released, "stable_steps": count, "success": count >= 10}


def reward_potential(metrics, cube_pos, finger):
    """State potential, NOT a reward for indefinitely holding a good state."""
    reach = 1 - torch.tanh(8 * metrics["grip_distance"])
    close = (finger / 0.55).clamp(0, 1)
    enclosure = reach * close * (metrics["grip_distance"] < 0.06)
    height = ((cube_pos[:, 2] - 0.03) / 0.10).clamp(0, 1)
    lift = height * (metrics["grip_distance"] < 0.10)
    goal = (1 - torch.tanh(5 * metrics["goal_distance"])) * metrics["ever_lifted"]
    release = metrics["at_goal"] * metrics["ever_lifted"] * (1 - close)
    return 20 * (0.3 * reach + 0.3 * enclosure + 1.0 * lift + 3.0 * goal + release)


def settling_potential(stable_steps, required_steps):
    """Additional state value for continuous settling, never a per-frame bonus."""
    if required_steps <= 10:
        return torch.zeros_like(stable_steps, dtype=torch.float32)
    return 20 * ((stable_steps.float() - 9) / (required_steps - 9)).clamp(0, 1)


def completion_transition(stable_steps, started, required_steps, retry_mode):
    """Optional training alignment with first-success, no-retry evaluation."""
    failed = started & (stable_steps < 10) if retry_mode == "fail" else torch.zeros_like(started)
    return (stable_steps >= required_steps) & ~failed, failed, started | (stable_steps >= 10)


def transition_reward(previous_potential, potential, success, failure, tool_z, actions, previous_actions, gamma=.995,
                      success_bonus=100., reward_scale=1.):
    """Potential-based shaping; absorbing terminal states have zero potential.

    Unlike a positive reward every frame for carrying the cube, this does not
    incentivize holding it forever instead of finishing the release task.
    """
    terminal = success | failure
    following = torch.where(terminal, torch.zeros_like(potential), potential)
    orientation = ((tool_z[:, 2] + 1) * 0.5).clamp(0, 1)
    cost = .002 * actions.square().sum(-1) + .005 * (actions - previous_actions).square().sum(-1)
    return reward_scale * (gamma * following - previous_potential + success_bonus * success
                           - 5 * failure - .02 * orientation - cost - .02)
