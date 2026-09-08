"""Training-only expansion of destination sampling; never generates actions.

The final task contract and every evaluation retain the full 5--30 cm range.
Start-height/noise expansion waits until the destination range is complete.
"""

import math

MODE = "adaptive_sector_v1"
MODES = ("none", MODE)
BINS = 8


def bounds(level):
    if not math.isfinite(level) or not 0 <= level <= 1:
        raise ValueError("Goal curriculum level must be in [0, 1]")
    return {"radius_min": .065 + (.05 - .065) * level,
            "radius_max": .080 + (.30 - .080) * level,
            "center_angle": math.atan2(.07, -.02),
            "half_angle": .10 + (math.pi - .10) * level}


def sample(cube, rng, level):
    import torch
    region = bounds(level)
    random = torch.rand((len(cube), 2), device=cube.device, generator=rng)
    radius = region["radius_min"] + (region["radius_max"] - region["radius_min"]) * random[:, 0]
    angle = region["center_angle"] + (2 * random[:, 1] - 1) * region["half_angle"]
    goal = cube.clone()
    goal[:, :2] += torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
    goal[:, 2] = .03
    # Two radial bands by four angular sectors. Count all completed episodes,
    # including failed/timeout trials, so easy destinations cannot mask others.
    bins = (random[:, 0] * 2).long() * 4 + (random[:, 1] * 4).long()
    return goal, bins


def progress(history, elapsed_steps, level, start_difficulty):
    """Records contain strict 70-step success at index 4 and goal bin at 5."""
    counts, successes = [0] * BINS, [0] * BINS
    for record in history:
        identity = int(record[5])
        if not 0 <= identity < BINS:
            raise ValueError("Missing curriculum destination-bin evidence")
        counts[identity] += 1
        successes[identity] += int(record[4])
    rates = [s / n if n else 0. for s, n in zip(successes, counts)]
    overall = sum(successes) / max(1, sum(counts))
    ready = (elapsed_steps >= 450 and len(history) >= 512 and min(counts) >= 32
             and overall >= .70 and min(rates) >= .60)
    next_level, next_start = level, start_difficulty
    if ready:
        if level < 1:
            next_level = round(min(1., level + .05), 8)
        elif start_difficulty < 1:
            next_start = round(min(1., start_difficulty + .05), 8)
    return next_level, next_start, {"bin_counts": counts, "bin_successes": successes,
                                   "bin_rates": rates, "overall_rate": overall, "ready": ready,
                                   "elapsed_steps_at_level": elapsed_steps, "completed_episodes": len(history),
                                   "full_episode_window_observed": elapsed_steps >= 450,
                                   "enough_completed_episodes": len(history) >= 512,
                                   "overall_rate_met": overall >= .70,
                                   "under_sampled_bins": [i for i, n in enumerate(counts) if n < 32],
                                   "low_success_bins": [i for i, (n, rate) in enumerate(zip(counts, rates))
                                                        if n >= 32 and rate < .60]}


def state(env):
    mode = getattr(env, "goal_curriculum_mode", "none")
    if mode == "none":
        return None
    level = getattr(env, "goal_curriculum_level", 0.)
    return {"mode": mode, "level": level, "sampling_region": bounds(level),
            "changes_training_destinations_only": True, "evaluation_range_restricted": False}
