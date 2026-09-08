"""Goal-conditioned task contract, command parsing, sampling, and audit geometry.

Commands specify destinations, never robot trajectories or gripper actions.
This module is stdlib-only except when sampling simulator tensors.
"""

import math
import re
import unicodedata

from .goals import right_offset_xy


MIN_DISTANCE = .05
MAX_DISTANCE = .30
WORKSPACE = ((.35, 1.10), (-.40, .40))
SUITE_VERSION = "directions_distances_v1"
# 16 balanced conditions. A single batch evaluates ONE policy on every case.
CASES = (("right_05", .05, 0.), ("right_10", .10, 0.), ("right_20", .20, 0.), ("right_30", .30, 0.),
         ("left_05", -.05, 0.), ("left_10", -.10, 0.), ("left_20", -.20, 0.), ("left_30", -.30, 0.),
         ("away_10", 0., .10), ("away_20", 0., .20), ("away_30", 0., .30),
         ("toward_10", 0., -.10), ("toward_20", 0., -.20), ("toward_30", 0., -.30),
         ("right15_away10", .15, .10), ("left15_toward10", -.15, -.10))


def contract(mode):
    if mode != "multi_goal":
        return None
    return {"version": 1, "frame": "fixed_overview_camera_floor", "goal_z": .03,
            "distance_m": [MIN_DISTANCE, MAX_DISTANCE], "workspace_xy_m": [list(axis) for axis in WORKSPACE],
            "training_distribution": "uniform_radius_and_angle"}


def check_relative(right, away):
    if not all(math.isfinite(v) for v in (right, away)):
        raise ValueError("指示の距離は有限の数値にしてください")
    if not MIN_DISTANCE - 1e-7 <= math.hypot(right, away) <= MAX_DISTANCE + 1e-7:
        raise ValueError("対応する移動距離は床面上で5〜30cmです。範囲外を丸めて実行しません")


def relative_request(right, away):
    right, away = float(right), float(away)
    check_relative(right, away)
    return {"kind": "relative", "right_m": right, "away_m": away}


def absolute_request(x, y):
    if not all(math.isfinite(v) and lo <= v <= hi for v, (lo, hi) in zip((x, y), WORKSPACE)):
        raise ValueError("XY座標が対応する作業範囲外です（x=0.35〜1.10m、y=-0.40〜0.40m）")
    return {"kind": "absolute", "x": float(x), "y": float(y)}


def parse_instruction(instruction):
    """Deliberately bounded Japanese grammar; never guess unknown instructions."""
    text = unicodedata.normalize("NFKC", instruction).lower().strip()
    text = re.sub(r"(?:へ|に)?(?:運んで|搬送して|移動して|置いて)$", "", text)
    text = re.sub(r"[\s、,]", "", text)
    pattern = re.compile(r"(手前|右|左|奥)(?:に|へ)?([0-9]+(?:\.[0-9]+)?)(センチメートル|センチ|cm|メートル|m)")
    position, right, away, axes = 0, 0., 0., set()
    for match in pattern.finditer(text):
        if match.start() != position:
            raise ValueError("解釈できない指示です。例：右20cm、左10cm、右15cm 奥10cm")
        direction, number, unit = match.groups()
        axis = "right" if direction in ("右", "左") else "away"
        if axis in axes:
            raise ValueError("同じ軸の指示を重複・矛盾させないでください")
        axes.add(axis)
        distance = float(number) * (.01 if unit in ("センチメートル", "センチ", "cm") else 1.)
        if distance <= 0:
            raise ValueError("指示する距離は正の値にしてください")
        if axis == "right":
            right = distance if direction == "右" else -distance
        else:
            away = distance if direction == "奥" else -distance
        position = match.end()
    if not axes or position != len(text):
        raise ValueError("解釈できない指示です。右・左・奥・手前と距離を指定してください")
    return relative_request(right, away)


def world_offset(right, away):
    rx, ry = (v / .1 for v in right_offset_xy())
    return right * rx - away * ry, right * ry + away * rx


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("Missing goal request")
    kind = request.get("kind")
    if kind == "random" and request == {"kind": "random"}:
        return
    if kind == "suite" and request == {"kind": "suite", "version": SUITE_VERSION}:
        return
    if kind == "relative" and set(request) == {"kind", "right_m", "away_m"}:
        check_relative(request["right_m"], request["away_m"])
        return
    if kind == "absolute" and set(request) == {"kind", "x", "y"}:
        absolute_request(request["x"], request["y"])
        return
    raise ValueError("Unknown or malformed goal request")


def case_name(request, index):
    return CASES[index % len(CASES)][0] if request.get("kind") == "suite" else request["kind"]


def expected_xy(cube, request, index):
    kind = request["kind"]
    if kind == "random":
        return None
    if kind == "absolute":
        return request["x"], request["y"]
    if kind == "suite":
        _, right, away = CASES[index % len(CASES)]
    else:
        right, away = request["right_m"], request["away_m"]
    dx, dy = world_offset(right, away)
    return cube[0] + dx, cube[1] + dy


def sample_goals(cube, ids, rng, request):
    import torch
    validate_request(request)
    # Consume the same random numbers for every request. Matched command trials
    # at the same seed retain identical object/joint initial conditions.
    random = torch.rand((len(cube), 2), generator=rng, device=cube.device)
    goal = cube.clone()
    kind = request["kind"]
    if kind == "random":
        radius = MIN_DISTANCE + (MAX_DISTANCE - MIN_DISTANCE) * random[:, 0]
        angle = 2 * math.pi * random[:, 1]
        goal[:, :2] += torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
    elif kind == "absolute":
        goal[:, :2] = cube.new_tensor((request["x"], request["y"]))
    elif kind == "relative":
        goal[:, :2] += cube.new_tensor(world_offset(request["right_m"], request["away_m"]))
    else:
        offsets = cube.new_tensor([world_offset(r, a) for _, r, a in CASES])
        goal[:, :2] += offsets[ids.long() % len(CASES)]
    goal[:, 2] = .03
    distance = (goal[:, :2] - cube[:, :2]).norm(dim=-1)
    valid = (distance >= MIN_DISTANCE - 1e-6) & (distance <= MAX_DISTANCE + 1e-6)
    for axis, (lo, hi) in enumerate(WORKSPACE):
        valid &= (goal[:, axis] >= lo - 1e-6) & (goal[:, axis] <= hi + 1e-6)
    if not bool(valid.all()):
        raise ValueError("指示位置が初期位置から5〜30cmの範囲外、または作業範囲外です。目標を勝手に補正しません")
    return goal


def validate_geometry(report):
    if report.get("goal_contract") != contract("multi_goal"):
        raise ValueError("Multi-goal task contract differs")
    request = report.get("goal_request")
    validate_request(request)
    count = report["requested_episodes"]
    if request["kind"] == "suite" and count % len(CASES):
        raise ValueError("Unbalanced command suite")
    initial = report["initial_conditions"]
    cubes, goals = initial.get("cube_position", []), initial.get("goal", [])
    if len(cubes) != count or len(goals) != count:
        raise ValueError("Missing multi-goal geometry")
    for index, (cube, goal) in enumerate(zip(cubes, goals)):
        if len(cube) != 3 or len(goal) != 3 or not all(math.isfinite(v) for v in (*cube, *goal)):
            raise ValueError("Invalid multi-goal coordinates")
        distance = math.hypot(goal[0] - cube[0], goal[1] - cube[1])
        if not MIN_DISTANCE - 1e-5 <= distance <= MAX_DISTANCE + 1e-5 or abs(goal[2] - .03) > 1e-5:
            raise ValueError("Goal outside the supported distance/floor range")
        if any(not lo - 1e-5 <= goal[i] <= hi + 1e-5 for i, (lo, hi) in enumerate(WORKSPACE)):
            raise ValueError("Goal outside workspace")
        expected = expected_xy(cube, request, index)
        if expected is not None and any(abs(a - b) > 1e-5 for a, b in zip(goal[:2], expected)):
            raise ValueError("Executed goal differs from instruction")
    for episode in report["episodes"]:
        if episode.get("goal_case") != case_name(request, episode["environment"]):
            raise ValueError("Episode goal case is missing or mislabeled")
