"""Explicit target geometry. These helpers never generate robot actions."""

import math


CAMERA_EYE = (2.4, 2.2, 2.0)
CAMERA_TARGET = (.6, 0., .45)
GOAL_MODES = ("curriculum", "right_10cm", "multi_goal")


def right_offset_xy():
    """10 cm on the floor, to the RIGHT of the fixed overview camera.

    Camera right is forward cross world-up, not world +X. Share the camera
    definition with the recorder so 'right' is independent of guesswork.
    """
    forward_x = CAMERA_TARGET[0] - CAMERA_EYE[0]
    forward_y = CAMERA_TARGET[1] - CAMERA_EYE[1]
    length = math.hypot(forward_x, forward_y)
    return (.10 * forward_y / length, -.10 * forward_x / length)


def right_goal(initial_cube_position):
    """Return a fixed per-episode target, not a view that follows the cube."""
    goal = initial_cube_position.clone()
    goal[..., :2] += initial_cube_position.new_tensor(right_offset_xy())
    goal[..., 2] = .03  # Resting cube CENTER, not the floor surface or reset height.
    return goal


def goal_label(mode):
    if mode == "multi_goal":
        return "床面上の指示位置：初期位置から5〜30cm、左右・奥・手前"
    if mode == "right_10cm":
        return "物体の初期位置から動画の右へ10cm（床面上）"
    if mode == "curriculum":
        return "従来のカリキュラム目標"
    raise ValueError("Unknown goal mode: " + str(mode))
