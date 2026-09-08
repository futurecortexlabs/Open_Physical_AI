"""Low-level force budgeting, independent of task goals and PPO actions."""

import torch


def gravity_force_budget(gravity, limits):
    # Fixed 50/50 allocation permits setting drive limits once. Updating drive
    # properties every step would require host transfers on the PhysX GPU backend.
    drive_limits = limits.clone()
    drive_limits[:, :6] *= .5
    feedforward = torch.zeros_like(gravity)
    feedforward[:, :6] = gravity[:, :6].clamp(-drive_limits[:, :6], drive_limits[:, :6])
    # Conservative bound: |feed-forward + drive torque| cannot exceed USD limit.
    return feedforward, drive_limits
