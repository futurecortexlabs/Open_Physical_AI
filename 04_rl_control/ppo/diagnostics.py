"""Scripted mechanical test. Never called by training or used as demonstrations."""

import json
import torch


def ik_action(env, target, close):
    pos, axis = env.grip_pose()
    link = env.robot.get_link_transforms()[:, env.grip_link, :3]
    jac = env.robot.get_jacobians()[:, env.grip_link - 1, :, :6].clone()
    offset = (pos - link).unsqueeze(1).expand(-1, 6, -1)
    jac[:, :3] += torch.cross(jac[:, 3:].transpose(1, 2), offset, dim=-1).transpose(1, 2)
    dp = target - pos
    dp *= .03 / dp.norm(dim=-1, keepdim=True).clamp_min(.03)
    error = torch.cat((dp, .6 * torch.cross(axis, -env.z_axis, dim=-1)), -1)
    damping = torch.eye(6, device=env.device).unsqueeze(0) * .0064
    delta = .5 * (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + damping, error.unsqueeze(-1))).squeeze(-1)
    delta *= .035 / delta.abs().amax(-1, keepdim=True).clamp_min(.035)
    raw = torch.zeros((env.num_envs, 7), device=env.device)
    raw[:, :6] = torch.atanh((delta / .04).clamp(-.99, .99))
    raw[:, 6] = 4 if close else -4
    return raw


def mechanical_test(env, args):
    env.prepare_task()
    env.curriculum = 1.0
    env.reset()
    cube = env.cube.get_transforms()[:, :3].clone()
    goal = env.goal.clone()
    phases = [
        ("descend", cube + torch.tensor([0., 0., .005], device=env.device), False, 100),
        ("close", cube + torch.tensor([0., 0., .005], device=env.device), True, 90),
        ("lift", cube + torch.tensor([0., 0., .23], device=env.device), True, 100),
        ("transport", goal + torch.tensor([0., 0., .23], device=env.device), True, 100),
        ("lower", goal + torch.tensor([0., 0., .012], device=env.device), True, 100),
        ("release", goal + torch.tensor([0., 0., .012], device=env.device), False, 60),
        ("retreat", goal + torch.tensor([0., 0., .2], device=env.device), False, 100),
    ]
    recorder = None
    if args.record:
        from .recording import Recorder
        recorder = Recorder(env, args.record)
    report = {"kind": "SCRIPTED_MECHANICS_NOT_PPO", "seed": args.seed, "num_envs": env.num_envs, "phases": []}
    try:
        with torch.no_grad():
            for name, target, close, count in phases:
                for _ in range(count):
                    _, _, _, _, info = env.step(ik_action(env, target, close), reset_done=False)
                    if recorder:
                        recorder.capture(f"SCRIPTED MECHANICAL TEST - NOT PPO | {name}")
                row = {"phase": name, "cube": info["cube_pos"].cpu().tolist(),
                       "grip": env.grip_pose()[0].cpu().tolist(), "finger": env.robot.get_dof_positions()[:, env.finger].cpu().tolist(),
                       "lifted": info["ever_lifted"].cpu().tolist(), "success": info["success"].cpu().tolist()}
                report["phases"].append(row)
                print("PPO_MECHANICS " + json.dumps(row), flush=True)
    finally:
        if recorder:
            recorder.close()
    (args.run_dir / "mechanics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
