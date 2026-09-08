"""Batched PhysX environment; observations come from physics tensors, not USD poses."""

from __future__ import annotations

import time
import torch
import omni.usd
import omni.timeline
import omni.physics.tensors as tensors
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux
from isaacsim.core.cloner import GridCloner
from isaacsim.core.simulation_manager import SimulationManager
from .task import task_metrics, reward_potential, transition_reward, settling_potential, completion_transition
from .curriculum import lift_completion, staged_potential, precision_potential
from .goals import right_goal
from .multigoal import sample_goals
from .goal_curriculum import sample as sample_curriculum_goals
from .control import gravity_force_budget


ROBOT_SOURCE = "/World/crx20ia_l"
GRIP_SUFFIX = "/J6_link/flange/ee_link/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/grip_frame"


def quat_apply(q, v):
    """Rotate vectors by PhysX xyzw quaternions."""
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def quat_mul(a, b):
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + torch.cross(a[..., :3], b[..., :3], dim=-1)
    w = a[..., 3:] * b[..., 3:] - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)
    return torch.cat((xyz, w), -1)


class PickPlaceEnvironment:
    def __init__(self, app, args, physics_guard=None):
        self.app, self.args = app, args
        self.physics_guard = physics_guard
        self.device = torch.device(args.device)
        self.num_envs = args.num_envs
        self.dt = 1 / args.physics_hz
        self.decimation = args.physics_hz // 30
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.Xform.Define(self.stage, "/World")
        self._setup_scene()
        omni.timeline.get_timeline_interface().play()
        for _ in range(5):
            app.update()
        if self.physics_guard:
            self.physics_guard.check()
        self.view = tensors.create_simulation_view("torch", stage_id=omni.usd.get_context().get_stage_id())
        self.view.set_subspace_roots("/World/envs/*")
        self.robot = self.view.create_articulation_view("/World/envs/*/Robot" + self.articulation_suffix)
        self.cube = self.view.create_rigid_body_view("/World/envs/*/Cube")
        if self.robot.count != self.num_envs or self.cube.count != self.num_envs:
            raise RuntimeError(f"Clone count mismatch: robot={self.robot.count}, cube={self.cube.count}, requested={self.num_envs}")
        self.ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.names = list(self.robot.shared_metatype.dof_names)
        self.links = list(self.robot.shared_metatype.link_names)
        self.finger = self.names.index("finger_joint")
        self.arm = [self.names.index(f"J{i}") for i in range(1, 7)]
        if self.arm != list(range(6)):
            raise ValueError("This task requires J1..J6 as the first six articulation DOFs")
        # Exact suffix avoids accidentally selecting the camera's similarly named base.
        link_paths = list(self.robot.link_paths[0])
        base_suffix = GRIP_SUFFIX.rsplit("/", 1)[0]
        self.grip_link = next(i for i, path in enumerate(link_paths) if str(path).endswith(base_suffix))
        grip_prim = self.stage.GetPrimAtPath("/World/envs/env_0/Robot" + GRIP_SUFFIX)
        local = UsdGeom.Xformable(grip_prim).GetLocalTransformation()
        p, q = local.ExtractTranslation(), local.ExtractRotationQuat()
        self.grip_offset = torch.tensor(list(p), dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.grip_rotation = torch.tensor([*q.GetImaginary(), q.GetReal()], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.z_axis = torch.tensor([0., 0., 1.], device=self.device).repeat(self.num_envs, 1)
        print("PPO_JOINTS", self.names, "PPO_LINKS", self.links, flush=True)
        self.initial_q = self.robot.get_dof_positions().clone()
        self.drive_force_limits = self.robot.get_dof_max_forces().clone().to(self.device)
        if args.gravity_compensation:
            _, drive_budget = gravity_force_budget(torch.zeros_like(self.initial_q), self.drive_force_limits)
            # Property APIs use host buffers, unlike the per-step actuation API.
            self.robot.set_dof_max_forces(drive_budget.cpu(), self.ids.cpu())
        if args.gripper_max_velocity is not None:
            velocity_limits = self.robot.get_dof_max_velocities().clone().cpu()
            velocity_limits[:, self.finger] = args.gripper_max_velocity
            self.robot.set_dof_max_velocities(velocity_limits, self.ids.cpu())
        self.initial_pose = self.cube.get_transforms().clone()
        self.targets = self.initial_q.clone()
        self.zero_vel = torch.zeros_like(self.initial_q)
        self.zero_cube_vel = torch.zeros((self.num_envs, 6), device=self.device)
        limits = self.robot.get_dof_limits().to(self.device)
        self.lower, self.upper = limits[..., 0], limits[..., 1]
        self.actions = torch.zeros((self.num_envs, 7), device=self.device)
        self.previous_actions = self.actions.clone()
        self.ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.completion_started = torch.zeros_like(self.ever_lifted)
        self.stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lift_stable_steps = torch.zeros_like(self.stable_steps)
        self.episode_steps = torch.zeros_like(self.stable_steps)
        self.episode_returns = torch.zeros(self.num_envs, device=self.device)
        self.potential = torch.zeros(self.num_envs, device=self.device)
        self.goal = torch.tensor([0.50, 0.32, 0.03], device=self.device).repeat(self.num_envs, 1)
        self.max_episode_steps = 450
        self.curriculum = args.initial_difficulty
        self.goal_curriculum_mode = getattr(args, "goal_curriculum", "none")
        self.goal_curriculum_level = 0.
        self.goal_bins = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.task_stage = "full" if args.task in ("full", "staged_whole") else "lift"
        self.obs_dim = 48
        self.action_dim = 7 if args.control == "joint" else 4
        self.rng = torch.Generator(device=self.device).manual_seed(args.seed)

    def _setup_scene(self):
        self.cloner = GridCloner(spacing=4.0)
        self.cloner.define_base_env("/World/envs")
        paths = self.cloner.generate_paths("/World/envs/env", self.num_envs)
        UsdGeom.Xform.Define(self.stage, paths[0])
        root = UsdGeom.Xform.Define(self.stage, paths[0] + "/Robot").GetPrim()
        root.GetReferences().AddReference(str(self.args.usd.resolve()), ROBOT_SOURCE)
        self.stage.Load()
        # Only the robot subtree is referenced. Source cameras/lights/physics scenes
        # outside it are not copied; the source USD is never edited or saved.
        root_paths = [str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        print("PPO_ROOTS", root_paths, flush=True)
        if len(root_paths) != 1:
            raise RuntimeError(f"Expected one articulation root, found {root_paths}")
        self.articulation_suffix = root_paths[0].removeprefix(paths[0] + "/Robot")
        if self.args.solver_velocity_iterations is not None:
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSolverVelocityIterationCountAttr(self.args.solver_velocity_iterations)
        if self.args.gripper_damping is not None:
            finger_joint = next(p for p in Usd.PrimRange(root) if p.GetName() == "finger_joint")
            UsdPhysics.DriveAPI.Get(finger_joint, "angular").GetDampingAttr().Set(self.args.gripper_damping)
        self.physics_properties = {}
        for prim in Usd.PrimRange(root):
            attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                     if any(word in a.GetName().lower() for word in ("iteration", "selfcollision", "stiffness", "damping", "maxforce"))}
            if attrs:
                self.physics_properties[str(prim.GetPath()).removeprefix(paths[0])] = attrs
        cube = UsdGeom.Cube.Define(self.stage, paths[0] + "/Cube")
        cube.CreateSizeAttr(0.06)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.72, 0, 0.03))
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.05, 0.05)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.08)
        if self.args.solver_velocity_iterations is not None:
            PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateSolverVelocityIterationCountAttr(self.args.solver_velocity_iterations)
        floor = UsdGeom.Cube.Define(self.stage, "/World/Floor")
        floor.CreateSizeAttr(1.0)
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
        floor.AddScaleOp().Set(Gf.Vec3f(1000, 1000, 0.2))
        floor.CreateDisplayColorAttr([Gf.Vec3f(0.3)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        light = UsdLux.DomeLight.Define(self.stage, "/World/Light")
        light.CreateIntensityAttr(1500)
        SimulationManager.setup_simulation(dt=self.dt, device=self.args.device)
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_api = PhysxSchema.PhysxSceneAPI.Apply(prim)
                # Explicit capacity for replicated articulated grippers. Insufficient
                # capacity loses contacts and invalidates any learning result.
                scene_api.CreateGpuFoundLostAggregatePairsCapacityAttr(2**22)
                scene_api.CreateGpuTotalAggregatePairsCapacityAttr(2**22)
                scene_api.CreateGpuFoundLostPairsCapacityAttr(2**21)
                scene_api.CreateGpuMaxRigidContactCountAttr(2**22)
                scene_api.CreateGpuMaxRigidPatchCountAttr(2**18)
                print("PPO_PHYSICS", str(prim.GetPath()), "aggregate_capacity", 2**22, flush=True)
        self.origins = self.cloner.clone(paths[0], paths, replicate_physics=True, enable_env_ids=True)
        import json
        (self.args.run_dir / "physics_properties.json").write_text(json.dumps(self.physics_properties, indent=2), encoding="utf-8")

    def physics_step(self, steps=1):
        if self.args.gravity_compensation:
            # Counter the robot's own gravity through joint torques. Do not disable
            # gravity, move objects, or use cube/goal positions in this servo.
            gravity = self.robot.get_gravity_compensation_forces()
            forces, _ = gravity_force_budget(gravity, self.drive_force_limits)
            if forces.shape != self.initial_q.shape or not torch.isfinite(forces).all():
                raise RuntimeError("Invalid gravity compensation forces")
            self.robot.set_dof_actuation_forces(forces, self.ids)
        SimulationManager.step(steps=steps, update_fabric=self.args.record is not None or self.args.gui)
        if self.physics_guard:
            self.physics_guard.check()
        if self.args.gui and self.args.record is None:
            from isaacsim.core.rendering_manager import RenderingManager
            RenderingManager.render()

    def grip_pose(self):
        pose = self.robot.get_link_transforms()[:, self.grip_link]
        position = pose[:, :3] + quat_apply(pose[:, 3:], self.grip_offset)
        rotation = quat_mul(pose[:, 3:], self.grip_rotation)
        return position, quat_apply(rotation, self.z_axis)

    def prepare_task(self):
        """Find ONE reset pose above the workpiece; never used for policy actions.

        Numerical IK here only initializes the environment. There are no teacher
        labels, demonstration rollouts, or inherited DAgger network weights.
        """
        reset_height = .06 + .10 * self.curriculum if self.task_stage == "lift" else .16
        if self.args.task == "staged_whole":
            reset_height = .085 + .075 * self.curriculum
        target = torch.tensor([0.72, 0., reset_height], device=self.device).repeat(self.num_envs, 1)
        down = -self.z_axis
        q = self.initial_q.clone()
        damping = torch.eye(6, device=self.device).unsqueeze(0) * 0.0064
        for _ in range(160):
            self.robot.set_dof_positions(q, self.ids)
            self.view.update_articulations_kinematic()
            pos, axis = self.grip_pose()
            link = self.robot.get_link_transforms()[:, self.grip_link, :3]
            jac = self.robot.get_jacobians()[:, self.grip_link - 1, :, :6].clone()
            offset = (pos - link).unsqueeze(1).expand(-1, 6, -1)
            jac[:, :3] += torch.cross(jac[:, 3:].transpose(1, 2), offset, dim=-1).transpose(1, 2)
            dp = target - pos
            dp = dp * (0.06 / dp.norm(dim=-1, keepdim=True).clamp_min(0.06))
            error = torch.cat((dp, 0.6 * torch.cross(axis, down, dim=-1)), -1)
            delta = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + damping, error.unsqueeze(-1))).squeeze(-1)
            delta *= (0.08 / delta.abs().amax(-1, keepdim=True).clamp_min(0.08))
            q[:, :6] = (q[:, :6] + 0.5 * delta).clamp(self.lower[:, :6] + .02, self.upper[:, :6] - .02)
        self.robot.set_dof_positions(q, self.ids)
        self.robot.set_dof_velocities(self.zero_vel, self.ids)
        self.robot.set_dof_position_targets(q, self.ids)
        self.physics_step(10)
        pos, axis = self.grip_pose()
        error = (target - pos).norm(dim=-1)
        print("PPO_RESET_POSE", q[0].cpu().tolist(), "position", pos[0].cpu().tolist(), "axis", axis[0].cpu().tolist(), "max_error", error.max().item(), flush=True)
        if error.max() > 0.035 or (axis - down).norm(dim=-1).max() > 0.2:
            raise RuntimeError("Could not initialize a valid downward-facing reset pose")
        self.reset_q = q.clone()
        self.reset_q[:, 6:] = 0
        link_pose = self.robot.get_link_transforms()[:, self.grip_link]
        self.reset_rotation = quat_mul(link_pose[:, 3:], self.grip_rotation).clone()
        return self.reset()

    def reset(self, ids=None):
        ids = self.ids if ids is None else ids.to(dtype=torch.int32)
        if ids.numel() == 0:
            return self.observations()
        # All set_* APIs expect full buffers and a subset of environment indices.
        q = self.robot.get_dof_positions().clone()
        q[ids.long()] = self.reset_q[ids.long()]
        noise = (torch.rand((len(ids), 6), generator=self.rng, device=self.device) - .5) * (.02 + .08 * self.curriculum)
        q[ids.long(), :6] += noise
        self.targets[ids.long()] = q[ids.long()]
        self.robot.set_dof_positions(q, ids)
        self.robot.set_dof_velocities(self.zero_vel, ids)
        self.robot.set_dof_position_targets(self.targets, ids)
        pose = self.cube.get_transforms().clone()
        pose[ids.long()] = torch.tensor([.72, 0., .031, 0., 0., 0., 1.], device=self.device)
        jitter = (torch.rand((len(ids), 2), generator=self.rng, device=self.device) - .5) * (.015 + .10 * self.curriculum)
        pose[ids.long(), :2] += jitter
        self.cube.set_transforms(pose, ids)
        self.cube.set_velocities(self.zero_cube_vel, ids)
        self.goal[ids.long()] = torch.tensor([.60 - .1 * self.curriculum, .18 + .14 * self.curriculum, .03], device=self.device)
        if self.args.task != "full" and self.task_stage == "full":
            self.goal[ids.long()] = torch.tensor([.70 - .20 * self.curriculum, .07 + .25 * self.curriculum, .03], device=self.device)
        self.goal[ids.long(), :2] += (torch.rand((len(ids), 2), generator=self.rng, device=self.device) - .5) * .06 * self.curriculum
        if getattr(self.args, "goal_mode", "curriculum") == "right_10cm":
            # Use the jittered INITIAL cube position, once per reset. The target
            # is not moved by policy actions or increased by the curriculum.
            self.goal[ids.long()] = right_goal(pose[ids.long(), :3])
        elif getattr(self.args, "goal_mode", "curriculum") == "multi_goal":
            if self.args.mode == "train" and self.goal_curriculum_mode != "none":
                destinations, bins = sample_curriculum_goals(pose[ids.long(), :3], self.rng, self.goal_curriculum_level)
                self.goal[ids.long()], self.goal_bins[ids.long()] = destinations, bins
            else:
                self.goal[ids.long()] = sample_goals(pose[ids.long(), :3], ids, self.rng, self.args.goal_request)
        self.actions[ids.long()] = 0
        self.previous_actions[ids.long()] = 0
        self.ever_lifted[ids.long()] = False
        self.completion_started[ids.long()] = False
        self.stable_steps[ids.long()] = 0
        self.lift_stable_steps[ids.long()] = 0
        self.episode_steps[ids.long()] = 0
        self.episode_returns[ids.long()] = 0
        self.view.update_articulations_kinematic()
        pos, _ = self.grip_pose()
        cube = self.cube.get_transforms()[:, :3]
        finger = self.robot.get_dof_positions()[:, self.finger]
        metrics = task_metrics(cube, self.cube.get_velocities(), pos, finger, self.goal, self.ever_lifted, self.stable_steps)
        self.potential[ids.long()] = self.compute_potential(metrics, cube, finger)[ids.long()]
        return self.observations()

    def compute_potential(self, metrics, cube, finger):
        settle = settling_potential(metrics["stable_steps"], self.args.completion_hold_steps) if self.task_stage == "full" else 0.
        if self.args.task in ("staged_precision", "staged_whole"):
            return precision_potential(self.task_stage, metrics, cube, finger, self.cube.get_velocities()) + settle
        if self.args.task != "full":
            velocity = self.cube.get_velocities() if self.args.task == "staged_hold" else None
            return staged_potential(self.task_stage, metrics, cube, finger, velocity) + settle
        return reward_potential(metrics, cube, finger) + settle

    def observations(self):
        q = self.robot.get_dof_positions()
        qd = self.robot.get_dof_velocities()
        pos, axis = self.grip_pose()
        cube = self.cube.get_transforms()
        velocity = self.cube.get_velocities()
        obs = torch.cat((q[:, :6] / 3.14159, qd[:, :6] / 2,
                         q[:, self.finger:self.finger + 1] / .55, qd[:, self.finger:self.finger + 1],
                         pos, axis, cube[:, :3], cube[:, 3:], velocity,
                         cube[:, :3] - pos, self.goal - cube[:, :3], self.actions,
                         self.ever_lifted.float().unsqueeze(-1),
                         (self.lift_stable_steps if self.task_stage == "lift" else self.stable_steps).float().unsqueeze(-1) / 10), -1)
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(f"Observation shape {obs.shape} != {self.obs_dim}")
        if not torch.isfinite(obs).all():
            raise FloatingPointError("Non-finite physics observations")
        return obs.clamp(-10, 10)

    def step(self, raw_actions, reset_done=True):
        self.previous_actions.copy_(self.actions)
        self.actions.zero_()
        self.actions[:, :raw_actions.shape[-1]] = torch.tanh(raw_actions)
        q = self.robot.get_dof_positions()
        if self.args.control == "cartesian":
            delta = self.cartesian_joint_delta(self.actions[:, :3])
            self.targets[:, :6] = (q[:, :6] + delta).clamp(self.lower[:, :6] + .02, self.upper[:, :6] - .02)
            self.targets[:, self.finger] = (.5 * self.actions[:, 3] + .5) * .55
        else:
            self.targets[:, :6] = (q[:, :6] + .04 * self.actions[:, :6]).clamp(self.lower[:, :6] + .02, self.upper[:, :6] - .02)
            self.targets[:, self.finger] = (.5 * self.actions[:, 6] + .5) * .55
        self.robot.set_dof_position_targets(self.targets, self.ids)
        self.physics_step(self.decimation)
        self.episode_steps += 1
        cube = self.cube.get_transforms()[:, :3].clone()
        velocity = self.cube.get_velocities()
        pos, axis = self.grip_pose()
        finger = self.robot.get_dof_positions()[:, self.finger]
        metrics = task_metrics(cube, velocity, pos, finger, self.goal, self.ever_lifted, self.stable_steps)
        self.ever_lifted.copy_(metrics["ever_lifted"])
        self.stable_steps.copy_(metrics["stable_steps"])
        failure = (cube[:, 2] < -.02) | (cube[:, :2].norm(dim=-1) > 1.5) | (pos[:, 2] < -.02)
        objective_success = metrics["success"]
        if self.args.mode == "train" and self.task_stage == "full":
            objective_success, lost_stability, started = completion_transition(
                metrics["stable_steps"], self.completion_started, self.args.completion_hold_steps,
                self.args.completion_retry_mode)
            self.completion_started.copy_(started)
            failure |= lost_stability
        if self.task_stage == "lift":
            objective_success, lift_count = lift_completion(metrics, velocity, self.lift_stable_steps)
            self.lift_stable_steps.copy_(lift_count)
        potential = self.compute_potential(metrics, cube, finger)
        reward = transition_reward(self.potential, potential, objective_success, failure, axis, self.actions,
                                   self.previous_actions, success_bonus=self.args.success_bonus,
                                   reward_scale=self.args.reward_scale)
        self.potential.copy_(potential)
        terminated = objective_success | failure
        episode_limit = 180 if self.task_stage == "lift" else self.max_episode_steps
        truncated = (self.episode_steps >= episode_limit) & ~terminated
        self.episode_returns += reward
        final_obs = self.observations()
        info = {**metrics, "final_observation": final_obs, "episode_return": self.episode_returns.clone(),
                "episode_length": self.episode_steps.clone(), "cube_pos": cube, "failure": failure,
                "objective_success": objective_success, "finger": finger.clone()}
        info["goal_bin"] = self.goal_bins.clone()
        reset_ids = (terminated | truncated).nonzero().flatten()
        # No physics step after reset: other environments must not advance secretly.
        obs = self.reset(reset_ids) if reset_done and len(reset_ids) else final_obs
        return obs, reward, terminated, truncated, info

    def cartesian_joint_delta(self, xyz_action):
        """Low-level IK only: PPO chooses XYZ increments and gripper timing.

        No cube/goal position or task phase is used by this controller. The wrist
        keeps its reset orientation; this is not joint-space end-to-end learning.
        """
        pose = self.robot.get_link_transforms()[:, self.grip_link]
        offset = quat_apply(pose[:, 3:], self.grip_offset)
        rotation = quat_mul(pose[:, 3:], self.grip_rotation)
        conjugate = rotation.clone()
        conjugate[:, :3] *= -1
        error_rotation = quat_mul(self.reset_rotation, conjugate)
        orientation = 2 * error_rotation[:, :3] * torch.where(error_rotation[:, 3:] >= 0, 1., -1.)
        jac = self.robot.get_jacobians()[:, self.grip_link - 1, :, :6].clone()
        jac[:, :3] += torch.cross(jac[:, 3:].transpose(1, 2), offset.unsqueeze(1).expand(-1, 6, -1), dim=-1).transpose(1, 2)
        error = torch.cat((.012 * xyz_action, .3 * orientation), -1)
        damping = torch.eye(6, device=self.device).unsqueeze(0) * .0064
        delta = (jac.transpose(1, 2) @ torch.linalg.solve(jac @ jac.transpose(1, 2) + damping, error.unsqueeze(-1))).squeeze(-1)
        return delta * (.04 / delta.abs().amax(-1, keepdim=True).clamp_min(.04))

    def probe(self):
        start = time.perf_counter()
        q0 = self.robot.get_dof_positions().clone()
        link0 = self.robot.get_link_transforms().clone()
        self.targets.copy_(q0)
        self.targets[:, 0] += 0.1
        self.robot.set_dof_position_targets(self.targets, self.ids)
        for _ in range(60):
            self.physics_step()
        q1 = self.robot.get_dof_positions().clone()
        link1 = self.robot.get_link_transforms().clone()
        result = {
            "device": str(q1.device), "num_envs": self.num_envs,
            "joint_names": self.names, "link_names": self.links,
            "initial_q": q0[0].cpu().tolist(), "final_q": q1[0].cpu().tolist(),
            "joint0_delta": (q1[:, 0] - q0[:, 0]).cpu().tolist(),
            "max_link_displacement": (link1[..., :3] - link0[..., :3]).norm(dim=-1).max().item(),
            "finite": bool(torch.isfinite(q1).all() and torch.isfinite(link1).all()),
            "cube_pose": self.cube.get_transforms()[0].cpu().tolist(),
            "grip_position": self.grip_pose()[0][0].cpu().tolist(),
            "seconds_60_steps": time.perf_counter() - start,
        }
        # Exercise asynchronous subset resets, the gripper and observations too.
        self.prepare_task()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(60):
                raw = torch.zeros((self.num_envs, self.action_dim), device=self.device)
                self.step(raw)
        result["seconds_60_control_steps"] = time.perf_counter() - start
        result["gripper_position_after_close_command"] = self.robot.get_dof_positions()[0, self.finger].item()
        result["control_transitions_per_second"] = 60 * self.num_envs / result["seconds_60_control_steps"]
        self.reset()
        hold_start = self.grip_pose()[0].clone()
        with torch.no_grad():
            raw = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            raw[:, -1] = -6  # Open fingers: isolate free-space arm hold from contacts.
            for _ in range(60):
                self.step(raw)
        hold_end = self.grip_pose()[0].clone()
        result["zero_arm_open_finger_hold_start"] = hold_start.cpu().tolist()
        result["zero_arm_open_finger_hold_end"] = hold_end.cpu().tolist()
        result["zero_arm_open_finger_hold_drift_m"] = (hold_end - hold_start).norm(dim=-1).cpu().tolist()
        result["gravity_compensation"] = self.args.gravity_compensation
        if self.args.gravity_compensation:
            result["gravity_feedforward_arm_torques"] = self.robot.get_gravity_compensation_forces()[0, :6].cpu().tolist()
        if self.num_envs > 1:
            before_q = self.robot.get_dof_positions().clone()
            before_cube = self.cube.get_transforms().clone()
            self.reset(self.ids[:1])
            after_q = self.robot.get_dof_positions()
            after_cube = self.cube.get_transforms()
            result["subset_reset_other_joint_delta"] = (after_q[1:] - before_q[1:]).abs().max().item()
            result["subset_reset_other_cube_delta"] = (after_cube[1:] - before_cube[1:]).abs().max().item()
            if result["subset_reset_other_joint_delta"] > 1e-5 or result["subset_reset_other_cube_delta"] > 1e-5:
                raise RuntimeError("Subset reset changed another environment")
        return result
