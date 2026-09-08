"""Run without Isaac Sim: python -m unittest discover -s 04_rl_control/tests."""

import sys
import unittest
import tempfile
import contextlib
import io
import json
from types import SimpleNamespace
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo.algorithm import ActorCritic, PPO, PPOConfig, generalized_advantage
from ppo.task import task_metrics, transition_reward, settling_potential, completion_transition
from ppo.training import save_checkpoint, load_checkpoint, initialize_actor, run
from ppo.physics_guard import is_invalid_physics_message
from ppo.curriculum import next_stage, lift_completion, staged_potential, precision_potential
from ppo.control import gravity_force_budget
from export_ppo_report import public_data, export_report
from ppo.evaluation import CompletionAudit, ProgressAudit
from validate_ppo import summarize
from ppo_pick_place import parse_args
from unittest.mock import patch


class PPOTests(unittest.TestCase):
    def test_strict_training_completion_matches_first_success_verification(self):
        started = torch.tensor([False, True, True, False])
        success, failed, started = completion_transition(torch.tensor([10, 0, 70, 0]), started, 70, "fail")
        self.assertEqual(success.tolist(), [False, False, True, False])
        self.assertEqual(failed.tolist(), [False, True, False, False])
        self.assertEqual(started.tolist(), [True, True, True, False])
        _, retry_failure, _ = completion_transition(torch.zeros(4), started, 70, "retry")
        self.assertFalse(retry_failure.any())

    def test_cli_validates_settling_duration_and_reward_parameters(self):
        with tempfile.TemporaryDirectory(prefix="ppo_settling_cli_") as directory:
            asset = Path(directory) / "fixture.usd"
            asset.write_bytes(b"fixture")
            command = ["ppo_pick_place.py", "--mode", "train", "--usd", str(asset)]
            with patch.object(sys, "argv", command + ["--completion-hold-steps", "70"]):
                self.assertEqual(parse_args().completion_hold_steps, 70)
            for flag, value in (("--completion-hold-steps", "9"), ("--completion-hold-steps", "451"),
                                ("--success-bonus", "nan"), ("--reward-scale", "0"), ("--learning-rate", "inf")):
                with patch.object(sys, "argv", command + [flag, value]), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_settling_potential_is_bounded_and_resets_on_instability(self):
        steps = torch.tensor([0, 9, 10, 40, 70, 100])
        values = settling_potential(steps, 70)
        self.assertEqual(values[:2].tolist(), [0., 0.])
        self.assertTrue((values[1:] >= values[:-1]).all())
        self.assertEqual(values[-2:].tolist(), [20., 20.])
        self.assertTrue((settling_potential(steps, 10) == 0).all())

    def test_milestones_keep_first_evidence_and_ignore_finished_episodes(self):
        audit = ProgressAudit(2, "cpu")
        metrics = {"grip_distance": torch.tensor([.02, .03]), "ever_lifted": torch.tensor([True, False]),
                   "at_goal": torch.tensor([True, True]), "success": torch.tensor([False, False]),
                   "quiet": torch.tensor([True, True])}
        audit.update(metrics, torch.tensor([.3, 0.]), torch.tensor([True, True]), 10)
        self.assertEqual(audit.counts(), dict(reached=2, grasp_proxy=1, lifted=1, at_goal_after_lift=1, completed=0,
                                             quiet_at_goal=1, opened_at_goal=0, retreated_at_goal=0))
        metrics["success"][:] = True
        audit.update(metrics, torch.tensor([.3, .3]), torch.tensor([True, False]), 20)
        self.assertEqual(audit.episode(0)["reached"], 10)
        self.assertEqual(audit.episode(0)["completed"], 20)
        self.assertIsNone(audit.episode(1)["grasp_proxy"])
        self.assertIsNone(audit.episode(1)["completed"])

    def test_place_open_and_retreat_are_independent_observation_only_milestones(self):
        audit = ProgressAudit(4, "cpu")
        metrics = {"grip_distance": torch.tensor([.15, .15, .04, .15]),
                   "ever_lifted": torch.tensor([False, True, True, True]),
                   "at_goal": torch.ones(4, dtype=torch.bool),
                   "quiet": torch.tensor([True, True, False, True]),
                   "success": torch.zeros(4, dtype=torch.bool)}
        before = {key: value.clone() for key, value in metrics.items()}
        audit.update(metrics, torch.tensor([0., .3, 0., 0.]), torch.ones(4, dtype=torch.bool), 1)
        self.assertIsNone(audit.episode(0)["quiet_at_goal"])
        self.assertIsNone(audit.episode(0)["opened_at_goal"])
        self.assertEqual(audit.episode(1)["quiet_at_goal"], 1)
        self.assertIsNone(audit.episode(1)["opened_at_goal"])
        self.assertEqual(audit.episode(2)["opened_at_goal"], 1)
        self.assertIsNone(audit.episode(2)["retreated_at_goal"])
        self.assertEqual(audit.episode(3)["retreated_at_goal"], 1)
        for key in before:
            torch.testing.assert_close(metrics[key], before[key])

    def test_cli_records_a_selected_environment_without_changing_batch_size(self):
        with tempfile.TemporaryDirectory(prefix="ppo_cli_test_") as directory:
            root = Path(directory)
            asset = root / "fixture.usd"
            checkpoint = root / "fixture.pt"
            asset.write_bytes(b"fixture")
            checkpoint.write_bytes(b"fixture")
            command = ["ppo_pick_place.py", "--mode", "eval", "--num-envs", "64", "--usd", str(asset),
                       "--checkpoint", str(checkpoint), "--record", str(root / "video.mp4"), "--record-env", "17"]
            with patch.object(sys, "argv", command):
                args = parse_args()
            self.assertEqual((args.num_envs, args.record_env), (64, 17))
            with patch.object(sys, "argv", command[:-1] + ["64"]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args()

    def test_pregrasp_potential_prefers_open_far_and_closed_near(self):
        cube = torch.tensor([[.72, 0., .03]]).repeat(4, 1)
        metrics = {"grip_distance": torch.tensor([.09, .09, .02, .02])}
        finger = torch.tensor([0., .5, 0., .5])
        potential = precision_potential("lift", metrics, cube, finger, torch.zeros(4, 6))
        self.assertGreater(potential[0].item(), potential[1].item())
        self.assertGreater(potential[3].item(), potential[2].item())
        self.assertEqual(next_stage("lift", .95, 1024, .8, elapsed_steps=256, increment=.1), ("lift", 1.))

    def test_suite_rejects_incomplete_or_wrong_model_results(self):
        reports = [{"seed": seed, "evaluation_task": "full", "stage": "full", "curriculum": 1.,
                    "checkpoint_sha256": "model", "requested_episodes": 64, "completed_episodes": 64,
                    "successes": 64, "verification_steps": 60, "verified_successes": 64,
                    "control_ablation": {}} for seed in (901, 902)]
        result = summarize(reports, [901, 902], 64, "model", 60)
        self.assertTrue(result["accepted_in_this_test_scope"])
        self.assertFalse(result["universal_perfection_proven"])
        self.assertFalse(summarize([], [], 64, "model", 60)["accepted_in_this_test_scope"])
        self.assertFalse(summarize(reports[:1], [901, 902], 64, "model", 60)["accepted_in_this_test_scope"])
        self.assertFalse(summarize(reports, [901, 902], 64, "another", 60)["accepted_in_this_test_scope"])
        reports[1]["verified_successes"] = 63
        self.assertFalse(summarize(reports, [901, 902], 64, "model", 60)["accepted_in_this_test_scope"])

    def test_curriculum_waits_for_a_complete_timeout_window(self):
        self.assertEqual(next_stage("lift", .25, 1024, 1., elapsed_steps=128), ("lift", .25))
        self.assertEqual(next_stage("lift", .25, 1024, 1., elapsed_steps=180), ("lift", .5))
        self.assertEqual(next_stage("full", .25, 1024, 1., elapsed_steps=256), ("full", .25))
    def test_verification_continues_policy_and_rejects_any_post_success_failure(self):
        audit = CompletionAudit(4, "cpu", verification_steps=2)
        no_failure = torch.zeros(4, dtype=torch.bool)
        first = torch.tensor([True, True, False, False])
        done = audit.update(first, no_failure, first, torch.tensor([False, False, True, False]))
        self.assertEqual(done.tolist(), [False, False, True, False])
        done = audit.update(torch.tensor([True, False, False, True]), no_failure,
                            torch.tensor([True, False, False, True]), no_failure)
        self.assertEqual(done.tolist(), [False, True, False, False])
        # Timeouts AFTER success do not shortcut verification; failed env 1 cannot retry.
        done = audit.update(torch.ones(4, dtype=torch.bool), no_failure, ~no_failure, ~no_failure)
        self.assertEqual(done.tolist(), [True, False, False, False])
        audit.update(~no_failure, no_failure, ~no_failure, ~no_failure)
        self.assertEqual(audit.initial_success.tolist(), [True, True, False, True])
        self.assertEqual(audit.passed.tolist(), [True, False, False, True])
        self.assertFalse(audit.active.any())

    def test_default_audit_keeps_original_completion_semantics(self):
        audit = CompletionAudit(3, "cpu")
        success = torch.tensor([True, False, False])
        failure = torch.tensor([False, True, False])
        done = audit.update(success, failure, success | failure, torch.tensor([False, False, True]))
        self.assertTrue(done.all())
        self.assertEqual(audit.passed.tolist(), [True, False, False])

    def test_curriculum_saves_completed_level_before_advancing(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="ppo_promotion_test_") as directory:
            root = Path(directory)
            usd = root / "asset.usd"
            usd.write_bytes(b"fixture")
            args = SimpleNamespace(usd=usd, control="cartesian", task="staged_hold", mode="train",
                                   run_dir=root, initialize_from=None, checkpoint=None, iterations=2,
                                   max_seconds=60, gravity_compensation=False, gripper_damping=None,
                                   gripper_max_velocity=None, solver_velocity_iterations=None, physics_hz=60)
            obs = torch.zeros(4, 48)
            done = torch.ones(4, dtype=torch.bool)
            score = torch.ones(4)
            env = SimpleNamespace(args=args, obs_dim=48, action_dim=4, num_envs=4, curriculum=0.,
                                  task_stage="lift", device=torch.device("cpu"),
                                  rng=torch.Generator().manual_seed(2), reset_q=torch.zeros(4, 12),
                                  prepare_task=lambda: obs.clone(), reset=lambda: obs.clone())
            env.step = lambda raw: (obs.clone(), score, done, ~done,
                                    {"final_observation": obs, "success": ~done, "ever_lifted": done,
                                     "episode_return": score, "goal_distance": score,
                                     "objective_success": done})
            with contextlib.redirect_stdout(io.StringIO()):
                run(env, args)
            completed = torch.load(root / "completed_lift_0.00_iteration2.pt", weights_only=True)
            latest = torch.load(root / "latest.pt", weights_only=True)
            self.assertEqual(completed["curriculum"], 0.)
            self.assertEqual(latest["curriculum"], .25)
            change = json.loads((root / "curriculum.jsonl").read_text())
            self.assertEqual(change["completed_checkpoint"], "completed_lift_0.00_iteration2.pt")

    def test_public_report_removes_absolute_paths(self):
        data = {"checkpoint": r"C:\Users\person\runs\latest.pt", "asset": "/home/person/asset.usd",
                "nested": [{"checkpoint": "runs/example/latest.pt"}], "score": 0}
        public = public_data(data)
        self.assertEqual(public["checkpoint"], "latest.pt")
        self.assertEqual(public["asset"], "asset.usd")
        self.assertEqual(public["nested"], data["nested"])
        self.assertEqual(public["score"], 0)
        with tempfile.TemporaryDirectory(prefix="ppo_report_test_") as directory:
            output = Path(directory) / "report.json"
            export_report([], [], output)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                export_report([], [], output)
            self.assertEqual(output.read_bytes(), before)

    def test_control_ablation_is_explicit_and_never_optimizer_resume(self):
        torch.set_num_threads(2)
        model = ActorCritic(48, 4)
        cfg = PPOConfig()
        learner = PPO(model, cfg)
        with tempfile.TemporaryDirectory(prefix="ppo_ablation_test_") as directory:
            root = Path(directory)
            usd = root / "test_asset.usd"
            usd.write_bytes(b"fixture")
            env = SimpleNamespace(obs_dim=48, action_dim=4, curriculum=0., task_stage="lift",
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2),
                                  reset_q=torch.zeros(1, 12),
                                  args=SimpleNamespace(usd=usd, control="cartesian", task="staged_hold"))
            path = root / "checkpoint.pt"
            save_checkpoint(path, model, learner, env, 2, cfg)
            for setting, value in (("gravity_compensation", True), ("gripper_damping", .1),
                                   ("gripper_max_velocity", .5), ("solver_velocity_iterations", 4), ("physics_hz", 120),
                                   ("success_bonus", 300.), ("reward_scale", .01), ("completion_hold_steps", 70),
                                   ("completion_retry_mode", "fail")):
                setattr(env.args, setting, value)
                env.args.eval_control_ablation = False
                with self.assertRaisesRegex(ValueError, "control settings"):
                    load_checkpoint(path, model, learner, env, False)
                env.args.eval_control_ablation = True
                self.assertEqual(load_checkpoint(path, model, learner, env, False), 2)
                self.assertEqual(env.control_ablation[setting]["evaluation"], value)
                with self.assertRaisesRegex(ValueError, "control settings"):
                    load_checkpoint(path, model, learner, env, True)

    def test_gravity_feedforward_and_drive_stay_within_force_limits(self):
        limits = torch.tensor([[10., 10., 10., 5., 5., 5., 2., 0.]])
        gravity = torch.tensor([[30., -30., 2., -2., 0., 3., 10., 10.]])
        feedforward, drive = gravity_force_budget(gravity, limits)
        torch.testing.assert_close(feedforward[:, 6:], torch.zeros(1, 2))
        self.assertTrue((drive >= 0).all())
        self.assertTrue((feedforward.abs() + drive <= limits).all())

    def test_hold_potential_values_slow_motion_above_fast_motion(self):
        metrics = {"grip_distance": torch.tensor([.02, .02]),
                   "goal_distance": torch.tensor([.2, .2]), "ever_lifted": torch.tensor([True, True])}
        cube = torch.tensor([[.72, 0., .13]]).repeat(2, 1)
        velocity = torch.tensor([[0., 0., 0., 0., 0., 0.], [0., 0., .5, 0., 0., 0.]])
        potential = staged_potential("lift", metrics, cube, torch.tensor([.3, .3]), velocity)
        self.assertGreater(potential[0].item(), potential[1].item())

    def test_actor_transfer_keeps_fresh_critic_and_task_settings(self):
        torch.set_num_threads(2)
        original = ActorCritic(48, 4)
        target = ActorCritic(48, 4)
        critic_before = target.critic[0].weight.detach().clone()
        cfg = PPOConfig()
        with tempfile.TemporaryDirectory(prefix="ppo_transfer_test_") as directory:
            root = Path(directory)
            usd = root / "test_asset.usd"
            usd.write_bytes(b"test fixture")
            env = SimpleNamespace(obs_dim=48, action_dim=4, curriculum=0., task_stage="lift",
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2),
                                  reset_q=torch.zeros(1, 12), args=SimpleNamespace(usd=usd, control="cartesian", task="staged"))
            path = root / "checkpoint.pt"
            save_checkpoint(path, original, PPO(original, cfg), env, 10, cfg)
            env.args.task = "staged_hold"
            env.args.gravity_compensation = True
            metadata = initialize_actor(path, target, env)
            self.assertEqual(metadata["kind"], "ppo_actor_only")
            torch.testing.assert_close(target.actor[0].weight, original.actor[0].weight)
            torch.testing.assert_close(target.critic[0].weight, critic_before)
            self.assertEqual(env.args.task, "staged_hold")

    def test_staged_checkpoint_restores_task_and_rejects_v2_mode(self):
        torch.set_num_threads(2)
        model = ActorCritic(48, 4)
        cfg = PPOConfig()
        learner = PPO(model, cfg)
        with tempfile.TemporaryDirectory(prefix="ppo_staged_test_") as directory:
            root = Path(directory)
            usd = root / "test_asset.usd"
            usd.write_bytes(b"test fixture")
            env = SimpleNamespace(obs_dim=48, action_dim=4, curriculum=.5, task_stage="lift",
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2),
                                  reset_q=torch.zeros(1, 12), reset_rotation=torch.tensor([[0., 0., 0., 1.]]),
                                  args=SimpleNamespace(usd=usd, control="cartesian", task="staged"))
            path = root / "checkpoint.pt"
            save_checkpoint(path, model, learner, env, 10, cfg)
            env.task_stage, env.curriculum = "full", 0.
            self.assertEqual(load_checkpoint(path, model, learner, env, True), 10)
            self.assertEqual((env.task_stage, env.curriculum), ("lift", .5))
            env.args.task = "full"
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                load_checkpoint(path, model, learner, env, False)

    def test_curriculum_requires_completed_successful_episodes(self):
        self.assertEqual(next_stage("lift", 0., 511, 1.), ("lift", 0.))
        self.assertEqual(next_stage("lift", 0., 1024, .74), ("lift", 0.))
        self.assertEqual(next_stage("lift", 0., 512, .75), ("lift", .25))
        self.assertEqual(next_stage("lift", 1., 512, .8), ("full", 0.))
        self.assertEqual(next_stage("full", 1., 512, .9), ("full", 1.))

    def test_lift_objective_rejects_fast_or_brief_airborne_cube(self):
        metrics = {"lifted_now": torch.tensor([True, True, True, False]),
                   "grip_distance": torch.tensor([.02, .02, .02, .02])}
        velocity = torch.zeros(4, 6)
        velocity[1, 2] = .2
        success, count = lift_completion(metrics, velocity, torch.tensor([9, 9, 0, 9]))
        self.assertEqual(success.tolist(), [True, False, False, False])
        self.assertEqual(count.tolist(), [10, 0, 1, 0])

    def test_staged_potential_rewards_progress_not_motionless_holding(self):
        metrics = {"grip_distance": torch.tensor([.02, .02]),
                   "goal_distance": torch.tensor([.2, .2]), "ever_lifted": torch.tensor([False, True])}
        cube = torch.tensor([[.72, 0., .03], [.72, 0., .13]])
        potential = staged_potential("lift", metrics, cube, torch.tensor([.2, .2]))
        self.assertGreater(potential[1].item(), potential[0].item())
        r = transition_reward(potential, potential, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                              torch.tensor([[0., 0., -1.]]).repeat(2, 1), torch.zeros(2, 7), torch.zeros(2, 7))
        self.assertTrue((r < 0).all())

    def test_cartesian_checkpoint_and_control_mismatch(self):
        torch.set_num_threads(2)
        model = ActorCritic(48, 4)
        cfg = PPOConfig()
        learner = PPO(model, cfg)
        with tempfile.TemporaryDirectory(prefix="ppo_cartesian_test_") as directory:
            root = Path(directory)
            usd = root / "test_asset.usd"
            usd.write_bytes(b"test fixture, not a real USD")
            env = SimpleNamespace(obs_dim=48, action_dim=4, curriculum=0.,
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2),
                                  reset_q=torch.zeros(1, 12), args=SimpleNamespace(usd=usd, control="cartesian"))
            checkpoint = root / "checkpoint.pt"
            save_checkpoint(checkpoint, model, learner, env, 42, cfg)
            self.assertEqual(load_checkpoint(checkpoint, model, learner, env, False), 42)
            with torch.no_grad():
                self.assertEqual(model.act(torch.zeros(2, 48))[0].shape, (2, 4))
            env.args.control = "joint"
            with self.assertRaisesRegex(ValueError, "control mode"):
                load_checkpoint(checkpoint, model, learner, env, False)

    def test_ppo_learns_reward_direction_in_a_bandit(self):
        torch.manual_seed(13)
        torch.set_num_threads(2)
        model = ActorCritic(4, 1)
        learner = PPO(model, PPOConfig(epochs=4, minibatch=256))
        observations = torch.ones(512, 4)
        with torch.no_grad():
            initial_error = (model.actor(observations).tanh() - .8).square().mean().item()
        for _ in range(30):
            with torch.no_grad():
                actions, logp, values = model.act(observations)
                reward = -(actions.tanh().squeeze(-1) - .8).square()
            learner.update(observations, actions, logp, reward, reward - values)
        with torch.no_grad():
            final_error = (model.actor(observations).tanh() - .8).square().mean().item()
        self.assertLess(final_error, initial_error * .2)

    def test_contact_overflow_is_fatal_but_missing_camera_warning_is_not(self):
        self.assertTrue(is_invalid_physics_message("omni.physx.plugin", "The application needs to increase capacity; simulation will miss interactions"))
        self.assertFalse(is_invalid_physics_message("omni.physx.tensors.plugin", "Failed to find rigid body at RSD455"))

    def test_checkpoint_roundtrip_and_asset_mismatch(self):
        torch.set_num_threads(2)
        model = ActorCritic(48)
        cfg = PPOConfig()
        learner = PPO(model, cfg)
        with tempfile.TemporaryDirectory(prefix="ppo_test_") as directory:
            root = Path(directory)
            usd = root / "test_asset.usd"
            usd.write_bytes(b"test fixture, not a real USD")
            env = SimpleNamespace(obs_dim=48, action_dim=7, curriculum=.25,
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2),
                                  reset_q=torch.zeros(1, 12), args=SimpleNamespace(usd=usd, control="joint"))
            path = root / "checkpoint.pt"
            original = model.actor[0].weight.detach().clone()
            save_checkpoint(path, model, learner, env, 3, cfg)
            with torch.no_grad():
                model.actor[0].weight.add_(1)
            self.assertEqual(load_checkpoint(path, model, learner, env, True), 3)
            torch.testing.assert_close(model.actor[0].weight, original)
            usd.write_bytes(b"different fixture")
            with self.assertRaisesRegex(ValueError, "USD differs"):
                load_checkpoint(path, model, learner, env, False)

    def test_success_rejects_motion_and_resets_stability_counter(self):
        cube = torch.tensor([[.5, .32, .03]]).repeat(2, 1)
        velocity = torch.tensor([[.1, 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1.]])
        metrics = task_metrics(cube, velocity, cube + torch.tensor([0., 0., .15]), torch.zeros(2), cube,
                               torch.ones(2, dtype=torch.bool), torch.tensor([9, 9]))
        self.assertEqual(metrics["success"].tolist(), [False, False])
        self.assertEqual(metrics["stable_steps"].tolist(), [0, 0])

    def test_holding_still_does_not_farm_positive_shaping_reward(self):
        potential = torch.tensor([50.])
        reward = transition_reward(potential, potential, torch.tensor([False]), torch.tensor([False]),
                                   torch.tensor([[0., 0., -1.]]), torch.zeros(1, 7), torch.zeros(1, 7))
        self.assertLess(reward.item(), 0.)

    def test_terminal_potential_is_zero(self):
        reward = transition_reward(torch.tensor([50.]), torch.tensor([90.]), torch.tensor([True]), torch.tensor([False]),
                                   torch.tensor([[0., 0., -1.]]), torch.zeros(1, 7), torch.zeros(1, 7))
        self.assertAlmostEqual(reward.item(), 49.98, places=4)

    def test_reward_scale_and_completion_bonus_leave_terminal_potential_zero(self):
        reward = transition_reward(torch.tensor([116.5]), torch.tensor([9999.]), torch.tensor([True]),
                                   torch.tensor([False]), torch.tensor([[0., 0., -1.]]),
                                   torch.zeros(1, 4), torch.zeros(1, 4), success_bonus=300., reward_scale=.01)
        self.assertAlmostEqual(reward.item(), 1.8348, places=4)
        args = (torch.tensor([50.]), torch.tensor([60.]), torch.tensor([False]), torch.tensor([False]),
                torch.tensor([[0., 0., -1.]]), torch.ones(1, 4), torch.zeros(1, 4))
        torch.testing.assert_close(transition_reward(*args, reward_scale=.01), transition_reward(*args) * .01)

    def test_success_requires_lift_release_and_stability(self):
        cube = torch.tensor([[.5, .32, .03]]).repeat(4, 1)
        goal = cube.clone()
        velocity = torch.zeros((4, 6))
        grip = cube + torch.tensor([0., 0., .15])
        finger = torch.tensor([0., 0., .4, 0.])
        lifted = torch.tensor([True, False, True, True])
        counter = torch.tensor([9, 9, 9, 0])
        metrics = task_metrics(cube, velocity, grip, finger, goal, lifted, counter)
        self.assertEqual(metrics["success"].tolist(), [True, False, False, False])

    def test_timeout_bootstraps_but_termination_does_not(self):
        rewards = torch.tensor([[1., 1.], [100., 100.]])
        values = torch.zeros_like(rewards)
        next_values = torch.full_like(rewards, 10.)
        terminated = torch.tensor([[True, False], [True, True]])
        truncated = torch.tensor([[False, True], [False, False]])
        advantage, returns = generalized_advantage(rewards, values, next_values, terminated, truncated, .9, .95)
        torch.testing.assert_close(advantage[0], torch.tensor([1., 10.]))
        torch.testing.assert_close(returns, advantage)

    def test_gae_continues_within_episode(self):
        r = torch.ones((2, 1))
        zeros = torch.zeros_like(r)
        term = torch.tensor([[False], [True]])
        a, _ = generalized_advantage(r, zeros, zeros, term, torch.zeros_like(term), .9, 1.)
        torch.testing.assert_close(a[:, 0], torch.tensor([1.9, 1.]))

    def test_seven_actions_and_finite_update(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = ActorCritic(48)
        ppo = PPO(model, PPOConfig(epochs=2, minibatch=32))
        obs = torch.randn(128, 48)
        with torch.no_grad():
            actions, logp, values = model.act(obs)
        self.assertEqual(actions.shape, (128, 7))
        initial = model.actor[0].weight.detach().clone()
        metrics = ppo.update(obs, actions, logp, values + torch.randn(128), torch.randn(128))
        self.assertGreater(metrics["minibatches"], 0)
        self.assertFalse(torch.equal(initial, model.actor[0].weight))
        self.assertTrue(all(torch.isfinite(x).all() for x in model.parameters()))


if __name__ == "__main__":
    unittest.main()
