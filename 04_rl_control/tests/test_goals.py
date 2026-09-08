import contextlib
import copy
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ppo.algorithm import ActorCritic, PPO, PPOConfig
from ppo.goals import CAMERA_EYE, CAMERA_TARGET, right_goal, right_offset_xy
from ppo.training import save_checkpoint, load_checkpoint
from ppo_pick_place import parse_args
from compare_ppo_evaluations import compare_reports, validate_report
from summarize_ppo_stages import summarize_stages
from test_comparison import fixture
from train_ppo_until_target import accepted, training_command
from validate_ppo import summarize


class GoalTests(unittest.TestCase):
    def test_exact_10cm_floor_offset_is_camera_right_not_world_x(self):
        dx, dy = right_offset_xy()
        self.assertAlmostEqual(math.hypot(dx, dy), .1)
        forward = [CAMERA_TARGET[i] - CAMERA_EYE[i] for i in range(3)]
        self.assertAlmostEqual(dx * forward[0] + dy * forward[1], 0.)
        self.assertGreater(dx * forward[1] - dy * forward[0], 0.)
        self.assertLess(dx, 0.)
        self.assertGreater(dy, 0.)

    def test_target_uses_each_initial_position_and_does_not_follow_cube(self):
        cubes = torch.tensor([[.72, 0., .031], [.70, -.04, .031]])
        original = cubes.clone()
        goals = right_goal(cubes)
        torch.testing.assert_close(cubes, original)
        torch.testing.assert_close((goals[:, :2] - cubes[:, :2]).norm(dim=-1), torch.full((2,), .1))
        torch.testing.assert_close(goals[:, 2], torch.full((2,), .03))
        saved_goals = goals.clone()
        cubes[:] += 1
        torch.testing.assert_close(goals, saved_goals)

    def test_cli_requires_explicit_train_checkpoint_for_retarget(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset.usd"
            checkpoint = Path(directory) / "model.pt"
            asset.write_bytes(b"fixture")
            checkpoint.write_bytes(b"fixture")
            base = ["ppo_pick_place.py", "--usd", str(asset), "--task", "staged_whole", "--goal-mode", "right_10cm"]
            with patch.object(sys, "argv", base + ["--mode", "train", "--checkpoint", str(checkpoint), "--retarget"]):
                self.assertTrue(parse_args().retarget)
            for extra in (["--mode", "train", "--retarget"],
                          ["--mode", "eval", "--checkpoint", str(checkpoint), "--retarget"],
                          ["--mode", "train", "--task", "full"]):
                with patch.object(sys, "argv", base + extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_legacy_resume_retarget_and_new_checkpoint_contract(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usd = root / "asset.usd"
            usd.write_bytes(b"fixture")
            args = SimpleNamespace(usd=usd, control="cartesian", task="staged_whole")
            env = SimpleNamespace(args=args, obs_dim=48, action_dim=4, curriculum=.2, task_stage="full",
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(2), reset_q=torch.zeros(1, 12))
            model = ActorCritic(48, 4)
            cfg = PPOConfig()
            learner = PPO(model, cfg)
            old = root / "old.pt"
            save_checkpoint(old, model, learner, env, 120, cfg)
            legacy = torch.load(old, weights_only=True)
            legacy.pop("goal_mode")
            torch.save(legacy, old)
            self.assertEqual(load_checkpoint(old, model, learner, env, True), 120)
            args.goal_mode = "right_10cm"
            with self.assertRaisesRegex(ValueError, "goal differs"):
                load_checkpoint(old, model, learner, env, True)
            args.retarget = True
            with self.assertRaisesRegex(ValueError, "goal differs"):
                load_checkpoint(old, model, learner, env, False)
            load_checkpoint(old, model, learner, env, True)
            self.assertTrue(env.task_transfer["ppo_optimizer_rng_restored"])
            self.assertEqual(env.curriculum, .2)
            new = root / "new.pt"
            save_checkpoint(new, model, learner, env, 121, cfg)
            self.assertEqual(torch.load(new, weights_only=True)["goal_mode"], "right_10cm")
            with self.assertRaisesRegex(ValueError, "different goal mode"):
                load_checkpoint(new, model, learner, env, True)
            args.retarget = False
            self.assertEqual(load_checkpoint(new, model, learner, env, False), 121)
            args.goal_mode = "curriculum"
            with self.assertRaisesRegex(ValueError, "goal differs"):
                load_checkpoint(new, model, learner, env, False)

    def right_report(self):
        report = fixture()
        report["goal_mode"] = "right_10cm"
        dx, dy = right_offset_xy()
        report["initial_conditions"]["goal"] = [[p[0] + dx, p[1] + dy, .03]
                                                 for p in report["initial_conditions"]["cube_position"]]
        return report

    def test_fixed_goal_evidence_and_no_mixing_with_legacy_results(self):
        report = self.right_report()
        validate_report(report)
        self.assertEqual(summarize_stages([report])["conditions"]["goal_mode"], "right_10cm")
        legacy = copy.deepcopy(report)
        legacy.pop("goal_mode")
        with self.assertRaisesRegex(ValueError, "goal_mode"):
            compare_reports(legacy, report)
        legacy["seed"] = 99
        with self.assertRaisesRegex(ValueError, "goal modes"):
            summarize_stages([legacy, report])
        report["initial_conditions"]["goal"][0][0] += .01
        with self.assertRaisesRegex(ValueError, "not 10 cm"):
            validate_report(report)

    def test_validation_and_continuation_keep_the_new_goal(self):
        data = dict(algorithm="PPO", teacher_samples=0, task_version="crx_whole_pick_place_v6",
                    stage="full", completion_hold_steps=70, completion_retry_mode="fail",
                    control="cartesian", ppo_config={"learning_rate": .0003}, goal_mode="right_10cm")
        command = training_command(data, Path("a"), Path("b"), Path("c"), 256, 1800)
        self.assertEqual(command[command.index("--goal-mode") + 1], "right_10cm")
        report = dict(seed=1, evaluation_task="full", stage="full", curriculum=1.,
                      goal_mode="right_10cm", checkpoint_sha256="fixed", requested_episodes=64,
                      completed_episodes=64, successes=64, verified_successes=64, verification_steps=60)
        self.assertFalse(summarize([report], [1], 64, "fixed", 60)["accepted_in_this_test_scope"])
        self.assertTrue(summarize([report], [1], 64, "fixed", 60, goal_mode="right_10cm")["accepted_in_this_test_scope"])
        validation = dict(accepted_in_this_test_scope=True, evaluation_task="full", curriculum=1.,
                          min_success_rate=.9, seeds=[1, 2, 3, 4], checkpoint_sha256="fixed",
                          verification_steps=60, requested_episodes=256, completed_episodes=256,
                          goal_mode="right_10cm", errors=[])
        self.assertFalse(accepted(validation, .9, [1, 2, 3, 4], "fixed"))
        self.assertTrue(accepted(validation, .9, [1, 2, 3, 4], "fixed", "right_10cm"))


if __name__ == "__main__":
    unittest.main()
