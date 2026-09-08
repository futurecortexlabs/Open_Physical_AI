import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from test_comparison import fixture
from ppo.multigoal import CASES, SUITE_VERSION, contract, expected_xy, parse_instruction, relative_request
from ppo.algorithm import ActorCritic, PPO, PPOConfig
from ppo.training import save_checkpoint, load_checkpoint
from ppo.goal_curriculum import MODE
from run_ppo_command import command_for
from summarize_ppo_stages import summarize_stages
from train_ppo_until_target import accepted
from validate_ppo import summarize


def suite_fixture():
    report = fixture()
    request = {"kind": "suite", "version": SUITE_VERSION}
    cubes = [[.72, 0., .03] for _ in CASES]
    report.update(goal_mode="multi_goal", goal_contract=contract("multi_goal"), goal_request=request,
                  evaluation_task="full", curriculum=1., requested_episodes=16, completed_episodes=16,
                  successes=16, verified_successes=16, lifted_episodes=16,
                  initial_conditions={"cube_position": cubes,
                                      "goal": [[*expected_xy(cube, request, i), .03] for i, cube in enumerate(cubes)]},
                  episodes=[{"environment": i, "goal_case": name, "success": True,
                             "verified_success": True, "lifted": True,
                             "milestone_first_steps": {"at_goal_after_lift": 120}}
                            for i, (name, _, _) in enumerate(CASES)])
    return report


class MultiGoalValidationTests(unittest.TestCase):
    def test_bad_direction_cannot_hide_in_good_overall_rate(self):
        report = suite_fixture()
        self.assertTrue(summarize_stages([report])["target_met_in_this_evaluation"])
        report["episodes"][0]["verified_success"] = False
        report["verified_successes"] = 15
        stages = summarize_stages([report])
        self.assertGreater(stages["verified_rate"], .9)
        self.assertFalse(stages["target_met_in_this_evaluation"])
        self.assertEqual(stages["per_goal"][0]["rate"], 0.)
        self.assertEqual(stages["per_goal"][1]["stages"][-1]["denominator"], 1)

    def test_requests_cannot_be_silently_mixed(self):
        reports = []
        for seed, right in ((12, .1), (13, -.1)):
            report = suite_fixture()
            request = relative_request(right, 0)
            report.update(seed=seed, goal_request=request)
            report["initial_conditions"]["goal"] = [[*expected_xy(cube, request, i), .03]
                                                    for i, cube in enumerate(report["initial_conditions"]["cube_position"])]
            for episode in report["episodes"]:
                episode["goal_case"] = "relative"
            reports.append(report)
        with self.assertRaisesRegex(ValueError, "goal_request"):
            summarize_stages(reports)

    def test_validation_requires_suite_and_each_position(self):
        report = suite_fixture()
        result = summarize([report], [12], 16, "baseline", 60, .9, "full", 1., "multi_goal")
        self.assertTrue(result["accepted_in_this_test_scope"])
        self.assertTrue(result["position_target_met"])
        report["verified_successes"] -= 1
        report["episodes"][0]["verified_success"] = False
        self.assertFalse(summarize([report], [12], 16, "baseline", 60, .9, "full", 1., "multi_goal")["accepted_in_this_test_scope"])
        bad = suite_fixture()
        bad["episodes"].pop()
        self.assertFalse(summarize([bad], [12], 16, "baseline", 60, .9, "full", 1., "multi_goal")["accepted_in_this_test_scope"])

    def test_supervisor_requires_1024_trials_and_16_case_evidence(self):
        result = dict(accepted_in_this_test_scope=True, evaluation_task="full", curriculum=1., goal_mode="multi_goal",
                      goal_contract=contract("multi_goal"), goal_request={"kind": "suite", "version": SUITE_VERSION},
                      min_success_rate=.9, seeds=[1, 2, 3, 4], checkpoint_sha256="fixed", verification_steps=60,
                      requested_episodes=1024, completed_episodes=1024, errors=[], position_target_met=True,
                      per_goal=[{"case": case[0], "episodes": 64, "rate": 1., "verified_successes": 64} for case in CASES])
        self.assertTrue(accepted(result, .9, [1, 2, 3, 4], "fixed", "multi_goal"))
        for changes in ({"goal_mode": "right_10cm"}, {"completed_episodes": 256}, {"position_target_met": False},
                        {"goal_contract": None}, {"per_goal": result["per_goal"][:-1]}):
            self.assertFalse(accepted(dict(result, **changes), .9, [1, 2, 3, 4], "fixed", "multi_goal"))
        bad = copy.deepcopy(result)
        bad["per_goal"][0]["verified_successes"] = 57
        self.assertFalse(accepted(bad, .9, [1, 2, 3, 4], "fixed", "multi_goal"))

    def test_command_uses_same_model_and_only_changes_target(self):
        data = dict(algorithm="PPO", teacher_samples=0, task_version="crx_whole_pick_place_v6", stage="full",
                    goal_mode="multi_goal", goal_contract=contract("multi_goal"), control="cartesian",
                    gravity_compensation=True, completion_hold_steps=70, completion_retry_mode="fail",
                    gripper_damping=.1, gripper_max_velocity=.5)
        right = command_for(data, Path("same.pt"), Path("same.usd"), Path("run"), relative_request(.2, 0))
        left = command_for(data, Path("same.pt"), Path("same.usd"), Path("run"), relative_request(-.1, 0))
        self.assertEqual(right[right.index("--mode") + 1], "eval")
        self.assertEqual(right[right.index("--checkpoint") + 1], left[left.index("--checkpoint") + 1])
        differences = [a for a, b in zip(right, left) if a != b]
        self.assertEqual(differences, ["右0.2m"])
        for forbidden in ("--retarget", "--initialize-from", "--eval-control-ablation"):
            self.assertNotIn(forbidden, right)
        self.assertEqual(right[right.index("--verify-steps") + 1], "60")
        for changes in ({"teacher_samples": 1}, {"goal_mode": "right_10cm"}, {"goal_contract": None}):
            with self.assertRaises(ValueError):
                command_for(dict(data, **changes), Path("a"), Path("b"), Path("c"), relative_request(.2, 0))
        precise = relative_request(.1234567891234567, 1e-20)
        exact = command_for(data, Path("a"), Path("b"), Path("c"), precise)
        self.assertEqual(parse_instruction(exact[exact.index("--instruction") + 1]), precise)

    def test_same_checkpoint_loads_different_instructions_without_retarget_or_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            usd = Path(directory) / "asset.usd"
            usd.write_bytes(b"fixture")
            args = SimpleNamespace(usd=usd, control="cartesian", task="staged_whole", goal_mode="multi_goal",
                                   goal_request={"kind": "random"})
            env = SimpleNamespace(args=args, obs_dim=48, action_dim=4, curriculum=.3, task_stage="full",
                                  device=torch.device("cpu"), rng=torch.Generator().manual_seed(9), reset_q=torch.zeros(1, 12))
            env.goal_curriculum_mode, env.goal_curriculum_level = MODE, .2
            model = ActorCritic(48, 4)
            cfg = PPOConfig()
            learner = PPO(model, cfg)
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, learner, env, 217, cfg)
            original_bytes = path.read_bytes()
            original_weights = copy.deepcopy(model.state_dict())
            for request in (relative_request(.2, 0), relative_request(-.1, 0)):
                args.goal_request = request
                self.assertEqual(load_checkpoint(path, model, learner, env, False), 217)
                self.assertEqual(env.goal_curriculum_mode, MODE)
                self.assertEqual(env.goal_curriculum_level, .2)
                self.assertEqual(path.read_bytes(), original_bytes)
                for key, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, original_weights[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "curriculum differs"):
                load_checkpoint(path, model, learner, env, True)
            args.goal_curriculum = MODE
            self.assertEqual(load_checkpoint(path, model, learner, env, True), 217)
            invalid = torch.load(path, weights_only=True)
            invalid["goal_contract"]["distance_m"][1] = 1.
            torch.save(invalid, path)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_checkpoint(path, model, learner, env, False)


if __name__ == "__main__":
    unittest.main()
