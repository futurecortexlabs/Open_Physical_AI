import copy
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from ppo.multigoal import (CASES, SUITE_VERSION, absolute_request, case_name, contract,
                           parse_instruction, relative_request, sample_goals, validate_geometry, world_offset)
from ppo.goal_curriculum import (BINS, MODE, bounds as curriculum_bounds, progress as curriculum_progress,
                                 sample as sample_curriculum)
from ppo_pick_place import parse_args


class MultiGoalTests(unittest.TestCase):
    def test_destination_curriculum_expands_but_never_changes_final_contract(self):
        cubes = torch.tensor([[.72, 0., .031]]).repeat(4096, 1)
        first, bins = sample_curriculum(cubes, torch.Generator().manual_seed(21), 0.)
        full, full_bins = sample_curriculum(cubes, torch.Generator().manual_seed(22), 1.)
        first_distance = (first[:, :2] - cubes[:, :2]).norm(dim=-1)
        full_distance = (full[:, :2] - cubes[:, :2]).norm(dim=-1)
        self.assertGreaterEqual(first_distance.min(), .065 - 1e-6)
        self.assertLessEqual(first_distance.max(), .08 + 1e-6)
        self.assertGreaterEqual(full_distance.min(), .05 - 1e-6)
        self.assertLessEqual(full_distance.max(), .30 + 1e-6)
        self.assertEqual(sorted(torch.unique(bins).tolist()), list(range(BINS)))
        self.assertEqual(sorted(torch.unique(full_bins).tolist()), list(range(BINS)))
        self.assertEqual(contract("multi_goal")["distance_m"], [.05, .30])
        self.assertEqual(curriculum_bounds(1.)["half_angle"], 3.141592653589793)

    def test_destination_curriculum_requires_every_bin_before_promotion(self):
        successful = [[1., 1., 0., .02, 1., float(identity)] for identity in range(BINS) for _ in range(64)]
        level, difficulty, evidence = curriculum_progress(successful, 450, 0., 0.)
        self.assertEqual((level, difficulty), (.05, 0.))
        self.assertTrue(evidence["ready"])
        weak_bin = copy.deepcopy(successful)
        for record in weak_bin:
            if record[5] == 7:
                record[4] = 0.
        level, difficulty, evidence = curriculum_progress(weak_bin, 450, 0., 0.)
        self.assertEqual((level, difficulty), (0., 0.))
        self.assertFalse(evidence["ready"])
        self.assertEqual(evidence["low_success_bins"], [7])
        self.assertEqual(evidence["under_sampled_bins"], [])
        self.assertTrue(evidence["overall_rate_met"])
        level, difficulty, _ = curriculum_progress(successful, 450, 1., .95)
        self.assertEqual((level, difficulty), (1., 1.))

    def test_curriculum_diagnostics_distinguish_missing_data_from_failure(self):
        level, difficulty, evidence = curriculum_progress([], 128, .15, 0.)
        self.assertEqual((level, difficulty), (.15, 0.))
        self.assertEqual(evidence["under_sampled_bins"], list(range(BINS)))
        self.assertEqual(evidence["low_success_bins"], [])
        self.assertFalse(evidence["full_episode_window_observed"])
        self.assertFalse(evidence["enough_completed_episodes"])
        successful = [[1., 1., 0., .02, 1., float(identity)] for identity in range(BINS) for _ in range(64)]
        original = copy.deepcopy(successful)
        level, difficulty, evidence = curriculum_progress(successful, 449, .15, 0.)
        self.assertEqual((level, difficulty), (.15, 0.))
        self.assertTrue(evidence["overall_rate_met"])
        self.assertEqual(evidence["low_success_bins"], [])
        self.assertFalse(evidence["ready"])
        self.assertEqual(successful, original)

    def test_japanese_instruction_units_directions_and_combinations(self):
        for text in ("右に20センチ", "右２０ｃｍ", "右へ0.2m運んで", "右20センチメートル"):
            self.assertEqual(parse_instruction(text), relative_request(.2, 0))
        self.assertEqual(parse_instruction("左10cm"), relative_request(-.1, 0))
        self.assertEqual(parse_instruction("奥15cm"), relative_request(0, .15))
        self.assertEqual(parse_instruction("手前20cm"), relative_request(0, -.2))
        self.assertEqual(parse_instruction("右15cm、奥10cmに置いて"), relative_request(.15, .1))

    def test_unknown_ambiguous_or_out_of_range_instruction_is_rejected(self):
        for text in ("右1m", "右0cm", "右-10cm", "右nanm", "前20cm", "右20cm左10cm", "右10cm右5cm",
                     "右30cm奥30cm", "右20cm電源を切って", "そこに置いて", "", "上10cm", "右4cm"):
            with self.assertRaises(ValueError, msg=text):
                parse_instruction(text)
        for xy in ((float("nan"), 0), (.2, 0), (1.2, 0), (.5, .5)):
            with self.assertRaises(ValueError):
                absolute_request(*xy)

    def test_random_training_goals_cover_all_directions_and_range(self):
        cubes = torch.tensor([[.72, 0., .031]]).repeat(4096, 1)
        initial = cubes.clone()
        rng = torch.Generator().manual_seed(17)
        goals = sample_goals(cubes, torch.arange(len(cubes)), rng, {"kind": "random"})
        offsets = goals[:, :2] - cubes[:, :2]
        radii = offsets.norm(dim=-1)
        self.assertGreaterEqual(radii.min(), .05 - 1e-6)
        self.assertLessEqual(radii.max(), .30 + 1e-6)
        self.assertGreater(int((radii > .25).sum()), 400)
        for sign_x in (-1, 1):
            for sign_y in (-1, 1):
                self.assertGreater(int(((offsets[:, 0] * sign_x > 0) & (offsets[:, 1] * sign_y > 0)).sum()), 700)
        torch.testing.assert_close(initial, cubes)
        self.assertGreater(len(torch.unique(goals[:, :2], dim=0)), 4000)

    def test_same_seed_preserves_rng_consumption_between_commands(self):
        cubes = torch.tensor([[.72, 0., .031], [.74, -.02, .031]])
        right_rng, left_rng = torch.Generator().manual_seed(8), torch.Generator().manual_seed(8)
        right = sample_goals(cubes, torch.arange(2), right_rng, parse_instruction("右20cm"))
        left = sample_goals(cubes, torch.arange(2), left_rng, parse_instruction("左20cm"))
        torch.testing.assert_close(right_rng.get_state(), left_rng.get_state())
        torch.testing.assert_close(right[:, :2] - cubes[:, :2], -(left[:, :2] - cubes[:, :2]))
        saved = right.clone()
        cubes[:] += 1
        torch.testing.assert_close(saved, right)

    def test_absolute_coordinate_outside_actual_displacement_is_not_clipped(self):
        cubes = torch.tensor([[.72, 0., .031]])
        request = absolute_request(.55, .10)
        goal = sample_goals(cubes, torch.tensor([0]), torch.Generator().manual_seed(2), request)
        torch.testing.assert_close(goal[0], torch.tensor([.55, .10, .03]))
        with self.assertRaises(ValueError):
            sample_goals(cubes, torch.tensor([0]), torch.Generator(), absolute_request(1.08, .35))

    def test_suite_uses_global_environment_identity_on_subset_reset(self):
        ids = torch.tensor([15, 0, 17, 6])
        cubes = torch.tensor([[.72, 0., .031]]).repeat(4, 1)
        request = {"kind": "suite", "version": SUITE_VERSION}
        goals = sample_goals(cubes, ids, torch.Generator().manual_seed(9), request)
        for i, identity in enumerate(ids.tolist()):
            name, r, a = CASES[identity % 16]
            self.assertEqual(case_name(request, identity), name)
            torch.testing.assert_close(goals[i, :2] - cubes[i, :2], torch.tensor(world_offset(r, a)))

    def test_cli_fixed_instructions_are_eval_only_and_suite_is_balanced(self):
        with tempfile.TemporaryDirectory() as directory:
            usd, model = Path(directory) / "asset.usd", Path(directory) / "model.pt"
            usd.write_bytes(b"fixture")
            model.write_bytes(b"fixture")
            base = ["ppo_pick_place.py", "--usd", str(usd), "--checkpoint", str(model),
                    "--task", "staged_whole", "--goal-mode", "multi_goal"]
            with patch.object(sys, "argv", base + ["--mode", "eval", "--instruction", "右20cm"]):
                self.assertEqual(parse_args().goal_request, relative_request(.2, 0))
            with patch.object(sys, "argv", base + ["--mode", "train"]):
                self.assertEqual(parse_args().goal_request, {"kind": "random"})
            for extra in (["--mode", "train", "--instruction", "右20cm"],
                          ["--mode", "eval", "--goal-suite", "--num-envs", "17"],
                          ["--mode", "eval", "--goal-mode", "right_10cm", "--instruction", "右20cm"],
                          ["--mode", "eval", "--goal-curriculum", MODE],
                          ["--mode", "eval", "--instruction", "右20cm", "--goal-xy", ".55", ".1"]):
                with patch.object(sys, "argv", base + extra), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args()

    def test_geometry_audit_checks_contract_instruction_and_labels(self):
        request = {"kind": "suite", "version": SUITE_VERSION}
        cube = torch.tensor([[.72, 0., .031]]).repeat(16, 1)
        goals = sample_goals(cube, torch.arange(16), torch.Generator(), request)
        report = {"goal_contract": contract("multi_goal"), "goal_request": request, "requested_episodes": 16,
                  "initial_conditions": {"cube_position": cube.tolist(), "goal": goals.tolist()},
                  "episodes": [{"environment": i, "goal_case": case_name(request, i)} for i in range(16)]}
        validate_geometry(report)
        for key in ("contract", "position", "label", "unbalanced"):
            bad = copy.deepcopy(report)
            if key == "contract":
                bad["goal_contract"]["version"] = 2
            elif key == "position":
                bad["initial_conditions"]["goal"][0][0] += .01
            elif key == "label":
                bad["episodes"][0]["goal_case"] = "left_05"
            else:
                bad["requested_episodes"] = 15
            with self.assertRaises(ValueError, msg=key):
                validate_geometry(bad)


if __name__ == "__main__":
    unittest.main()
