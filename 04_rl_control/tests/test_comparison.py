"""Comparison logic needs no Isaac Sim or GPU."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_ppo_evaluations import CONDITIONS, compare_reports


def fixture():
    report = dict.fromkeys(CONDITIONS)
    report.update(task_version="crx_whole_pick_place_v6", control="cartesian",
                  evaluation_task="trained", stage="full", curriculum=0., seed=12,
                  requested_episodes=2, completed_episodes=2, incomplete_episodes=0,
                  verification_steps=60, control_ablation={}, checkpoint_sha256="baseline",
                  initial_conditions={"cube_position": [[.72, 0., .03], [.73, 0., .03]]},
                  successes=1, verified_successes=0, lifted_episodes=1,
                  episodes=[{"environment": 0, "success": False, "verified_success": False,
                             "lifted": False, "milestone_first_steps": {"at_goal_after_lift": None}},
                            {"environment": 1, "success": True, "verified_success": False,
                             "lifted": True, "milestone_first_steps": {"at_goal_after_lift": 120}}])
    return report


class ComparisonTests(unittest.TestCase):
    def test_matched_improvement_and_failure_groups(self):
        baseline, candidate = fixture(), fixture()
        candidate["checkpoint_sha256"] = "candidate"
        candidate["verified_successes"] = 1
        candidate["episodes"][1]["verified_success"] = True
        result = compare_reports(baseline, candidate)
        self.assertEqual(result["counts"]["verified_successes"]["delta"], 1)
        self.assertEqual(result["paired_verification_outcomes"]["newly_verified"], 1)
        self.assertEqual(result["failure_groups"]["baseline"]["lost_post_success_condition"], 1)
        self.assertEqual(result["failure_groups"]["candidate"]["verified"], 1)
        self.assertFalse(result["universal_perfection_proven"])

    def test_reject_changed_seed_or_difficulty(self):
        for key, value in (("seed", 13), ("curriculum", .1), ("physics_hz", 120)):
            candidate = fixture()
            candidate[key] = value
            with self.assertRaisesRegex(ValueError, "conditions differ"):
                compare_reports(fixture(), candidate)

    def test_reject_different_initial_states(self):
        candidate = fixture()
        candidate["initial_conditions"]["cube_position"][0][0] = .71
        with self.assertRaisesRegex(ValueError, "Initial states differ"):
            compare_reports(fixture(), candidate)

    def test_reject_incomplete_or_missing_verification(self):
        for key, value in (("completed_episodes", 1), ("incomplete_episodes", 1), ("verification_steps", 0)):
            candidate = fixture()
            candidate[key] = value
            with self.assertRaises(ValueError):
                compare_reports(fixture(), candidate)

    def test_reject_duplicate_episode_or_incorrect_totals(self):
        candidate = fixture()
        candidate["episodes"][1]["environment"] = 0
        with self.assertRaisesRegex(ValueError, "episode identities"):
            compare_reports(fixture(), candidate)
        candidate = fixture()
        candidate["verified_successes"] = 1
        with self.assertRaisesRegex(ValueError, "totals"):
            compare_reports(fixture(), candidate)

    def test_reject_missing_or_contradictory_outcomes(self):
        for value in (None, 1, True):
            candidate = fixture()
            candidate["episodes"][0]["verified_success"] = value
            with self.assertRaises(ValueError):
                compare_reports(fixture(), candidate)

    def test_comparison_is_read_only_and_order_independent(self):
        baseline, candidate = fixture(), fixture()
        baseline_copy = copy.deepcopy(baseline)
        candidate["episodes"].reverse()
        candidate_copy = copy.deepcopy(candidate)
        result = compare_reports(baseline, candidate)
        self.assertEqual(result["counts"]["verified_successes"]["delta"], 0)
        self.assertEqual(baseline, baseline_copy)
        self.assertEqual(candidate, candidate_copy)


if __name__ == "__main__":
    unittest.main()
