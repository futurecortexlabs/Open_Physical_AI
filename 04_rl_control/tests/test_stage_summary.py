import copy
import unittest

from test_comparison import fixture
from summarize_ppo_stages import markdown, summarize_stages


class StageSummaryTests(unittest.TestCase):
    def test_all_episode_denominator_and_old_unrecorded_fields(self):
        report = summarize_stages([fixture()])
        rows = {row["key"]: row for row in report["stages"]}
        self.assertEqual(rows["at_goal_after_lift"]["attained"], 1)
        self.assertEqual(rows["at_goal_after_lift"]["rate"], .5)
        self.assertIsNone(rows["opened_at_goal"]["rate"])
        self.assertIn("未計測", markdown(report))
        self.assertIn("seed別の最終成功", markdown(report))
        self.assertFalse(report["target_met_in_this_evaluation"])

    def test_reject_duplicate_seed_different_policy_or_conditions(self):
        for change in ({}, {"seed": 13, "checkpoint_sha256": "other"}, {"seed": 13, "curriculum": .1}):
            candidate = fixture()
            candidate.update(change)
            with self.assertRaises(ValueError):
                summarize_stages([fixture(), candidate])

    def test_requires_each_seed_to_meet_target(self):
        first = fixture()
        first["successes"] = first["verified_successes"] = first["lifted_episodes"] = 2
        for episode in first["episodes"]:
            episode.update(success=True, verified_success=True, lifted=True)
        second = copy.deepcopy(first)
        second["seed"] = 13
        self.assertTrue(summarize_stages([first, second])["target_met_in_this_evaluation"])
        second["verified_successes"] = 1
        second["episodes"][0]["verified_success"] = False
        # Aggregate is 75%, but seed 13 is only 50%.
        result = summarize_stages([first, second], .75)
        self.assertEqual(result["verified_rate"], .75)
        self.assertFalse(result["target_met_in_this_evaluation"])

    def test_reject_partial_measurement_and_invalid_target(self):
        report = fixture()
        report["episodes"][0]["milestone_first_steps"]["opened_at_goal"] = None
        with self.assertRaisesRegex(ValueError, "Partially recorded"):
            summarize_stages([report])
        for target in (0, -1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize_stages([fixture()], target)

    def test_summary_does_not_mutate_input(self):
        report = fixture()
        original = copy.deepcopy(report)
        summarize_stages([report])
        self.assertEqual(report, original)

    def test_reject_invalid_steps_inconsistent_totals_and_outcomes(self):
        for step in (0, -1, True, "120", 1.5):
            report = fixture()
            report["episodes"][1]["milestone_first_steps"]["at_goal_after_lift"] = step
            with self.assertRaisesRegex(ValueError, "Invalid milestone"):
                summarize_stages([report])
        report = fixture()
        report["milestone_counts"] = {"at_goal_after_lift": 2}
        with self.assertRaisesRegex(ValueError, "Milestone total"):
            summarize_stages([report])
        report = fixture()
        report["episodes"][0]["milestone_first_steps"]["completed"] = 5
        with self.assertRaisesRegex(ValueError, "contradicts"):
            summarize_stages([report])


if __name__ == "__main__":
    unittest.main()
