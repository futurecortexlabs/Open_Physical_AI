import copy
import unittest

from validate_ppo import summarize


class ValidationThresholdTests(unittest.TestCase):
    def reports(self):
        return [{"seed": seed, "evaluation_task": "full", "stage": "full", "curriculum": 1.,
                 "checkpoint_sha256": "fixed", "requested_episodes": 64, "completed_episodes": 64,
                 "incomplete_episodes": 0, "successes": 60, "verified_successes": 58,
                 "verification_steps": 60, "control_ablation": {}} for seed in (1, 2)]

    def test_90_percent_requires_ceiling_per_seed_and_default_stays_perfect(self):
        reports = self.reports()
        result = summarize(reports, [1, 2], 64, "fixed", 60, .9)
        self.assertTrue(result["accepted_in_this_test_scope"])
        self.assertEqual(result["required_successes_per_seed"], 58)
        self.assertFalse(summarize(reports, [1, 2], 64, "fixed", 60)["accepted_in_this_test_scope"])
        reports[0]["successes"] = reports[0]["verified_successes"] = 64
        reports[1]["verified_successes"] = 57
        self.assertFalse(summarize(reports, [1, 2], 64, "fixed", 60, .9)["accepted_in_this_test_scope"])

    def test_near_distance_requires_explicit_scope(self):
        reports = self.reports()
        for report in reports:
            report.update(evaluation_task="trained", curriculum=0.)
        self.assertFalse(summarize(reports, [1, 2], 64, "fixed", 60, .9)["accepted_in_this_test_scope"])
        self.assertTrue(summarize(reports, [1, 2], 64, "fixed", 60, .9, "trained", 0.)["accepted_in_this_test_scope"])

    def test_reject_invalid_rate_missing_and_impossible_counts(self):
        for rate in (0, -1, 1.1, float("nan"), float("inf")):
            self.assertFalse(summarize(self.reports(), [1, 2], 64, "fixed", 60, rate)["accepted_in_this_test_scope"])
        for changes in ({"verified_successes": None}, {"verified_successes": 65},
                        {"successes": 57}, {"incomplete_episodes": 1}, {"stage": "lift"}):
            reports = self.reports()
            reports[0].update(changes)
            self.assertFalse(summarize(reports, [1, 2], 64, "fixed", 60, .9)["accepted_in_this_test_scope"])

    def test_no_mutation(self):
        reports = self.reports()
        original = copy.deepcopy(reports)
        summarize(reports, [1, 2], 64, "fixed", 60, .9)
        self.assertEqual(reports, original)


if __name__ == "__main__":
    unittest.main()
