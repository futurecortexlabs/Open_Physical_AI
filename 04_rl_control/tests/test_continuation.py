import unittest
from pathlib import Path

from train_ppo_until_target import accepted, fresh_seeds, confirmation_seeds, training_command, validation_command
from ppo.goal_curriculum import MODE, bounds


class ContinuationTests(unittest.TestCase):
    def test_reuses_ppo_and_strict_task_without_teacher_or_actor_reinitialization(self):
        data = dict(algorithm="PPO", teacher_samples=0, task_version="crx_whole_pick_place_v6",
                    stage="full", completion_hold_steps=70, completion_retry_mode="fail",
                    control="cartesian", gravity_compensation=True, ppo_config={"learning_rate": .0003})
        command = training_command(data, Path("source.pt"), Path("asset.usd"), Path("run"), 256, 1800)
        self.assertIn("--checkpoint", command)
        self.assertIn("--gravity-compensation", command)
        self.assertNotIn("--initialize-from", command)
        self.assertNotIn("--initial-difficulty", command)
        for changes in ({"teacher_samples": 1}, {"stage": "lift"}, {"completion_hold_steps": 10},
                        {"completion_retry_mode": "retry"}, {"algorithm": "DAgger"}):
            with self.assertRaises(ValueError):
                training_command(dict(data, **changes), Path("a"), Path("b"), Path("c"), 256, 1800)

    def test_evaluation_seed_sets_do_not_repeat(self):
        sets = [set(fresh_seeds(2001, index)) for index in range(4)]
        self.assertEqual(len(set.union(*sets)), 16)

    def test_confirmation_is_fresh_and_always_full_scope(self):
        selection = set.union(*(set(fresh_seeds(7001, index)) for index in range(12)))
        confirmations = [set(confirmation_seeds(7001, index, 12)) for index in range(12)]
        self.assertEqual(len(set.union(*confirmations)), 48)
        self.assertFalse(selection & set.union(*confirmations))
        command = validation_command(Path("fixed.pt"), Path("asset.usd"), Path("run"), .9, [8001, 8002, 8003, 8004], 256)
        self.assertEqual(command[command.index("--eval-task") + 1], "full")
        self.assertEqual(command[command.index("--num-envs") + 1], "256")
        self.assertEqual(command[command.index("--verify-steps") + 1], "60")
        self.assertNotIn("--goal-curriculum", command)

    def test_training_continuation_preserves_adaptive_goal_curriculum(self):
        data = dict(algorithm="PPO", teacher_samples=0, task_version="crx_whole_pick_place_v6",
                    stage="full", completion_hold_steps=70, completion_retry_mode="fail", control="cartesian",
                    goal_mode="multi_goal", ppo_config={"learning_rate": .0003},
                    goal_curriculum={"mode": MODE, "level": .15, "sampling_region": bounds(.15)})
        command = training_command(data, Path("source.pt"), Path("asset.usd"), Path("run"), 256, 1800)
        self.assertEqual(command[command.index("--goal-curriculum") + 1], MODE)
        self.assertNotIn("--retarget", command)

    def test_near_success_or_partial_or_failed_validation_cannot_stop_training(self):
        result = dict(accepted_in_this_test_scope=True, evaluation_task="full", curriculum=1.,
                      min_success_rate=.9, seeds=[1, 2, 3, 4], checkpoint_sha256="fixed",
                      verification_steps=60, requested_episodes=256, completed_episodes=256, errors=[])
        self.assertTrue(accepted(result, .9, [1, 2, 3, 4], "fixed"))
        for changes in ({"evaluation_task": "trained", "curriculum": 0.}, {"completed_episodes": 255},
                        {"errors": ["failed"]}, {"checkpoint_sha256": "other"},
                        {"verification_steps": 0}, {"accepted_in_this_test_scope": False}):
            self.assertFalse(accepted(dict(result, **changes), .9, [1, 2, 3, 4], "fixed"))


if __name__ == "__main__":
    unittest.main()
