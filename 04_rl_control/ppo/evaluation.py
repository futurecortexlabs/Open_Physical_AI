"""Independent post-success verification without supplying policy actions."""

import torch


class ProgressAudit:
    """Observe milestone evidence only; never chooses actions or task phases."""

    names = ("reached", "grasp_proxy", "lifted", "at_goal_after_lift", "completed",
             "quiet_at_goal", "opened_at_goal", "retreated_at_goal")

    def __init__(self, num_envs, device):
        self.first_steps = torch.zeros((num_envs, len(self.names)), dtype=torch.long, device=device)

    def update(self, metrics, finger, active, step):
        delivered = metrics["ever_lifted"] & metrics["at_goal"]
        opened = delivered & (finger < .12)
        conditions = torch.stack((metrics["grip_distance"] < .05,
                                  (metrics["grip_distance"] < .06) & (finger > .1),
                                  metrics["ever_lifted"],
                                  delivered, metrics["success"], delivered & metrics["quiet"],
                                  opened, opened & (metrics["grip_distance"] > .09)), -1)
        newly = conditions & active.unsqueeze(-1) & (self.first_steps == 0)
        self.first_steps[newly] = step

    def episode(self, index):
        return {name: int(step) or None for name, step in zip(self.names, self.first_steps[index].cpu().tolist())}

    def counts(self):
        return dict(zip(self.names, (self.first_steps > 0).sum(0).cpu().tolist()))


class CompletionAudit:
    """Require continued success after first completion, under the SAME policy.

    A failure during the extra window is final, not retried until a lucky window.
    A time limit before completion stays a failure. A time limit after initial
    success does not skip the requested verification window.
    """

    def __init__(self, num_envs, device, verification_steps=0):
        self.verification_steps = verification_steps
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.initial_success = torch.zeros_like(self.active)
        self.passed = torch.zeros_like(self.active)
        self.remaining = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, success, failure, terminated, truncated):
        pending = self.active & (self.remaining > 0)
        failed_verification = pending & (~success | failure)
        self.remaining[pending & ~failed_verification] -= 1
        verified = pending & ~failed_verification & (self.remaining == 0)
        newly_successful = self.active & ~self.initial_success & success & ~failure
        failed_episode = self.active & ~self.initial_success & ~newly_successful & (terminated | truncated)
        self.initial_success |= newly_successful
        self.remaining[newly_successful] = self.verification_steps
        immediately_verified = newly_successful & (self.verification_steps == 0)
        self.passed |= verified | immediately_verified
        finished = failed_episode | failed_verification | verified | immediately_verified
        self.active &= ~finished
        return finished
