"""Stop on contact-buffer overflow instead of training on incomplete physics."""

import json
import re


def is_invalid_physics_message(source, message):
    text = message.lower()
    return "physx" in source.lower() and any(fragment in text for fragment in (
        "needs to increase", "will miss interactions", "buffer overflow", "gpu memory allocation failed",
    ))


class PhysicsLogGuard:
    def __init__(self, run_dir):
        import carb
        self.run_dir = run_dir
        self.fatal = []
        self.messages = {}
        self.service = carb.logging.acquire_logging()
        self.handle = self.service.add_logger(self._on_message)

    def _on_message(self, source, level, filename, line, message):
        if "physx" not in source.lower():
            return
        if any(word in message.lower() for word in ("error", "failed", "did not match", "needs to increase")):
            normalized = re.sub(r"env_\d+", "env_*", message)
            self.messages[normalized] = self.messages.get(normalized, 0) + 1
        if is_invalid_physics_message(source, message):
            self.fatal.append(message)

    def check(self):
        if self.fatal:
            raise RuntimeError("Invalid PhysX simulation; refusing to train: " + self.fatal[0])

    def close(self):
        self.service.remove_logger(self.handle)
        report = {"invalid_physics_detected": bool(self.fatal), "fatal": self.fatal, "messages": self.messages}
        (self.run_dir / "physics_messages.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
