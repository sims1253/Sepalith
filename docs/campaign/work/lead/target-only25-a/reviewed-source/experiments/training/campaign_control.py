"""Single-owner training telemetry and graceful campaign stop controls.

A callback cannot interrupt a hung kernel or enforce a hard process deadline.
The launch recipe must also supply an independent process timeout. This control
reserves time for a full save, then requests a stop at an optimizer boundary.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time


def utc_deadline(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Campaign deadline requires an explicit timezone")
    return result.astimezone(timezone.utc).timestamp()


def control_callback(*, telemetry_path, identity, deadline, reserve_seconds,
                     stop_steps=(), clock=time.time, monotonic=time.monotonic,
                     resource_probe=None):
    """Return a TrainerCallback. A resumed attempt needs a new telemetry path.

    stop_steps are lead decision boundaries, not a changed optimizer schedule.
    A later, explicitly admitted resume may continue from that same full state.
    The checkpoint evaluator can consult callback.evaluation_allowed() to defer
    generation near the deadline while preserving the available checkpoint.
    """
    from transformers import TrainerCallback

    deadline_seconds = utc_deadline(deadline)
    if not math.isfinite(reserve_seconds) or reserve_seconds <= 0:
        raise ValueError("A finite, positive checkpoint reserve is required")
    stop_steps = frozenset(stop_steps)
    if any(not isinstance(step, int) or step < 1 for step in stop_steps):
        raise ValueError("Decision steps must be positive integers")
    telemetry_path = Path(telemetry_path)

    class CampaignControl(TrainerCallback):
        stop_reason = None

        def __init__(self):
            self.recent_step_seconds = []
            self.step_started = None
            self.initial_step = None

        def emit(self, kind, state, **values):
            record = {"event": kind, "at": datetime.fromtimestamp(clock(), timezone.utc).isoformat(),
                      "step": state.global_step, **values}
            # Reject nonfinite telemetry instead of silently accepting NaN JSON.
            encoded = json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
            with telemetry_path.open("a") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

        def evaluation_allowed(self):
            return self.stop_reason != "deadline" and clock() + reserve_seconds < deadline_seconds

        def on_train_begin(self, args, state, control, **kwargs):
            if clock() + reserve_seconds >= deadline_seconds:
                raise ValueError("No training budget remains before the checkpoint reserve")
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            # Every attempt preserves its own telemetry, including a resumed one.
            with telemetry_path.open("x"):
                pass
            self.initial_step = state.global_step
            self.emit("train_begin", state, identity=identity, deadline=deadline,
                      reserve_seconds=reserve_seconds, decision_steps=sorted(stop_steps))
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_started = monotonic()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            duration = monotonic() - self.step_started
            self.recent_step_seconds.append(duration)
            self.recent_step_seconds = self.recent_step_seconds[-20:]
            # Use the slowest recent step, not an optimistic mean. This remains
            # an estimate; the supervisor handles a hung or unexpectedly long step.
            next_step_reserve = max(self.recent_step_seconds)
            if clock() + reserve_seconds + next_step_reserve >= deadline_seconds:
                self.stop_reason = "deadline"
            elif state.global_step in stop_steps and state.global_step > self.initial_step:
                self.stop_reason = "lead_decision"
            self.emit("optimizer_step", state, seconds=duration,
                      remaining_seconds=deadline_seconds - clock(), stop_reason=self.stop_reason,
                      resources=resource_probe() if resource_probe is not None else {})
            if self.stop_reason:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.emit("trainer_metrics", state, metrics=logs or {})
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.emit("train_end", state, stop_reason=self.stop_reason or "trainer_terminal",
                      identity=identity)
            return control

    return CampaignControl()
