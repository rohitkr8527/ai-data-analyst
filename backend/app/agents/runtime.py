"""Reusable bounded autonomous agent runtime and execution abstractions.

Implements the Understand -> Plan -> Act -> Observe -> Reflect -> Continue / Finish
loop with structured working memory, resource budgets, trace emission, and cooperative
cancellation — without unbounded prompt concatenation or infinite recursion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Literal

from app.config import settings
from app.schemas import ModelUsageItem
from app.services.telemetry import cancellation_registry, trace_span

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopBudget:
    """Hard resource bounds for an autonomous agent's iterative reasoning loop."""

    max_steps: int = 6
    max_repair_attempts: int = 3
    max_dataset_requests: int = 2
    max_wall_seconds: float = 90.0

    @classmethod
    def from_settings(cls, max_steps: int | None = None) -> "AgentLoopBudget":
        steps = max(1, min(int(max_steps or settings.max_agent_steps), 12))
        return cls(
            max_steps=steps,
            max_repair_attempts=min(3, max(1, steps - 1)),
            max_dataset_requests=2,
            max_wall_seconds=90.0,
        )


@dataclass
class AgentStepRecord:
    """Compact structured record of one iteration in an autonomous agent loop."""

    step_index: int
    phase: Literal["understand", "plan", "act", "observe", "reflect", "finish", "error"]
    action: str
    thought: str
    tool_input_summary: str = ""
    observation_summary: str = ""
    status: Literal["ok", "error", "blocked", "completed"] = "ok"
    duration_ms: int = 0
    artifacts: list[str] = field(default_factory=list)
    code_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "phase": self.phase,
            "action": self.action,
            "thought": self.thought,
            "tool_input_summary": self.tool_input_summary,
            "observation_summary": self.observation_summary,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "artifacts": list(self.artifacts),
            "code_snapshot": self.code_snapshot,
        }


@dataclass
class StepExecutionOutcome:
    """Structured result returned by an agent's tool/action handler to the runtime."""

    action: str
    status: Literal["ok", "error", "blocked", "completed"]
    observation_summary: str
    should_finish: bool = False
    is_repair: bool = False
    is_dataset_request: bool = False
    artifacts: list[str] = field(default_factory=list)
    code_snapshot: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class AutonomousAgentLoop:
    """Coordinates a bounded iterative reasoning loop for research agents."""

    def __init__(
        self,
        agent_name: str,
        run_id: str,
        task_id: str,
        objective: str,
        budget: AgentLoopBudget | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.agent_name = agent_name
        self.run_id = run_id
        self.task_id = task_id
        self.objective = objective
        self.budget = budget or AgentLoopBudget.from_settings()
        self.on_event = on_event
        self.steps: list[AgentStepRecord] = []
        self.usages: list[ModelUsageItem] = []
        self.repair_attempts = 0
        self.dataset_requests = 0
        self.start_time = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def remaining_steps(self) -> int:
        return max(0, self.budget.max_steps - len(self.steps))

    def _emit(self, event_type: str, **payload: Any) -> None:
        if not self.on_event:
            return
        try:
            step_num = payload.get("step", len(self.steps) + 1)
            action_name = payload.get("action")
            if event_type == "agent_step_started":
                msg = f"Step {step_num} ({self.task_id}): reasoning and planning next action..."
            elif event_type == "agent_step_completed":
                msg = f"Step {step_num} ({self.task_id}): completed '{action_name}' in {payload.get('duration_ms', 0)} ms"
            else:
                msg = str(payload.get("message") or f"{self.agent_name} ({self.task_id}): {event_type}")
            raw_status = str(payload.get("status") or "ok")
            ui_status = "error" if raw_status == "error" else "ok"
            self.on_event({
                "stage": f"{self.agent_name}_{self.task_id}",
                "message": msg,
                "status": ui_status,
                "timestamp": time.time(),
                "event": event_type,
                "agent": self.agent_name,
                "run_id": self.run_id,
                "task_id": self.task_id,
                **payload,
            })
        except Exception:
            logger.debug("Agent loop event callback failed", exc_info=True)

    def build_compact_history(self, max_recent: int = 5) -> list[dict[str, Any]]:
        """Return bounded observation history so prompts never grow unboundedly."""
        recent = self.steps[-max_recent:]
        return [
            {
                "step": s.step_index,
                "action": s.action,
                "thought": s.thought[:300],
                "status": s.status,
                "observation": s.observation_summary[:1200],
                "artifacts": s.artifacts,
            }
            for s in recent
        ]

    def run(
        self,
        decide_step: Callable[[int, "AutonomousAgentLoop"], tuple[Any, ModelUsageItem | None]],
        execute_step: Callable[[int, Any, "AutonomousAgentLoop"], StepExecutionOutcome],
    ) -> list[AgentStepRecord]:
        """Execute Understand -> Plan -> Act -> Observe -> Reflect -> Finish within budget."""
        for step_idx in range(1, self.budget.max_steps + 1):
            if cancellation_registry.is_cancelled(self.run_id):
                raise RuntimeError(cancellation_registry.reason(self.run_id))
            if self.elapsed_seconds() > self.budget.max_wall_seconds:
                self.steps.append(
                    AgentStepRecord(
                        step_index=step_idx,
                        phase="finish",
                        action="budget_timeout",
                        thought="Wall-clock budget exceeded; concluding with verified observations so far.",
                        observation_summary=f"Stopped after {self.elapsed_seconds():.1f}s.",
                        status="completed",
                    )
                )
                break

            step_start = time.monotonic()
            with trace_span(
                f"{self.agent_name}_step",
                run_id=self.run_id,
                task_id=self.task_id,
                attributes={"step": step_idx},
            ):
                self._emit("agent_step_started", step=step_idx, remaining=self.remaining_steps())
                decision, usage = decide_step(step_idx, self)
                if usage is not None:
                    self.usages.append(usage)

                outcome = execute_step(step_idx, decision, self)
                if outcome.is_repair:
                    self.repair_attempts += 1
                if outcome.is_dataset_request:
                    self.dataset_requests += 1

                dur_ms = round((time.monotonic() - step_start) * 1000)
                thought_text = str(getattr(decision, "reasoning", None) or getattr(decision, "thought", None) or "")
                input_sum = str(getattr(decision, "action_summary", None) or outcome.action)

                record = AgentStepRecord(
                    step_index=step_idx,
                    phase="finish" if outcome.should_finish else ("error" if outcome.status == "error" else "observe"),
                    action=outcome.action,
                    thought=thought_text[:500],
                    tool_input_summary=input_sum[:300],
                    observation_summary=outcome.observation_summary[:1500],
                    status=outcome.status,
                    duration_ms=dur_ms,
                    artifacts=list(outcome.artifacts),
                    code_snapshot=outcome.code_snapshot,
                )
                self.steps.append(record)
                self._emit(
                    "agent_step_completed",
                    step=step_idx,
                    action=outcome.action,
                    status=outcome.status,
                    duration_ms=dur_ms,
                )

                if outcome.should_finish:
                    break
                if outcome.status == "error" and self.repair_attempts > self.budget.max_repair_attempts:
                    logger.info("%s reached max repair attempts (%d) on task %s", self.agent_name, self.budget.max_repair_attempts, self.task_id)
                    break

        return self.steps
