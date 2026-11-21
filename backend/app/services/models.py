"""OpenAI model selection, workload routing, and structured Responses API adapter.

Applies configurable models and reasoning effort per task type.
Enforces real API execution with strict error propagation and no mock/offline fallbacks.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from typing import Any, Type, TypeVar
from pydantic import BaseModel
import openai
from openai import OpenAI
from app.config import settings
from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderError
from app.services.telemetry import estimate_cost_usd, trace_span

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class TaskType(str, Enum):
    SQL_GENERATION = "sql_generation"
    SQL_RESEARCH = "sql_research"
    INVESTIGATION_PLANNING = "investigation_planning"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    PYTHON_RESEARCH = "python_research"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_REVIEW = "evidence_review"
    PYTHON_TOOL_SELECTION = "python_tool_selection"
    INSIGHT_GENERATION = "insight_generation"
    REPORT_WRITING = "report_writing"
    SCHEMA_INTERPRETATION = "schema_interpretation"


@dataclass(frozen=True)
class TaskProfile:
    model: str
    reasoning_effort: str  # "low", "medium", "high"


@dataclass
class ModelUsage:
    task_type: str
    model: str
    reasoning_effort: str
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ModelSelector:
    """Manages task-to-model resolution and executes structured OpenAI calls."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ProviderAuthError("CONFIG_OPENAI_MISSING: OPENAI_API_KEY is required. Configure it in your .env file.")
        self.client = OpenAI(api_key=self.api_key, timeout=settings.openai_timeout_seconds, max_retries=0)

    def get_profile(self, task_type: TaskType) -> TaskProfile:
        """Map analytical task type to configured model and reasoning effort."""
        mappings = {
            TaskType.SQL_GENERATION: TaskProfile(
                model=settings.openai_sql_model,
                reasoning_effort="low"
            ),
            TaskType.SQL_RESEARCH: TaskProfile(
                model=settings.openai_sql_research_model,
                reasoning_effort="medium"
            ),
            TaskType.INVESTIGATION_PLANNING: TaskProfile(
                model=settings.openai_planner_model,
                reasoning_effort="medium"
            ),
            TaskType.ROOT_CAUSE_ANALYSIS: TaskProfile(
                model=settings.openai_rca_model,
                reasoning_effort="high"
            ),
            TaskType.PYTHON_RESEARCH: TaskProfile(
                model=settings.openai_python_model,
                reasoning_effort="medium"
            ),
            TaskType.HYPOTHESIS_GENERATION: TaskProfile(
                model=settings.openai_hypothesis_model,
                reasoning_effort="medium"
            ),
            TaskType.EVIDENCE_REVIEW: TaskProfile(
                model=settings.openai_review_model,
                reasoning_effort="medium"
            ),
            TaskType.PYTHON_TOOL_SELECTION: TaskProfile(
                model=settings.openai_tool_selection_model,
                reasoning_effort="low"
            ),
            TaskType.INSIGHT_GENERATION: TaskProfile(
                model=settings.openai_insight_model,
                reasoning_effort="medium"
            ),
            TaskType.REPORT_WRITING: TaskProfile(
                model=settings.openai_report_model,
                reasoning_effort="low"
            ),
            TaskType.SCHEMA_INTERPRETATION: TaskProfile(
                model=settings.openai_schema_model,
                reasoning_effort="low"
            ),
        }
        if task_type not in mappings:
            raise ValueError(f"Unknown task type: {task_type}")
        return mappings[task_type]

    def parse_structured(
        self,
        task_type: TaskType,
        response_model: Type[T],
        system_instructions: str,
        input_payload: Any,
        custom_effort: str | None = None
    ) -> tuple[T, ModelUsage]:
        """Execute structured output generation using OpenAI Responses API.

        Never silently falls back to another model or offline mock.
        Applies a bounded retry for transient network errors using the SAME model.
        """
        profile = self.get_profile(task_type)
        effort = custom_effort or profile.reasoning_effort

        if isinstance(input_payload, (dict, list)):
            input_text = json.dumps(input_payload, default=str)
        else:
            input_text = str(input_payload)

        last_error = None
        max_attempts = 2

        for attempt in range(max_attempts):
            start_time = time.monotonic()
            attempt_effort = "low" if (attempt > 0 and effort in ("medium", "high")) else effort
            try:
                with trace_span(
                    f"openai.{task_type.value}",
                    attributes={"model": profile.model, "effort": attempt_effort, "attempt": attempt + 1}
                ):
                    response = self.client.responses.parse(
                        model=profile.model,
                        instructions=system_instructions,
                        input=input_text,
                        text_format=response_model,
                        reasoning={"effort": attempt_effort}
                    )
                duration_ms = round((time.monotonic() - start_time) * 1000)

                if response.output_parsed is None:
                    raise ProviderError(
                        f"Model {profile.model} did not return structured output for {task_type.value}.",
                        code="MODEL_OUTPUT_INVALID"
                    )

                in_tok = getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") and response.usage else 0
                out_tok = getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") and response.usage else 0
                cost = estimate_cost_usd(profile.model, in_tok, out_tok)

                usage = ModelUsage(
                    task_type=task_type.value,
                    model=profile.model,
                    reasoning_effort=attempt_effort,
                    duration_ms=duration_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost
                )
                logger.info(
                    "OpenAI model %s executed %s in %d ms (tokens=%d)",
                    profile.model, task_type.value, duration_ms, in_tok + out_tok
                )
                return response.output_parsed, usage

            except openai.AuthenticationError as exc:
                raise ProviderAuthError(
                    "PROVIDER_AUTH_FAILED: Authentication failed for OpenAI. Verify OPENAI_API_KEY."
                ) from exc
            except openai.RateLimitError as exc:
                raise ProviderRateLimitError(
                    "PROVIDER_RATE_LIMITED: The AI provider rate limit was reached. Retry later."
                ) from exc
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_error = exc
                logger.warning("Transient network error calling %s (attempt %d/%d): %s", profile.model, attempt + 1, max_attempts, exc)
                time.sleep(1.0)
            except Exception as exc:
                if isinstance(exc, (ProviderAuthError, ProviderRateLimitError, ProviderError)):
                    raise
                logger.error("OpenAI execution failure on %s (%s): %s", profile.model, task_type.value, exc)
                raise ProviderError(
                    f"OpenAI model call failed ({type(exc).__name__}): {exc}",
                    code="PROVIDER_CALL_FAILED"
                ) from exc

        raise ProviderError(
            f"Transient network error connecting to OpenAI with model {profile.model}: {last_error}",
            code="PROVIDER_NETWORK_ERROR"
        )
