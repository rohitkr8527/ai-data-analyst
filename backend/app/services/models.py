"""OpenAI model selection and structured Responses API adapter."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
import logging
import time
from typing import Any, Type, TypeVar
import openai
from openai import OpenAI
from pydantic import BaseModel
from app.config import settings
from app.exceptions import ProviderAuthError, ProviderError, ProviderRateLimitError
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
    reasoning_effort: str


@dataclass
class ModelUsage:
    task_type: str
    model: str
    reasoning_effort: str
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


_TASK_CONFIG: dict[TaskType, tuple[str, str]] = {
    TaskType.SQL_GENERATION: ("openai_sql_model", "low"),
    TaskType.SQL_RESEARCH: ("openai_sql_research_model", "medium"),
    TaskType.INVESTIGATION_PLANNING: ("openai_planner_model", "medium"),
    TaskType.ROOT_CAUSE_ANALYSIS: ("openai_rca_model", "high"),
    TaskType.PYTHON_RESEARCH: ("openai_python_model", "medium"),
    TaskType.HYPOTHESIS_GENERATION: ("openai_hypothesis_model", "medium"),
    TaskType.EVIDENCE_REVIEW: ("openai_review_model", "medium"),
    TaskType.PYTHON_TOOL_SELECTION: ("openai_tool_selection_model", "low"),
    TaskType.INSIGHT_GENERATION: ("openai_insight_model", "medium"),
    TaskType.REPORT_WRITING: ("openai_report_model", "low"),
    TaskType.SCHEMA_INTERPRETATION: ("openai_schema_model", "low"),
}


class ModelSelector:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ProviderAuthError("CONFIG_OPENAI_MISSING: OPENAI_API_KEY is required.")
        self.client = OpenAI(api_key=self.api_key, timeout=settings.openai_timeout_seconds, max_retries=0)

    def get_profile(self, task_type: TaskType) -> TaskProfile:
        if task_type not in _TASK_CONFIG:
            raise ValueError(f"Unknown task type: {task_type}")
        attr, effort = _TASK_CONFIG[task_type]
        return TaskProfile(model=getattr(settings, attr), reasoning_effort=effort)

    def parse_structured(
        self,
        task_type: TaskType,
        response_model: Type[T],
        system_instructions: str,
        input_payload: Any,
        custom_effort: str | None = None,
    ) -> tuple[T, ModelUsage]:
        profile = self.get_profile(task_type)
        effort = custom_effort or profile.reasoning_effort
        input_text = json.dumps(input_payload, default=str) if isinstance(input_payload, (dict, list)) else str(input_payload)

        last_error = None
        for attempt in range(2):
            start_time = time.monotonic()
            attempt_effort = "low" if (attempt > 0 and effort in ("medium", "high")) else effort
            try:
                with trace_span(f"openai.{task_type.value}", attributes={"model": profile.model, "effort": attempt_effort}):
                    response = self.client.responses.parse(
                        model=profile.model,
                        instructions=system_instructions,
                        input=input_text,
                        text_format=response_model,
                        reasoning={"effort": attempt_effort},
                    )
                duration_ms = round((time.monotonic() - start_time) * 1000)
                if response.output_parsed is None:
                    raise ProviderError(f"Model {profile.model} returned empty output for {task_type.value}.", code="MODEL_OUTPUT_INVALID")

                usage_obj = getattr(response, "usage", None)
                in_tok = getattr(usage_obj, "input_tokens", 0) if usage_obj else 0
                out_tok = getattr(usage_obj, "output_tokens", 0) if usage_obj else 0

                return response.output_parsed, ModelUsage(
                    task_type=task_type.value,
                    model=profile.model,
                    reasoning_effort=attempt_effort,
                    duration_ms=duration_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=estimate_cost_usd(profile.model, in_tok, out_tok),
                )
            except openai.AuthenticationError as exc:
                raise ProviderAuthError("PROVIDER_AUTH_FAILED: Authentication failed for OpenAI. Verify OPENAI_API_KEY.") from exc
            except openai.RateLimitError as exc:
                raise ProviderRateLimitError("PROVIDER_RATE_LIMITED: The AI provider rate limit was reached.") from exc
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                last_error = exc
                time.sleep(1.0)
            except (ProviderAuthError, ProviderRateLimitError, ProviderError):
                raise
            except Exception as exc:
                raise ProviderError(f"OpenAI model call failed ({type(exc).__name__}): {exc}", code="PROVIDER_CALL_FAILED") from exc

        raise ProviderError(f"Transient network error connecting to OpenAI ({profile.model}): {last_error}", code="PROVIDER_NETWORK_ERROR")
