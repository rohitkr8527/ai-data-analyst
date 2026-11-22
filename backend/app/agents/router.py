"""OpenRouter Jev Decisions API adapter."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any
import httpx
from app.config import settings
from app.exceptions import ProviderAuthError, ProviderError, ProviderRateLimitError
from app.services.telemetry import trace_span

logger = logging.getLogger(__name__)


class RouteType(str, Enum):
    SIMPLE_SQL = "SIMPLE_SQL"
    COMPARATIVE_ANALYSIS = "COMPARATIVE_ANALYSIS"
    DEEP_INVESTIGATION = "DEEP_INVESTIGATION"
    CONTEXTUAL_FOLLOWUP = "CONTEXTUAL_FOLLOWUP"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"


@dataclass
class JevRoutingDecision:
    route: RouteType
    confidence: float
    probabilities: dict[str, float]
    complexity_hint: str = "lookup"
    requires_hypotheses_hint: bool = False
    preferred_output_hint: str = "direct_answer"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    decision_id: str = ""
    duration_ms: int = 0


_ROUTE_INSTRUCTIONS = (
    "Select the analytical execution route for this user request against the ClassicModels retail sales database "
    "and Autonomous Python Data Science Sandbox. Why/decrease/root-cause questions and open-ended data science requests "
    "(segmentation, RFM, K-Means clustering, cohort analysis, seasonality, outlier detection, regression, hypothesis testing) "
    "require DEEP_INVESTIGATION. Comparing periods or entities requires COMPARATIVE_ANALYSIS. Single metrics, top-N, or simple "
    "filters require SIMPLE_SQL. Follow-ups referencing earlier context require CONTEXTUAL_FOLLOWUP. Unclear requests require "
    "CLARIFICATION_REQUIRED. Off-topic or write/DDL requests require UNSUPPORTED_REQUEST."
)

_ROUTE_CRITERIA = {
    RouteType.SIMPLE_SQL.value: "Single metric aggregation, top-N ranking, or simple filter query",
    RouteType.COMPARATIVE_ANALYSIS.value: "Direct comparison between two time periods, product lines, or territories",
    RouteType.DEEP_INVESTIGATION.value: "Multi-factor root cause analysis, statistical modeling, clustering/RFM, or hypothesis investigation",
    RouteType.CONTEXTUAL_FOLLOWUP.value: "Refining, pivoting, or filtering the previous analytical result",
    RouteType.CLARIFICATION_REQUIRED.value: "Ambiguous or underspecified analytical intent",
    RouteType.UNSUPPORTED_REQUEST.value: "Off-topic request unrelated to ClassicModels business data",
}


class OpenRouterJevRouter:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.url = base_url or settings.openrouter_decisions_url
        self.model = model or settings.openrouter_jev_model
        if not self.api_key:
            raise ProviderAuthError("CONFIG_OPENROUTER_MISSING: OPENROUTER_API_KEY is required for Jev routing.")

    def route(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        schema_summary: list[str] | None = None,
    ) -> JevRoutingDecision:
        context = context or {}
        request_body = {
            "model": self.model,
            "state": {
                "question": question,
                "has_prior_context": bool(context.get("last_metric") or context.get("last_filters") or context.get("previous_question")),
                "prior_metric": context.get("last_metric"),
                "prior_period": context.get("last_period"),
                "available_tables": schema_summary or ["customers", "orders", "orderdetails", "products", "productlines", "payments", "employees", "offices"],
            },
            "questions": {"route": {"type": "choice", "instructions": _ROUTE_INSTRUCTIONS, "criteria": _ROUTE_CRITERIA}},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error = None
        for attempt in range(2):
            start_time = time.monotonic()
            try:
                with trace_span("openrouter.jev.route", attributes={"model": self.model, "attempt": attempt + 1}):
                    with httpx.Client(timeout=20.0) as client:
                        response = client.post(self.url, headers=headers, json=request_body)
                duration_ms = round((time.monotonic() - start_time) * 1000)

                if response.status_code in (401, 403):
                    raise ProviderAuthError("PROVIDER_AUTH_FAILED: Authentication failed for OpenRouter. Verify OPENROUTER_API_KEY.")
                if response.status_code == 429:
                    raise ProviderRateLimitError("PROVIDER_RATE_LIMITED: The AI provider rate limit was reached.")
                if response.status_code >= 400:
                    raise ProviderError(
                        f"OpenRouter Decisions API returned HTTP {response.status_code}: {response.text[:300]}",
                        code="JEV_HTTP_ERROR", status_code=response.status_code,
                    )

                body = response.json()
                route_answer = body.get("answers", {}).get("route", {})
                choice_raw = route_answer.get("choice")
                if not choice_raw:
                    raise ProviderError("OpenRouter Decisions API did not return a valid choice for 'route'.", code="JEV_MALFORMED_RESPONSE")

                try:
                    selected_route = RouteType(choice_raw)
                except ValueError as err:
                    raise ProviderError(f"Unknown route enum value: '{choice_raw}'.", code="JEV_UNKNOWN_ROUTE") from err

                q_lower = question.lower()
                wants_report = any(k in q_lower for k in ("report", "executive summary", "comprehensive", "board", "pdf", "briefing"))
                out_hint = "business_report" if wants_report else "answer_with_visuals"
                if selected_route == RouteType.DEEP_INVESTIGATION:
                    complexity_hint, requires_hyp = "investigation", True
                elif selected_route == RouteType.COMPARATIVE_ANALYSIS:
                    complexity_hint, requires_hyp = "comparison", any(k in q_lower for k in ("why", "driver", "cause", "explain"))
                else:
                    complexity_hint, requires_hyp = "lookup", False

                usage = body.get("usage", {})
                return JevRoutingDecision(
                    route=selected_route,
                    confidence=float(route_answer.get("confidence", 1.0)),
                    probabilities=route_answer.get("probabilities", {}),
                    complexity_hint=complexity_hint,
                    requires_hypotheses_hint=requires_hyp,
                    preferred_output_hint=out_hint,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cost=float(usage.get("cost", 0.0)),
                    decision_id=body.get("id", ""),
                    duration_ms=duration_ms,
                )
            except (ProviderAuthError, ProviderRateLimitError):
                raise
            except httpx.RequestError as exc:
                last_error = exc
                time.sleep(1.0)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"OpenRouter Jev router error ({type(exc).__name__}): {exc}", code="JEV_ROUTING_FAILED") from exc

        raise ProviderError(f"Transient network error connecting to OpenRouter Jev API: {last_error}", code="PROVIDER_NETWORK_ERROR")
