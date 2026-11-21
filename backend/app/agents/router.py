"""OpenRouter Jev Decisions API adapter.

Dedicated integration for typesafe/jev-1.14 on OpenRouter Alpha Decisions endpoint.
Used once per request at the entry point for bounded routing, complexity, and priority decisions.
No local keyword fallbacks: failures are surfaced cleanly and explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any
import httpx
from app.config import settings
from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderError
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


class OpenRouterJevRouter:
    """Dedicated Jev decision adapter communicating exclusively with OpenRouter."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.url = base_url or settings.openrouter_decisions_url
        self.model = model or settings.openrouter_jev_model
        if not self.api_key:
            raise ProviderAuthError(
                "CONFIG_OPENROUTER_MISSING: OPENROUTER_API_KEY is required for Jev routing."
            )

    def route(
        self,
        question: str,
        context: dict[str, Any] | None = None,
        schema_summary: list[str] | None = None
    ) -> JevRoutingDecision:
        """Query Jev 1.14 via OpenRouter Decisions API to select the analytical route and hints.

        Never silently falls back to keyword matching. Bounded retry on transient network errors.
        """
        context = context or {}
        has_context = bool(context.get("last_metric") or context.get("last_filters") or context.get("previous_question"))

        state_payload = {
            "question": question,
            "has_prior_context": has_context,
            "prior_metric": context.get("last_metric"),
            "prior_period": context.get("last_period"),
            "available_tables": schema_summary or [
                "customers", "orders", "orderdetails", "products",
                "productlines", "payments", "employees", "offices"
            ],
            "analytical_capabilities": [
                "sql_aggregations_and_window_functions",
                "autonomous_python_data_science_sandbox",
                "customer_rfm_segmentation_and_kmeans_clustering",
                "time_series_seasonality_and_outlier_detection",
                "statistical_hypothesis_testing_and_regression",
                "cohort_retention_and_forecasting",
            ],
        }

        request_body = {
            "model": self.model,
            "state": state_payload,
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": (
                        "Select the exact analytical execution route for this user request against "
                        "the ClassicModels retail sales, orders, customers, and payments database AND "
                        "the system's Autonomous Python Data Science Sandbox (scikit-learn, statsmodels, scipy, pandas). "
                        "Why/decrease/root-cause questions AND open-ended data science requests on ClassicModels tables "
                        "(such as customer segmentation, RFM, K-Means clustering, cohort analysis, seasonality, "
                        "outlier detection, regression, or statistical hypothesis testing) are FULLY SUPPORTED and "
                        "require DEEP_INVESTIGATION. "
                        "Comparing periods or entities requires COMPARATIVE_ANALYSIS. "
                        "Single metrics, top-N, or simple filters require SIMPLE_SQL. "
                        "Follow-ups referencing earlier conversation context require CONTEXTUAL_FOLLOWUP. "
                        "Unclear requests require CLARIFICATION_REQUIRED. "
                        "Only genuinely off-topic non-database requests (e.g., weather, general trivia, external stocks, "
                        "or write/DDL SQL requests) require UNSUPPORTED_REQUEST."
                    ),
                    "criteria": {
                        RouteType.SIMPLE_SQL.value: "Single metric aggregation, top-N ranking, or simple filter query",
                        RouteType.COMPARATIVE_ANALYSIS.value: "Direct comparison between two time periods, product lines, or territories",
                        RouteType.DEEP_INVESTIGATION.value: (
                            "Multi-factor root cause analysis, statistical modeling, ML clustering/RFM segmentation, "
                            "seasonality/outlier testing, cohort analysis, or hypothesis investigation on business data"
                        ),
                        RouteType.CONTEXTUAL_FOLLOWUP.value: "Refining, pivoting, or filtering the previous analytical result",
                        RouteType.CLARIFICATION_REQUIRED.value: "Ambiguous or underspecified analytical intent that needs user input",
                        RouteType.UNSUPPORTED_REQUEST.value: "Off-topic request unrelated to ClassicModels business data (not advanced analytics/ML on sales or customers)"
                    }
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            start_time = time.monotonic()
            try:
                with trace_span("openrouter.jev.route", attributes={"model": self.model, "attempt": attempt + 1}):
                    with httpx.Client(timeout=20.0) as client:
                        response = client.post(self.url, headers=headers, json=request_body)

                duration_ms = round((time.monotonic() - start_time) * 1000)

                if response.status_code in (401, 403):
                    raise ProviderAuthError(
                        "PROVIDER_AUTH_FAILED: Authentication failed for OpenRouter. Verify OPENROUTER_API_KEY."
                    )
                if response.status_code == 429:
                    raise ProviderRateLimitError(
                        "PROVIDER_RATE_LIMITED: The AI provider rate limit was reached. Retry later."
                    )
                if response.status_code >= 400:
                    raise ProviderError(
                        f"OpenRouter Decisions API returned HTTP {response.status_code}: {response.text[:300]}",
                        code="JEV_HTTP_ERROR",
                        status_code=response.status_code
                    )

                body = response.json()
                answers = body.get("answers", {})
                route_answer = answers.get("route", {})
                choice_raw = route_answer.get("choice")

                if not choice_raw:
                    raise ProviderError(
                        "OpenRouter Decisions API did not return a valid choice for 'route'.",
                        code="JEV_MALFORMED_RESPONSE"
                    )

                try:
                    selected_route = RouteType(choice_raw)
                except ValueError as err:
                    raise ProviderError(
                        f"OpenRouter Decisions API returned unknown route enum value: '{choice_raw}'.",
                        code="JEV_UNKNOWN_ROUTE"
                    ) from err

                confidence = float(route_answer.get("confidence", 1.0))
                probabilities = route_answer.get("probabilities", {})
                usage = body.get("usage", {})

                # Derive advisory hints for the Planning Agent without constraining dynamic planning
                q_lower = question.lower()
                wants_report = any(k in q_lower for k in ("report", "executive summary", "comprehensive", "board", "pdf", "briefing"))
                if selected_route == RouteType.DEEP_INVESTIGATION:
                    complexity_hint = "investigation"
                    requires_hyp = True
                    out_hint = "business_report" if wants_report else "answer_with_visuals"
                elif selected_route == RouteType.COMPARATIVE_ANALYSIS:
                    complexity_hint = "comparison"
                    requires_hyp = any(k in q_lower for k in ("why", "driver", "cause", "explain"))
                    out_hint = "business_report" if wants_report else "answer_with_visuals"
                else:
                    complexity_hint = "lookup"
                    requires_hyp = False
                    out_hint = "business_report" if wants_report else "answer_with_visuals"

                logger.info(
                    "Jev Decisions routed to %s (confidence: %.2f) in %d ms",
                    selected_route.value, confidence, duration_ms
                )

                return JevRoutingDecision(
                    route=selected_route,
                    confidence=confidence,
                    probabilities=probabilities,
                    complexity_hint=complexity_hint,
                    requires_hypotheses_hint=requires_hyp,
                    preferred_output_hint=out_hint,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cost=float(usage.get("cost", 0.0)),
                    decision_id=body.get("id", ""),
                    duration_ms=duration_ms
                )

            except (ProviderAuthError, ProviderRateLimitError):
                raise
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("Transient error contacting OpenRouter Decisions API (attempt %d/%d): %s", attempt + 1, max_attempts, exc)
                time.sleep(1.0)
            except Exception as exc:
                if isinstance(exc, ProviderError):
                    raise
                logger.error("OpenRouter Jev execution failed: %s", exc)
                raise ProviderError(
                    f"OpenRouter Jev router error ({type(exc).__name__}): {exc}",
                    code="JEV_ROUTING_FAILED"
                ) from exc

        raise ProviderError(
            f"Transient network error connecting to OpenRouter Jev API: {last_error}",
            code="PROVIDER_NETWORK_ERROR"
        )
