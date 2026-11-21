"""Isolated unit tests for AI provider adapters (OpenRouter Jev and OpenAI ModelSelector)."""
import pytest
from unittest.mock import MagicMock, patch
import httpx
import openai
from pydantic import BaseModel

from app.agents.router import OpenRouterJevRouter, RouteType, JevRoutingDecision
from app.services.models import ModelSelector, TaskType, TaskProfile
from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderError


class DummyOutput(BaseModel):
    summary: str
    score: int


def test_jev_router_successful_parse():
    router = OpenRouterJevRouter(api_key="test-key")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "typesafe/jev-1.14",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "DEEP_INVESTIGATION",
                "confidence": 0.95,
                "probabilities": {
                    "DEEP_INVESTIGATION": 0.95,
                    "SIMPLE_SQL": 0.05
                }
            }
        },
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "cost": 0.0001
        },
        "id": "gen-dec-12345"
    }

    with patch("httpx.Client.post", return_value=mock_response):
        decision = router.route("Why did sales decline in November 2004?")
        assert isinstance(decision, JevRoutingDecision)
        assert decision.route == RouteType.DEEP_INVESTIGATION
        assert decision.confidence == 0.95
        assert decision.probabilities["DEEP_INVESTIGATION"] == 0.95
        assert decision.input_tokens == 120


def test_jev_router_auth_failure_raises_typed_error():
    router = OpenRouterJevRouter(api_key="invalid-key")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(ProviderAuthError) as exc_info:
            router.route("Any question")
        assert "PROVIDER_AUTH_FAILED" in str(exc_info.value)


def test_jev_router_rate_limit_raises_typed_error():
    router = OpenRouterJevRouter(api_key="valid-key")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = "Rate limit exceeded"

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(ProviderRateLimitError) as exc_info:
            router.route("Any question")
        assert "PROVIDER_RATE_LIMITED" in str(exc_info.value)


def test_jev_router_malformed_route_raises_typed_error():
    router = OpenRouterJevRouter(api_key="valid-key")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "answers": {
            "route": {
                "type": "choice",
                "choice": "NON_EXISTENT_ROUTE"
            }
        }
    }

    with patch("httpx.Client.post", return_value=mock_response):
        with pytest.raises(ProviderError) as exc_info:
            router.route("Any question")
        assert "JEV_UNKNOWN_ROUTE" in exc_info.value.code


def test_model_selector_task_profiles():
    selector = ModelSelector(api_key="dummy-key")

    sql_profile = selector.get_profile(TaskType.SQL_GENERATION)
    assert sql_profile.model == "gpt-6-luna"
    assert sql_profile.reasoning_effort == "low"

    rca_profile = selector.get_profile(TaskType.ROOT_CAUSE_ANALYSIS)
    assert rca_profile.model == "gpt-6.1-sol"
    assert rca_profile.reasoning_effort == "high"

    insights_profile = selector.get_profile(TaskType.INSIGHT_GENERATION)
    assert insights_profile.model == "gpt-6.1-terra"
    assert insights_profile.reasoning_effort == "medium"


def test_model_selector_openai_auth_error_handling():
    selector = ModelSelector(api_key="invalid-key")
    with patch.object(
        selector.client.responses,
        "parse",
        side_effect=openai.AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401),
            body={}
        )
    ):
        with pytest.raises(ProviderAuthError) as exc_info:
            selector.parse_structured(
                task_type=TaskType.SQL_GENERATION,
                response_model=DummyOutput,
                system_instructions="sys",
                input_payload="inp"
            )
        assert "PROVIDER_AUTH_FAILED" in str(exc_info.value)


def test_model_selector_openai_rate_limit_error_handling():
    selector = ModelSelector(api_key="valid-key")
    with patch.object(
        selector.client.responses,
        "parse",
        side_effect=openai.RateLimitError(
            message="Rate limit reached",
            response=MagicMock(status_code=429),
            body={}
        )
    ):
        with pytest.raises(ProviderRateLimitError) as exc_info:
            selector.parse_structured(
                task_type=TaskType.SQL_GENERATION,
                response_model=DummyOutput,
                system_instructions="sys",
                input_payload="inp"
            )
        assert "PROVIDER_RATE_LIMITED" in str(exc_info.value)
