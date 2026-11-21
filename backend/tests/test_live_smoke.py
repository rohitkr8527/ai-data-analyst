"""Live end-to-end smoke tests against real MySQL, OpenRouter Jev, and OpenAI models.

Verifies genuine provider execution without mocks or fallbacks.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.agents.router import OpenRouterJevRouter, RouteType
from app.services.models import ModelSelector, TaskType
from app.schemas import SQLDraft, PlanTask, QuestionIntent
from app.services.db import ReadOnlyGateway


from app.config import settings


@pytest.fixture(scope="module")
def client():
    headers = {"X-API-Token": settings.api_token} if settings.api_token else {}
    with TestClient(app, headers=headers) as test_client:
        yield test_client


def test_live_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert "router" in data["models"]
    assert "sql_analyst" in data["models"]


def test_live_catalog_endpoint(client):
    resp = client.get("/api/v1/catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "gross_revenue" in data["metrics"]
    assert "tables" in data
    assert "orders" in data["tables"]
    assert "orderdetails" in data["tables"]


def test_live_openrouter_jev_routing():
    router = OpenRouterJevRouter()
    decision = router.route("Show the top 10 products by revenue in 2004")
    assert decision.route in (RouteType.SIMPLE_SQL, RouteType.CONTEXTUAL_FOLLOWUP)
    assert decision.confidence > 0.5
    assert len(decision.probabilities) > 0


def test_live_openai_sql_analyst_generation():
    selector = ModelSelector()
    gateway = ReadOnlyGateway()
    schema_context = gateway.catalog.format_llm_schema_context(["orders", "orderdetails", "products"])

    task = PlanTask(
        id="top_products",
        kind="top_products",
        purpose="Find top 5 products by revenue in 2004"
    )
    intent = QuestionIntent(kind="top_products", metric="gross_revenue", period="2004", n=5)

    draft, usage = selector.parse_structured(
        task_type=TaskType.SQL_GENERATION,
        response_model=SQLDraft,
        system_instructions="Generate one safe MySQL SELECT query using supplied tables.",
        input_payload={
            "task": task.model_dump(),
            "intent": intent.model_dump(),
            "schema": schema_context,
            "parameters": ["target_start", "target_end", "limit_n"]
        }
    )

    assert draft.sql.upper().startswith("SELECT")
    assert usage.model == "gpt-6-luna"
    assert usage.duration_ms > 0

    # Execute the generated SQL on MySQL to verify it is valid
    params = {"target_start": "2004-01-01", "target_end": "2005-01-01", "limit_n": 5}
    exec_result = gateway.execute(draft.sql, params)
    assert len(exec_result["rows"]) <= 5


def test_live_full_api_question_flow(client):
    # Ask a real analytical question through the live FastAPI endpoint
    resp = client.post("/api/v1/questions", json={
        "question": "Show the top 5 products by revenue in 2004"
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["session_id"] is not None
    assert data["route"] is not None
    assert data["report"]["status"] in ("complete", "partial")
    assert len(data["report"]["evidence"]) >= 1

    # Verify that models_used contains real models invoked
    models_used = data["report"].get("models_used", [])
    assert len(models_used) >= 1
    model_names = [m["model"] for m in models_used]
    # Check that Jev or OpenAI models are present
    assert any("gpt-6" in m or "jev" in m for m in model_names)

    # Test contextual follow-up
    sid = data["session_id"]
    followup_resp = client.post("/api/v1/questions", json={
        "session_id": sid,
        "question": "Now only show customers from USA"
    })
    assert followup_resp.status_code == 200
    followup_data = followup_resp.json()
    assert followup_data["session_id"] == sid
