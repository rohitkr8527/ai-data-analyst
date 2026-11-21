"""FastAPI application entry point for Multi-Agent AI Data Analyst.

Centralized error handling, startup validation, real-time SSE progress streaming,
cancellation, HTML/PDF executive report exports, and REST API endpoints.
No mock or offline fallbacks: live providers and MySQL only.
"""
from contextlib import asynccontextmanager
import json
import logging
import queue
import threading
from uuid import uuid4
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from app.agents.report_agent import render_report_html, render_report_pdf
from app.config import settings
from app.exceptions import (
    AppError,
    DatabaseConnectionError,
    app_error_handler,
    generic_exception_handler,
)
from app.orchestration.workflow import AnalyticalWorkflow
from app.schemas import ChatResponse, QuestionRequest
from app.services.db import ReadOnlyGateway
from app.services.semantics import public_catalog
from app.services.store import SessionStore
from app.services.telemetry import cancellation_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("app.main")

store: SessionStore | None = None
gateway: ReadOnlyGateway | None = None
workflow: AnalyticalWorkflow | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, gateway, workflow
    logger.info("Initializing Multi-Agent AI Data Analyst...")

    # 1. Validate required environment configuration
    settings.validate()

    # 2. Initialize ReadOnlyGateway and verify MySQL connectivity
    try:
        gateway = ReadOnlyGateway()
        _ = gateway.schema()
        logger.info("Successfully connected to MySQL ClassicModels database.")
    except Exception as exc:
        logger.error("Failed to connect to MySQL database at startup: %s", exc)
        raise DatabaseConnectionError(
            f"DATABASE_CONNECTION_FAILED: Could not connect to the configured MySQL database. {exc}"
        ) from exc

    # 3. Initialize SQLite store for sessions/runs
    store = SessionStore()

    # 4. Initialize Multi-Agent LangGraph workflow
    workflow = AnalyticalWorkflow(gateway=gateway)

    logger.info("Startup complete. System is ready for live multi-agent analytical queries.")
    yield


app = FastAPI(
    title="Multi-Agent AI Data Analyst",
    version="2.0.0",
    description="Dynamic Supervisor-Worker multi-agent data analyst powered by OpenRouter Jev, OpenAI models, and LangGraph.",
    lifespan=lifespan,
)

# Exception Handlers
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# CORS Middleware
origins = [x.strip() for x in settings.cors_origins.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Token"],
)


def require_token(x_api_token: str | None = Header(default=None)):
    token = settings.api_token.strip()
    if token and token not in ("your_secure_api_token_here", "your_api_token_here", "none", "null"):
        if x_api_token != token:
            raise HTTPException(status_code=401, detail="Invalid API token. Pass header X-API-Token.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "database": "connected",
        "models": {
            "router": settings.openrouter_jev_model,
            "sql_analyst": settings.openai_sql_model,
            "planner": settings.openai_planner_model,
            "rca": settings.openai_rca_model,
            "hypothesis": settings.openai_hypothesis_model,
            "evidence_reviewer": settings.openai_review_model,
            "insights": settings.openai_insight_model,
            "report_writer": settings.openai_report_model,
        },
    }


@app.get("/api/v1/catalog", dependencies=[Depends(require_token)])
def catalog():
    if not workflow or not gateway:
        raise DatabaseConnectionError()
    return {
        **public_catalog(),
        "tables": gateway.schema(),
    }


@app.post("/api/v1/sessions", dependencies=[Depends(require_token)])
def new_session():
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    return {"session_id": store.create_session()}


@app.get("/api/v1/sessions", dependencies=[Depends(require_token)])
def list_sessions():
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    return store.list_sessions()


@app.get("/api/v1/sessions/{session_id}", dependencies=[Depends(require_token)])
def get_session(session_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    obj = store.get(session_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Session not found")
    return obj


@app.get("/api/v1/sessions/{session_id}/runs", dependencies=[Depends(require_token)])
def get_session_runs(session_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    if not store.get(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return store.list_runs(session_id)


@app.post("/api/v1/questions", response_model=ChatResponse, dependencies=[Depends(require_token)])
def ask(req: QuestionRequest):
    if not store or not workflow:
        raise HTTPException(status_code=500, detail="System not ready")

    sid = req.session_id or store.create_session()
    session = store.get(sid)
    if not session:
        sid = store.create_session()
        session = store.get(sid) or {"context": {}}

    run_id = req.run_id or str(uuid4())
    answer = workflow.run(
        req.question,
        session.get("context"),
        session_id=sid,
        run_id=run_id,
    )

    response = {
        "run_id": run_id,
        "session_id": sid,
        "route": answer["route"],
        "route_confidence": answer.get("route_confidence", 1.0),
        "route_probabilities": answer.get("route_probabilities", {}),
        "report": answer["report"],
    }

    store.save(sid, req.question, {**response, "intent": answer.get("intent", {})})
    return response


@app.post("/api/v1/questions/stream", dependencies=[Depends(require_token)])
def ask_stream(req: QuestionRequest):
    """Stream real LangGraph node & worker execution progress via Server-Sent Events (SSE)."""
    if not store or not workflow:
        raise HTTPException(status_code=500, detail="System not ready")

    sid = req.session_id or store.create_session()
    session = store.get(sid)
    if not session:
        sid = store.create_session()
        session = store.get(sid) or {"context": {}}

    run_id = req.run_id or str(uuid4())
    event_q: queue.Queue[dict | None] = queue.Queue()

    def on_event(evt: dict):
        event_q.put({"type": "progress", "run_id": run_id, "event": evt})

    def worker_fn():
        try:
            answer = workflow.run(
                req.question,
                session.get("context"),
                session_id=sid,
                run_id=run_id,
                on_event=on_event,
            )
            response = {
                "run_id": run_id,
                "session_id": sid,
                "route": answer["route"],
                "route_confidence": answer.get("route_confidence", 1.0),
                "route_probabilities": answer.get("route_probabilities", {}),
                "report": answer["report"],
            }
            store.save(sid, req.question, {**response, "intent": answer.get("intent", {})})
            event_q.put({"type": "complete", "data": response})
        except AppError as app_err:
            event_q.put({"type": "error", "code": app_err.code, "message": app_err.message})
        except Exception as exc:
            event_q.put({"type": "error", "code": "INTERNAL_SERVER_ERROR", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            event_q.put(None)

    threading.Thread(target=worker_fn, daemon=True).start()

    def sse_generator():
        while True:
            item = event_q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.post("/api/v1/runs/{run_id}/cancel", dependencies=[Depends(require_token)])
def cancel_run(run_id: str):
    cancellation_registry.cancel(run_id, reason="Cancelled via API request")
    return {"run_id": run_id, "status": "cancellation_requested"}


@app.get("/api/v1/runs/{run_id}", dependencies=[Depends(require_token)])
def get_run(run_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    obj = store.run(run_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Run not found")
    return obj


@app.get("/api/v1/runs/{run_id}/report.html", response_class=HTMLResponse, dependencies=[Depends(require_token)])
def export_run_html(run_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    obj = store.run(run_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Run not found")
    html_doc = render_report_html(obj.get("report", {}), run_id=run_id)
    return HTMLResponse(content=html_doc)


@app.get("/api/v1/runs/{run_id}/report.pdf", dependencies=[Depends(require_token)])
def export_run_pdf(run_id: str):
    if not store:
        raise HTTPException(status_code=500, detail="Store uninitialized")
    obj = store.run(run_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Run not found")
    pdf_bytes = render_report_pdf(obj.get("report", {}), run_id=run_id)
    headers = {"Content-Disposition": f'attachment; filename="atlas-report-{run_id[:8]}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
