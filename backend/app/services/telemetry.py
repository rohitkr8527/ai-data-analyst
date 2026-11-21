"""OpenTelemetry-compatible tracing, structured logging, cancellation registry, and cost tracking."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import threading
import time
from typing import Any, Iterator
from opentelemetry import trace as otel_trace
from app.config import settings

logger = logging.getLogger("app.telemetry")
tracer = otel_trace.get_tracer("multi_agent_data_analyst")


@dataclass
class SpanRecord:
    span_id: str
    name: str
    run_id: str
    task_id: str | None
    stage: str
    status: str
    message: str
    started_at: float
    duration_ms: int
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_trace_event(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "status": self.status,
            "timestamp": self.started_at,
            "duration_ms": self.duration_ms,
            "task_id": self.task_id,
            "attributes": self.attributes,
        }


@contextmanager
def trace_span(
    name: str,
    run_id: str = "",
    task_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Create an OpenTelemetry span and emit structured trace metadata."""
    attrs = dict(attributes or {})
    if run_id:
        attrs["run_id"] = run_id
    if task_id:
        attrs["task_id"] = task_id

    start_wall = time.time()
    start_mono = time.monotonic()
    span_meta: dict[str, Any] = {
        "name": name,
        "run_id": run_id,
        "task_id": task_id,
        "started_at": start_wall,
        "duration_ms": 0,
        "status": "ok",
        "attributes": attrs,
    }
    with tracer.start_as_current_span(name) as span:
        for k, v in attrs.items():
            if isinstance(v, (str, int, float, bool)):
                span.set_attribute(k, v)
        try:
            yield span_meta
        except Exception as exc:
            span_meta["status"] = "error"
            span_meta["error"] = f"{type(exc).__name__}: {exc}"
            span.record_exception(exc)
            raise
        finally:
            span_meta["duration_ms"] = round((time.monotonic() - start_mono) * 1000)
            if settings.log_format == "json":
                logger.info(json.dumps(span_meta, default=str))


class CancellationRegistry:
    """Thread-safe per-run cancellation registry supporting cooperative task cancellation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._reasons: dict[str, str] = {}

    def register(self, run_id: str) -> threading.Event:
        with self._lock:
            ev = self._events.get(run_id)
            if ev is None:
                ev = threading.Event()
                self._events[run_id] = ev
            return ev

    def cancel(self, run_id: str, reason: str = "Cancelled by user request") -> bool:
        with self._lock:
            ev = self._events.setdefault(run_id, threading.Event())
            ev.set()
            self._reasons[run_id] = reason
            return True

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            ev = self._events.get(run_id)
            return bool(ev and ev.is_set())

    def reason(self, run_id: str) -> str:
        with self._lock:
            return self._reasons.get(run_id, "Cancelled")

    def clear(self, run_id: str) -> None:
        with self._lock:
            self._events.pop(run_id, None)
            self._reasons.pop(run_id, None)


cancellation_registry = CancellationRegistry()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, provider_cost: float | None = None) -> float:
    """Compute run cost using explicit provider cost or configured MODEL_PRICING_USD_PER_1M."""
    if provider_cost is not None and provider_cost > 0:
        return round(float(provider_cost), 6)
    pricing = settings.pricing_for(model)
    if pricing is None:
        return 0.0
    in_price, out_price = pricing
    return round((input_tokens * in_price + output_tokens * out_price) / 1_000_000.0, 6)
