"""Tracing, cancellation registry, and cost tracking."""
from __future__ import annotations
from contextlib import contextmanager
import json
import logging
import threading
import time
from typing import Any, Iterator
from opentelemetry import trace as otel_trace
from app.config import settings

logger = logging.getLogger("app.telemetry")
tracer = otel_trace.get_tracer("multi_agent_data_analyst")


@contextmanager
def trace_span(
    name: str,
    run_id: str = "",
    task_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    attrs = {**(attributes or {}), **({"run_id": run_id} if run_id else {}), **({"task_id": task_id} if task_id else {})}
    start_mono = time.monotonic()
    span_meta: dict[str, Any] = {
        "name": name,
        "run_id": run_id,
        "task_id": task_id,
        "started_at": time.time(),
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
    def __init__(self):
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._reasons: dict[str, str] = {}

    def register(self, run_id: str) -> threading.Event:
        with self._lock:
            return self._events.setdefault(run_id, threading.Event())

    def cancel(self, run_id: str, reason: str = "Cancelled by user request") -> bool:
        with self._lock:
            self._events.setdefault(run_id, threading.Event()).set()
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
    if provider_cost is not None and provider_cost > 0:
        return round(float(provider_cost), 6)
    pricing = settings.pricing_for(model)
    if not pricing:
        return 0.0
    return round((input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000.0, 6)
