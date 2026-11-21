"""Centralized application exceptions and error codes.

Never leak internal database passwords, stack traces, or API keys in error messages.
"""
from typing import Any
from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base application exception with standardized error code."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", status_code: int = 500, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(AppError):
    def __init__(self, message: str, code: str = "CONFIG_ERROR"):
        super().__init__(message=message, code=code, status_code=500)


class ProviderAuthError(AppError):
    def __init__(self, message: str = "PROVIDER_AUTH_FAILED: Authentication failed for the configured provider. Verify the relevant API key."):
        super().__init__(message=message, code="PROVIDER_AUTH_FAILED", status_code=401)


class ProviderRateLimitError(AppError):
    def __init__(self, message: str = "PROVIDER_RATE_LIMITED: The AI provider rate limit was reached. Retry later."):
        super().__init__(message=message, code="PROVIDER_RATE_LIMITED", status_code=429)


class ProviderError(AppError):
    def __init__(self, message: str, code: str = "PROVIDER_ERROR", status_code: int = 502):
        super().__init__(message=message, code=code, status_code=status_code)


class DatabaseConnectionError(AppError):
    def __init__(self, message: str = "DATABASE_CONNECTION_FAILED: Could not connect to the configured MySQL database. Check host, port, credentials, and server status."):
        super().__init__(message=message, code="DATABASE_CONNECTION_FAILED", status_code=503)


class QuerySecurityError(AppError):
    def __init__(self, message: str):
        super().__init__(message=message, code="QUERY_SECURITY_VIOLATION", status_code=400)


class PlanValidationError(AppError):
    """The Planning Agent could not produce a plan that passes deterministic validation."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message=message, code="PLAN_VALIDATION_FAILED", status_code=422, details=details)


class AnalysisToolError(AppError):
    """An approved deterministic analysis tool rejected its inputs or preconditions."""
    def __init__(self, message: str):
        super().__init__(message=message, code="ANALYSIS_TOOL_ERROR", status_code=422)


class SandboxExecutionError(AppError):
    """Raised when sandboxed Python execution fails unrecoverably."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message=message, code="SANDBOX_EXECUTION_ERROR", status_code=422, details=details)


class SandboxSecurityError(AppError):
    """Raised when sandboxed code violates isolation or security policies."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message=message, code="SANDBOX_SECURITY_VIOLATION", status_code=400, details=details)


class SandboxUnavailableError(AppError):
    """Raised when the configured sandbox backend is unavailable (never falls back to host exec)."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message=message, code="SANDBOX_UNAVAILABLE", status_code=503, details=details)


class RunNotFoundError(AppError):
    def __init__(self, message: str = "Run not found."):
        super().__init__(message=message, code="RUN_NOT_FOUND", status_code=404)


def create_error_response(exc: AppError) -> JSONResponse:
    """Format sanitized, structured JSON error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details
        }
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return create_error_response(exc)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Do not leak stack traces or internal secrets to user
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "code": "INTERNAL_SERVER_ERROR",
            "message": f"An unexpected error occurred during execution: {type(exc).__name__}",
            "details": {}
        }
    )
