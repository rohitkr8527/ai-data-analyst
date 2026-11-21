"""Runtime configuration loaded from environment.

Mandatory live AI and MySQL settings. Offline and mock modes are not permitted.
"""
from dataclasses import dataclass, field
from pathlib import Path
import os
from app.exceptions import ConfigurationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


def _load_env():
    path = PROJECT_ROOT / '.env'
    if path.exists():
        for raw in path.read_text(encoding='utf-8').splitlines():
            raw = raw.strip()
            if not raw or raw.startswith('#') or '=' not in raw:
                continue
            k, v = raw.split('=', 1)
            clean_v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), clean_v)


_load_env()


def _int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(hi, max(lo, value))


def _float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return min(hi, max(lo, value))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Database
    database_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))
    app_database_path: str = field(default_factory=lambda: os.getenv('APP_DATABASE_PATH', str(BACKEND_ROOT / 'data/app.db')))
    checkpoint_database_path: str = field(default_factory=lambda: os.getenv('CHECKPOINT_DATABASE_PATH', str(BACKEND_ROOT / 'data/checkpoints.db')))
    dataset_store_path: str = field(default_factory=lambda: os.getenv('DATASET_STORE_PATH', str(BACKEND_ROOT / 'data/datasets.db')))

    # AI Providers
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY', ''))
    openai_timeout_seconds: float = field(default_factory=lambda: _float('OPENAI_TIMEOUT_SECONDS', 90.0, 15.0, 300.0))

    # Models by Task (all configurable; verified against the live OpenAI model list)
    openai_sql_model: str = field(default_factory=lambda: os.getenv('OPENAI_SQL_MODEL', 'gpt-6-luna'))
    openai_sql_research_model: str = field(default_factory=lambda: os.getenv('OPENAI_SQL_RESEARCH_MODEL', 'gpt-6.1-sol'))
    openai_planner_model: str = field(default_factory=lambda: os.getenv('OPENAI_PLANNER_MODEL', 'gpt-6.1-sol'))
    openai_rca_model: str = field(default_factory=lambda: os.getenv('OPENAI_RCA_MODEL', 'gpt-6.1-sol'))
    openai_python_model: str = field(default_factory=lambda: os.getenv('OPENAI_PYTHON_MODEL', 'gpt-6.1-sol'))
    openai_hypothesis_model: str = field(default_factory=lambda: os.getenv('OPENAI_HYPOTHESIS_MODEL', 'gpt-6.1-sol'))
    openai_review_model: str = field(default_factory=lambda: os.getenv('OPENAI_REVIEW_MODEL', 'gpt-6.1-sol'))
    openai_report_model: str = field(default_factory=lambda: os.getenv('OPENAI_REPORT_MODEL', 'gpt-6.1-terra'))
    openai_insight_model: str = field(default_factory=lambda: os.getenv('OPENAI_INSIGHT_MODEL', 'gpt-6.1-terra'))
    openai_schema_model: str = field(default_factory=lambda: os.getenv('OPENAI_SCHEMA_MODEL', 'gpt-6-luna'))
    openai_tool_selection_model: str = field(default_factory=lambda: os.getenv('OPENAI_TOOL_SELECTION_MODEL', 'gpt-6-luna'))

    # Sandbox Execution Configuration
    sandbox_provider: str = field(default_factory=lambda: os.getenv('SANDBOX_PROVIDER', 'isolated_process').strip().lower())
    sandbox_timeout_seconds: int = field(default_factory=lambda: _int('SANDBOX_TIMEOUT_SECONDS', 30, 2, 180))
    sandbox_max_memory_mb: int = field(default_factory=lambda: _int('SANDBOX_MAX_MEMORY_MB', 512, 64, 4096))
    sandbox_max_output_bytes: int = field(default_factory=lambda: _int('SANDBOX_MAX_OUTPUT_BYTES', 65536, 4096, 1048576))
    sandbox_allow_external_data_transfer: bool = field(default_factory=lambda: _bool('SANDBOX_ALLOW_EXTERNAL_DATA_TRANSFER', False))
    max_agent_steps: int = field(default_factory=lambda: _int('MAX_AGENT_STEPS', 4, 1, 10))

    # Optional model pricing for cost tracking: "model=input_per_1m/output_per_1m,..."
    # Left empty, OpenAI cost is reported as unknown (never guessed).
    model_pricing: str = field(default_factory=lambda: os.getenv('MODEL_PRICING_USD_PER_1M', ''))

    # OpenRouter Jev
    openrouter_jev_model: str = field(default_factory=lambda: os.getenv('OPENROUTER_JEV_MODEL', 'typesafe/jev-1.14'))
    openrouter_decisions_url: str = field(default_factory=lambda: os.getenv('OPENROUTER_DECISIONS_URL', 'https://openrouter.ai/api/alpha/decisions'))

    # SQL generation strategy:
    #   compiled -> structured specs compile deterministically (0 tokens); Luna drafts only custom queries (default)
    #   llm      -> Luna drafts every query against a compiled reference contract
    sql_generation_mode: str = field(default_factory=lambda: os.getenv('SQL_GENERATION_MODE', 'compiled').strip().lower())

    # Guardrails and Limits
    max_sql_retries: int = field(default_factory=lambda: _int('MAX_SQL_RETRIES', 2, 0, 5))
    max_investigation_queries: int = field(default_factory=lambda: _int('MAX_INVESTIGATION_QUERIES', 12, 1, 30))
    max_parallel_db_queries: int = field(default_factory=lambda: _int('MAX_PARALLEL_DB_QUERIES', 3, 1, 10))
    max_parallel_tasks: int = field(default_factory=lambda: _int('MAX_PARALLEL_TASKS', 4, 1, 12))
    max_plan_tasks: int = field(default_factory=lambda: _int('MAX_PLAN_TASKS', 16, 2, 40))
    max_plan_refinements: int = field(default_factory=lambda: _int('MAX_PLAN_REFINEMENTS', 2, 0, 5))
    max_refinement_tasks_per_round: int = field(default_factory=lambda: _int('MAX_REFINEMENT_TASKS_PER_ROUND', 4, 1, 10))
    max_task_retries: int = field(default_factory=lambda: _int('MAX_TASK_RETRIES', 1, 0, 3))
    task_timeout_seconds: int = field(default_factory=lambda: _int('TASK_TIMEOUT_SECONDS', 90, 5, 600))
    max_run_seconds: int = field(default_factory=lambda: _int('MAX_RUN_SECONDS', 240, 10, 1800))
    max_run_tokens: int = field(default_factory=lambda: _int('MAX_RUN_TOKENS', 250000, 1000, 5_000_000))
    max_run_cost_usd: float = field(default_factory=lambda: _float('MAX_RUN_COST_USD', 2.0, 0.0, 100.0))
    max_result_rows: int = field(default_factory=lambda: min(5000, max(10, int(os.getenv('MAX_RESULT_ROWS', '500')))))
    max_query_seconds: int = field(default_factory=lambda: min(60, max(1, int(os.getenv('MAX_QUERY_SECONDS', '10')))))
    global_max_db_connections: int = field(default_factory=lambda: _int('GLOBAL_MAX_DB_CONNECTIONS', 8, 1, 50))

    # Permissions (single-tenant profile; see docs/SECURITY_AND_LIMITS.md)
    allowed_tables: str = field(default_factory=lambda: os.getenv('ALLOWED_TABLES', ''))
    allow_reports: bool = field(default_factory=lambda: _bool('ALLOW_REPORTS', True))
    allow_forecasting: bool = field(default_factory=lambda: _bool('ALLOW_FORECASTING', True))

    # Observability
    log_format: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', 'text').strip().lower())

    # API and CORS
    cors_origins: str = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173'))
    api_token: str = field(default_factory=lambda: os.getenv('API_TOKEN', ''))

    def validate(self) -> None:
        """Validate required configuration at application startup."""
        if not self.openai_api_key:
            raise ConfigurationError(
                "CONFIG_OPENAI_MISSING: OPENAI_API_KEY is required. Configure it in your .env file.",
                code="CONFIG_OPENAI_MISSING"
            )
        if not self.openrouter_api_key:
            raise ConfigurationError(
                "CONFIG_OPENROUTER_MISSING: OPENROUTER_API_KEY is required for Jev routing.",
                code="CONFIG_OPENROUTER_MISSING"
            )
        if not self.database_url:
            raise ConfigurationError(
                "DATABASE_CONNECTION_FAILED: Could not connect to the configured MySQL database. Check host, port, credentials, and server status.",
                code="DATABASE_CONNECTION_FAILED"
            )
        if self.sql_generation_mode not in ("llm", "compiled"):
            raise ConfigurationError(
                f"CONFIG_INVALID: SQL_GENERATION_MODE must be 'llm' or 'compiled' (got '{self.sql_generation_mode}').",
                code="CONFIG_INVALID"
            )
        if self.sandbox_provider not in ("isolated_process", "e2b", "docker"):
            raise ConfigurationError(
                f"CONFIG_INVALID: SANDBOX_PROVIDER must be 'isolated_process', 'e2b', or 'docker' (got '{self.sandbox_provider}').",
                code="CONFIG_INVALID"
            )

    def pricing_for(self, model: str) -> tuple[float, float] | None:
        """Return (input, output) USD per 1M tokens if configured for the model."""
        for item in self.model_pricing.split(","):
            if "=" not in item:
                continue
            name, prices = item.split("=", 1)
            if name.strip() != model:
                continue
            try:
                inp, out = prices.split("/", 1)
                return float(inp), float(out)
            except ValueError:
                return None
        return None


settings = Settings()
