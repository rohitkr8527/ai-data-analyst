from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
from app.exceptions import ConfigurationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:

    database_url: str = os.getenv("DATABASE_URL", "")
    app_database_path: str = os.getenv("APP_DATABASE_PATH", str(BACKEND_ROOT / "data/app.db"))
    checkpoint_database_path: str = os.getenv("CHECKPOINT_DATABASE_PATH", str(BACKEND_ROOT / "data/checkpoints.db"))
    dataset_store_path: str = os.getenv("DATASET_STORE_PATH", str(BACKEND_ROOT / "data/datasets.db"))

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", 90.0))

    openai_sql_model: str = os.getenv("OPENAI_SQL_MODEL", "gpt-6-luna")
    openai_sql_research_model: str = os.getenv("OPENAI_SQL_RESEARCH_MODEL", "gpt-6.1-sol")
    openai_planner_model: str = os.getenv("OPENAI_PLANNER_MODEL", "gpt-6.1-sol")
    openai_rca_model: str = os.getenv("OPENAI_RCA_MODEL", "gpt-6.1-sol")
    openai_python_model: str = os.getenv("OPENAI_PYTHON_MODEL", "gpt-6.1-sol")
    openai_hypothesis_model: str = os.getenv("OPENAI_HYPOTHESIS_MODEL", "gpt-6.1-sol")
    openai_review_model: str = os.getenv("OPENAI_REVIEW_MODEL", "gpt-6.1-sol")
    openai_report_model: str = os.getenv("OPENAI_REPORT_MODEL", "gpt-6.1-terra")
    openai_insight_model: str = os.getenv("OPENAI_INSIGHT_MODEL", "gpt-6.1-terra")
    openai_schema_model: str = os.getenv("OPENAI_SCHEMA_MODEL", "gpt-6-luna")
    openai_tool_selection_model: str = os.getenv("OPENAI_TOOL_SELECTION_MODEL", "gpt-6-luna")

    sandbox_provider: str = os.getenv("SANDBOX_PROVIDER", "isolated_process").strip().lower()
    sandbox_timeout_seconds: int = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", 30))
    sandbox_max_memory_mb: int = int(os.getenv("SANDBOX_MAX_MEMORY_MB", 512))
    sandbox_max_output_bytes: int = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", 65536))
    sandbox_allow_external_data_transfer: bool = os.getenv("SANDBOX_ALLOW_EXTERNAL_DATA_TRANSFER", "false").lower() == "true"
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", 4))

    model_pricing: str = os.getenv("MODEL_PRICING_USD_PER_1M", "")

    openrouter_jev_model: str = os.getenv("OPENROUTER_JEV_MODEL", "typesafe/jev-1.14")
    openrouter_decisions_url: str = os.getenv("OPENROUTER_DECISIONS_URL", "https://openrouter.ai/api/alpha/decisions")

    sql_generation_mode: str = os.getenv("SQL_GENERATION_MODE", "compiled").strip().lower()

    max_sql_retries: int = int(os.getenv("MAX_SQL_RETRIES", 2))
    max_investigation_queries: int = int(os.getenv("MAX_INVESTIGATION_QUERIES", 12))
    max_parallel_db_queries: int = int(os.getenv("MAX_PARALLEL_DB_QUERIES", 3))
    max_parallel_tasks: int = int(os.getenv("MAX_PARALLEL_TASKS", 4))
    max_plan_tasks: int = int(os.getenv("MAX_PLAN_TASKS", 16))
    max_plan_refinements: int = int(os.getenv("MAX_PLAN_REFINEMENTS", 2))
    max_refinement_tasks_per_round: int = int(os.getenv("MAX_REFINEMENT_TASKS_PER_ROUND", 4))
    max_task_retries: int = int(os.getenv("MAX_TASK_RETRIES", 1))
    task_timeout_seconds: int = int(os.getenv("TASK_TIMEOUT_SECONDS", 90))
    max_run_seconds: int = int(os.getenv("MAX_RUN_SECONDS", 240))
    max_run_tokens: int = int(os.getenv("MAX_RUN_TOKENS", 250000))
    max_run_cost_usd: float = float(os.getenv("MAX_RUN_COST_USD", 2.0))
    max_result_rows: int = int(os.getenv("MAX_RESULT_ROWS", 500))
    max_query_seconds: int = int(os.getenv("MAX_QUERY_SECONDS", 10))
    global_max_db_connections: int = int(os.getenv("GLOBAL_MAX_DB_CONNECTIONS", 8))

    allowed_tables: str = os.getenv("ALLOWED_TABLES", "")
    allow_reports: bool = os.getenv("ALLOW_REPORTS", "true").lower() == "true"
    allow_forecasting: bool = os.getenv("ALLOW_FORECASTING", "true").lower() == "true"

    log_format: str = os.getenv("LOG_FORMAT", "text").strip().lower()

    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    api_token: str = os.getenv("API_TOKEN", "")

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ConfigurationError("CONFIG_OPENAI_MISSING: OPENAI_API_KEY is required.", code="CONFIG_OPENAI_MISSING")
        if not self.openrouter_api_key:
            raise ConfigurationError("CONFIG_OPENROUTER_MISSING: OPENROUTER_API_KEY is required.", code="CONFIG_OPENROUTER_MISSING")
        if not self.database_url:
            raise ConfigurationError("DATABASE_CONNECTION_FAILED: DATABASE_URL is required.", code="DATABASE_CONNECTION_FAILED")

    def pricing_for(self, model: str) -> tuple[float, float] | None:
        for item in self.model_pricing.split(","):
            if "=" in item:
                name, prices = item.split("=", 1)
                if name.strip() == model and "/" in prices:
                    inp, out = prices.split("/", 1)
                    return float(inp), float(out)
        return None


settings = Settings()

