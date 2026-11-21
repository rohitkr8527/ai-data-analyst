# Security Model, Guardrails, and Operational Limits

## 1. Threat Model

User questions and autonomous agent code generation both introduce untrusted code or query risks:
- Prompt injection attempting SQL mutations, multi-statement injection, unauthorized table reads, or credential exfiltration.
- Generated Python code attempting filesystem traversal outside the task workspace, environment variable disclosure (`OPENAI_API_KEY`, `DATABASE_URL`, `OPENROUTER_API_KEY`), network connections, subprocess execution, dynamic `eval`/`exec` escapes, or CPU/memory/stdout exhaustion.
- Unverified statistical or numerical claims propagating into executive insights without reproduction.

## 2. Enforced Defense-in-Depth Safeguards

1. **Read-Only MySQL Gateway (`app/services/db.py`)**:
   - Connects using a dedicated read-only MySQL user (`ai_analyst`) with `SELECT` privileges only on `classicmodels.*` (unless explicitly scoped to alternate authorized test databases via `approved_tables`).
   - Parses every SQL query into a MySQL AST using **SQLGlot** before execution.
   - Blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, `TRUNCATE`, `MERGE`, `INTO`, transactions, and administrative statements anywhere in the AST (including CTEs and subqueries).
   - Blocks `SELECT *`, multi-statement semicolons (`;`), inline comments (`--`, `#`), and block comments (`/* */`).
   - Enforces table whitelisting and blocks dangerous functions (`SLEEP`, `BENCHMARK`, `LOAD_FILE`, `INTO OUTFILE`, `USER`, `DATABASE`, `VERSION`, `GET_LOCK`), while permitting standard analytical, date, math, conditional, and window functions.
   - Enforces bounded query concurrency (`BoundedSemaphore(max_parallel_db_queries)`), read timeouts (`MAX_QUERY_SECONDS`), and row caps (`MAX_RESULT_ROWS`).

2. **Isolated Python Sandbox Execution Service (`app/services/sandbox.py`)**:
   - Generated Python code **never executes inside the FastAPI server process** and **never falls back to unsafe host `exec()`/`eval()`**.
   - **Pre-Execution Static AST Gate**: Rejects disallowed imports (`socket`, `subprocess`, `http`, `urllib`, `requests`, `httpx`, `ctypes`, `importlib`, `pty`, `shutil`, `winreg`, `asyncio`, `multiprocessing`, etc.), banned calls (`eval`, `exec`, `compile`, `__import__`, `breakpoint`, `input`), and banned dunder attribute escapes (`__subclasses__`, `__bases__`, `__globals__`, `__code__`, etc.).
   - **Process Isolation & Secret Scrubbing**: Executes scripts in a fresh child subprocess inside a dedicated per-task workspace directory (`tmp/sandbox/<run_id>_<task_id>`). All API keys, database URLs, credentials, and host `.env` variables are stripped from `os.environ`.
   - **Runtime `sys.addaudithook` Enforcement**: After pre-warming approved scientific packages (`numpy`, `pandas`, `scipy`, `polars`, `statsmodels`, `scikit-learn`, `plotly`, `matplotlib`), installs an irreversible `sys.addaudithook` handler blocking runtime `socket.*` connections, `subprocess.*` / `os.system` / `os.posix_spawn` execution, Windows registry access, external DLL loading outside Python's environment/system runtime paths, file writes outside the task workspace, and reads targeting sensitive files (`.env`, `.git`, `.ssh`, application backend code outside site-packages/workspace).
   - **Resource & Output Limits**: Enforces wall-clock execution timeouts (`SANDBOX_TIMEOUT_SECONDS`), stdout/stderr byte caps (`SANDBOX_MAX_OUTPUT_BYTES`), and Unix virtual memory / CPU `resource` limits where supported.
   - **External Provider Policy (`ExternalSandboxProviderStub` & `DataHandlingPolicy`)**: External remote sandbox providers (e.g., E2B/Docker) raise explicit configuration errors if selected without provider setup and forbid uploading authorized datasets when `SANDBOX_ALLOW_EXTERNAL_DATA_TRANSFER=false`.

3. **Authorized Dataset Broker & Immutable Snapshots (`app/services/dataset_broker.py`, `app/services/dataset_store.py`)**:
   - Autonomous Python and Hypothesis agents have **zero direct database credentials**; extra datasets can only be requested through `AuthorizedDatasetBroker` -> `ReadOnlyGateway`, bounded by `max_dataset_requests` per task.
   - Datasets in `DatasetStore` are stored outside the sandbox with full **SHA-256 content hashes** so `EvidenceReviewerAgent.reproduce_finding()` can verify findings against exact immutable data snapshots.

4. **Bounded Autonomous Execution & Budget Enforcement (`app/agents/runtime.py`, `app/orchestration/validator.py`)**:
   - Autonomous agent loops (`SQLAnalyst.investigate_sql`, `PythonAnalysisAgent.investigate`, `HypothesisAgent.investigate_hypotheses`) obey strict `AgentLoopBudget` limits (`MAX_AGENT_STEPS`, `max_wall_seconds`, `max_dataset_requests`, `max_repairs`) and cooperative cancellation via `CancellationRegistry`.
   - Graph-level limits enforce `MAX_PLAN_REFINEMENTS`, `MAX_INVESTIGATION_QUERIES`, `MAX_RUN_TOKENS`, `MAX_RUN_COST_USD`, and `MAX_RUN_SECONDS`.

5. **Numerical Reproduction & Epistemic Guardrails (`app/agents/evidence_reviewer.py`, `app/agents/reporter.py`)**:
   - `EvidenceReviewerAgent.reproduce_finding()` independently re-executes saved Python code in a fresh sandbox workspace against the original SHA-256-tracked datasets; any discrepancy emits a `major` deterministic issue and blocks sufficiency (`recalculate` refinement).
   - Hypotheses are constrained by `causal_disclaimer`, and executive narratives are checked via `verify_narrative_numbers`.

## 3. Remaining Production Considerations

- **OS Container / MicroVM Hardening on Linux**: On Windows development hosts, `IsolatedProcessSandboxProvider` provides AST checking, scrubbed environment variables, workspace scoping, timeouts, and Python audit-hook containment. For multi-tenant production Linux deployments, run the sandbox worker inside an ephemeral Docker container (rootless, `--network=none`, `--read-only`, cgroups memory/CPU limits), gVisor (`runsc`), or microVM provider (Firecracker / E2B).
- **Single-Tenant Authentication**: Optional `X-API-Token` header gates API access; multi-tenant deployments should pair this with OIDC/JWT auth and per-tenant database role/row-level security.
- **Historical Data Window**: The `classicmodels` sample database contains records from `2003-01-06` through `2005-05-31`.
