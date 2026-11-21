"""Secure, provider-independent Python Sandbox Execution Service."""
from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Literal
import uuid
import pandas as pd

from app.config import PROJECT_ROOT, settings
from app.exceptions import SandboxSecurityError, SandboxUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_output_bytes: int = 65536
    max_files: int = 25
    allow_network: bool = False


@dataclass(frozen=True)
class DataHandlingPolicy:
    allow_external_transfer: bool = False
    redacted_columns: tuple[str, ...] = ()


@dataclass
class SandboxArtifact:
    name: str
    kind: Literal["table", "chart", "figure", "json", "text"]
    path: str
    content: Any = None
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "path": self.path, "sha256": self.sha256, "content": self.content}


@dataclass
class SandboxExecutionResult:
    execution_id: str
    workspace_id: str
    status: Literal["COMPLETED", "ERROR", "TIMED_OUT", "SECURITY_VIOLATION"]
    exit_code: int
    stdout: str
    stderr: str
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[SandboxArtifact] = field(default_factory=list)
    code_hash: str = ""
    environment_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxWorkspace:
    workspace_id: str
    run_id: str
    task_id: str
    root_dir: Path
    staged_datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    execution_history: list[SandboxExecutionResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    limits: SandboxLimits = field(default_factory=SandboxLimits)


FORBIDDEN_IMPORTS = {
    "socket", "subprocess", "http", "urllib", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "poplib", "imaplib", "telnetlib", "xmlrpc",
    "pymysql", "mysql", "sqlalchemy", "sqlite3", "psycopg2", "psycopg", "pyodbc",
    "ctypes", "cffi", "multiprocessing", "winreg", "webbrowser", "shutil",
    "pty", "fcntl", "resource", "signal", "importlib", "pkgutil", "runpy",
    "code", "codeop", "socketserver", "asyncio", "ssl", "certifi",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
FORBIDDEN_ATTRS = {
    "__subclasses__", "__bases__", "__mro__", "__globals__", "__code__", "__closure__",
    "system", "popen", "spawn", "spawnl", "spawnlp", "spawnv", "spawnvp",
    "execl", "execle", "execlp", "execv", "execve", "execvp", "fork", "forkpty",
    "startfile", "kill", "killpg", "dlopen",
}
FORBIDDEN_OS_ATTRS = {
    "system", "popen", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe", "execl", "execle",
    "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "fork", "forkpty", "startfile", "kill", "killpg",
}


def validate_python_code_ast(code: str) -> tuple[bool, str | None, str | None]:
    if not code or not code.strip():
        return False, "EmptyCodeError", "No Python code was provided for execution."
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, "SyntaxError", f"SyntaxError on line {exc.lineno}: {exc.msg}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0].lower() in FORBIDDEN_IMPORTS:
                    raise SandboxSecurityError(f"Forbidden module import '{alias.name}' is not permitted in the analytical sandbox.")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0].lower() in FORBIDDEN_IMPORTS:
                raise SandboxSecurityError(f"Forbidden module import 'from {node.module}' is not permitted in the analytical sandbox.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise SandboxSecurityError(f"Forbidden builtin call '{node.func.id}()' is not permitted in the analytical sandbox.")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRS:
                raise SandboxSecurityError(f"Forbidden method call '.{node.func.attr}()' is not permitted in the analytical sandbox.")
        elif isinstance(node, ast.Attribute):
            if node.attr in ("__subclasses__", "__bases__", "__globals__", "__code__", "__closure__"):
                raise SandboxSecurityError(f"Forbidden reflection attribute '.{node.attr}' is not permitted in the analytical sandbox.")
            if isinstance(node.value, ast.Name) and node.value.id == "os" and node.attr in FORBIDDEN_OS_ATTRS:
                raise SandboxSecurityError(f"Forbidden OS process operation 'os.{node.attr}' is not permitted in the analytical sandbox.")
    return True, None, None


def _json_safe(val: Any) -> Any:
    if val is None or isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, float):
        return None if (pd.isna(val) or val in (float("inf"), float("-inf"))) else val
    if hasattr(val, "item") and callable(val.item):
        try:
            return _json_safe(val.item())
        except Exception:
            pass
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val]
    if hasattr(val, "isoformat"):
        return val.isoformat()
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return str(val)


def _runner_script_template() -> str:
    return r'''import json, math, os, platform, sys, traceback, warnings
import numpy as np, pandas as pd, scipy
from scipy import stats

WORKSPACE_DIR, USER_CODE_PATH, OUTPUT_JSON_PATH = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
PROJECT_ROOT_DIR = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else ""
SYS_PREFIX_DIR, SYS_BASE_PREFIX_DIR = os.path.abspath(sys.prefix), os.path.abspath(sys.base_prefix)
ALLOWED_READ_DIRS = [WORKSPACE_DIR, SYS_PREFIX_DIR, SYS_BASE_PREFIX_DIR]
if os.name == "nt":
    win_dir = os.environ.get("SYSTEMROOT", r"C:\Windows")
    ALLOWED_READ_DIRS.extend([os.path.abspath(os.path.join(win_dir, sub)) for sub in ("Fonts", "System32", "Globalization")])
else:
    ALLOWED_READ_DIRS.extend(["/usr/lib", "/usr/local/lib", "/usr/share/fonts", "/lib"])

def _norm(p):
    if isinstance(p, bytes): p = p.decode("utf-8", errors="replace")
    return os.path.normcase(os.path.abspath(str(p)))

WS_NORM, PROJ_NORM = _norm(WORKSPACE_DIR), (_norm(PROJECT_ROOT_DIR) if PROJECT_ROOT_DIR else "")
ALLOWED_READ_NORMS = [_norm(d) for d in ALLOWED_READ_DIRS if d]

def _in_dir(path_norm, dir_norm):
    return bool(dir_norm) and (path_norm == dir_norm or path_norm.startswith(dir_norm + os.sep))

def _is_secret_or_app_file(path_norm):
    base = os.path.basename(path_norm)
    if base == ".env" or base.startswith(".env."): return True
    if PROJ_NORM and _in_dir(path_norm, PROJ_NORM):
        return not (_in_dir(path_norm, WS_NORM) or _in_dir(path_norm, _norm(SYS_PREFIX_DIR)))
    return False

_USER_CODE_ACTIVE = False

def _sandbox_audit_hook(event, args):
    if not _USER_CODE_ACTIVE: return
    if event.startswith(("urllib.", "http.", "ftplib.", "smtplib.", "poplib.", "imaplib.")):
        raise PermissionError(f"Sandbox security violation: network event '{event}' is prohibited.")
    if event.startswith("socket.") and event != "socket.gethostname":
        raise PermissionError(f"Sandbox security violation: socket operation '{event}' is prohibited.")
    if event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.startfile", "os.fork", "os.kill", "winreg.OpenKey", "winreg.CreateKey"):
        raise PermissionError(f"Sandbox security violation: system event '{event}' is prohibited.")
    if event == "ctypes.dlopen":
        lib_name = str(args[0]) if args and args[0] is not None else ""
        lib_norm = _norm(lib_name) if (os.sep in lib_name or "/" in lib_name) else lib_name.lower()
        in_py_env = _in_dir(lib_norm, _norm(SYS_PREFIX_DIR)) or _in_dir(lib_norm, _norm(SYS_BASE_PREFIX_DIR))
        if not (in_py_env or lib_name.lower() in {"kernel32", "kernel32.dll", "psapi.dll", "ucrtbase.dll"}):
            raise PermissionError(f"Sandbox security violation: dynamic library load '{lib_name}' is prohibited.")
    if event == "open" and args:
        target = args[0]
        if isinstance(target, int): return
        mode = str(args[1]) if len(args) > 1 and args[1] is not None else "r"
        p_norm = _norm(target)
        if _is_secret_or_app_file(p_norm):
            raise PermissionError(f"Sandbox security violation: unauthorized file access to '{target}'.")
        if any(ch in mode for ch in ("w", "a", "x", "+")):
            if not _in_dir(p_norm, WS_NORM):
                raise PermissionError(f"Sandbox security violation: file writes outside workspace are prohibited ('{target}').")
        elif not any(_in_dir(p_norm, d) for d in ALLOWED_READ_NORMS):
            raise PermissionError(f"Sandbox security violation: file reads outside workspace/runtime are prohibited ('{target}').")
    elif event in ("os.listdir", "os.scandir", "os.stat") and args and args[0] is not None:
        target = args[0]
        if isinstance(target, int): return
        p_norm = _norm(target)
        if _is_secret_or_app_file(p_norm) or not any(_in_dir(p_norm, d) for d in ALLOWED_READ_NORMS):
            raise PermissionError(f"Sandbox security violation: unauthorized path inspection '{target}'.")
    elif event in ("os.remove", "os.unlink", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate") and args:
        if not _in_dir(_norm(args[0]), WS_NORM):
            raise PermissionError(f"Sandbox security violation: filesystem mutation outside workspace is prohibited ('{args[0]}').")
    elif event in ("os.rename", "os.replace") and len(args) >= 2:
        for arg in args[:2]:
            if not _in_dir(_norm(arg), WS_NORM):
                raise PermissionError(f"Sandbox security violation: filesystem move outside workspace is prohibited ('{arg}').")

def _clean_json(val):
    if val is None or isinstance(val, (bool, str, int)): return val
    if isinstance(val, float): return None if (math.isnan(val) or math.isinf(val)) else round(val, 6)
    if hasattr(val, "item") and callable(val.item):
        try: return _clean_json(val.item())
        except Exception: pass
    if isinstance(val, dict): return {str(k): _clean_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)): return [_clean_json(x) for x in val]
    if hasattr(val, "isoformat"): return val.isoformat()
    try:
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except Exception:
        return str(val)

manifest_path = os.path.join(WORKSPACE_DIR, "datasets_manifest.json")
DATASETS = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for ds_name, meta in manifest.items():
        with open(os.path.join(WORKSPACE_DIR, meta["rel_path"]), "r", encoding="utf-8") as df_fh:
            payload = json.load(df_fh)
        rows, cols = payload.get("rows", []), payload.get("columns", [])
        DATASETS[ds_name] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

datasets_list = list(DATASETS.values())
primary_df = datasets_list[0].copy() if datasets_list else pd.DataFrame()
_RESULTS = {"summary": {}, "records": [], "charts": [], "tables": [], "assumptions": [], "warnings": [], "generated_files": []}

def list_datasets():
    return {k: {"rows": int(len(v)), "columns": list(v.columns)} for k, v in DATASETS.items()}

def load_dataset(name=None):
    if name is None: return primary_df.copy()
    if name in DATASETS: return DATASETS[name].copy()
    raise KeyError(f"Dataset '{name}' not found. Available: {list(DATASETS.keys())}")

def save_result(summary, records=None, assumptions=None, warnings_list=None):
    if isinstance(summary, dict): _RESULTS["summary"].update(_clean_json(summary))
    if isinstance(records, list): _RESULTS["records"] = _clean_json(records[:200])
    elif isinstance(records, pd.DataFrame): _RESULTS["records"] = _clean_json(records.head(200).to_dict(orient="records"))
    if isinstance(assumptions, list): _RESULTS["assumptions"].extend(str(a) for a in assumptions)
    if isinstance(warnings_list, list): _RESULTS["warnings"].extend(str(w) for w in warnings_list)

def save_table(title, df_or_rows):
    rows = _clean_json(df_or_rows.head(100).to_dict(orient="records")) if isinstance(df_or_rows, pd.DataFrame) else (_clean_json(df_or_rows[:100]) if isinstance(df_or_rows, list) else [])
    _RESULTS["tables"].append({"title": str(title), "rows": rows})

def save_chart(title, kind, labels, values, unit="USD", x_label="", y_label="", series=None):
    clean_vals = [float(fv) if isinstance((fv := _clean_json(v)), (int, float)) else 0.0 for v in values]
    chart_obj = {"title": str(title), "kind": str(kind) if kind in ("bar", "line", "waterfall", "pie", "grouped_bar") else "bar", "labels": [str(l) for l in labels], "values": clean_vals, "unit": str(unit), "x_label": str(x_label), "y_label": str(y_label)}
    if series: chart_obj["series"] = _clean_json(series)
    _RESULTS["charts"].append(chart_obj)

def save_figure(fig, filename="figure.png"):
    safe_name = os.path.basename(str(filename)) or "figure.png"
    out_path = os.path.join(WORKSPACE_DIR, safe_name)
    if hasattr(fig, "savefig"):
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        _RESULTS["generated_files"].append(safe_name)
    elif hasattr(fig, "to_json"):
        json_name = safe_name if safe_name.endswith(".json") else safe_name + ".json"
        with open(os.path.join(WORKSPACE_DIR, json_name), "w", encoding="utf-8") as fh:
            fh.write(fig.to_json())
        _RESULTS["generated_files"].append(json_name)

def save_state(key, value):
    state_file = os.path.join(WORKSPACE_DIR, "session_state.json")
    data = json.load(open(state_file, "r", encoding="utf-8")) if os.path.exists(state_file) else {}
    data[str(key)] = _clean_json(value)
    with open(state_file, "w", encoding="utf-8") as fh: json.dump(data, fh)

def load_state(key, default=None):
    state_file = os.path.join(WORKSPACE_DIR, "session_state.json")
    return json.load(open(state_file, "r", encoding="utf-8")).get(str(key), default) if os.path.exists(state_file) else default

platform.uname()
with open(USER_CODE_PATH, "r", encoding="utf-8") as fh: user_code = fh.read()

if "matplotlib" in user_code:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot
    except Exception: pass
if "plotly" in user_code:
    try: import plotly.express, plotly.graph_objects
    except Exception: pass
if "sklearn" in user_code:
    try: import sklearn, threadpoolctl; threadpoolctl.threadpool_info()
    except Exception: pass
if "statsmodels" in user_code:
    try: import statsmodels.api
    except Exception: pass
if "polars" in user_code:
    try: import polars
    except Exception: pass

env_versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
sandbox_globals = {
    "__name__": "__main__", "__builtins__": __builtins__, "np": np, "pd": pd, "stats": stats, "scipy": scipy,
    "math": math, "json": json, "df": primary_df, "datasets": datasets_list, "DATASETS": DATASETS,
    "load_dataset": load_dataset, "list_datasets": list_datasets, "save_result": save_result,
    "save_table": save_table, "save_chart": save_chart, "save_figure": save_figure,
    "save_state": save_state, "load_state": load_state, "RESULTS": _RESULTS,
}

sys.addaudithook(_sandbox_audit_hook)
exit_code, error_type, error_message = 0, None, None
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    try:
        compiled = compile(user_code, "sandbox_analysis.py", "exec")
        _USER_CODE_ACTIVE = True
        exec(compiled, sandbox_globals)
    except PermissionError as sec_exc:
        exit_code, error_type, error_message = 2, "SecurityViolation", str(sec_exc)
    except Exception as exc:
        exit_code, error_type, error_message = 1, type(exc).__name__, traceback.format_exc()
    finally:
        _USER_CODE_ACTIVE = False
        for w in caught_warnings[:10]:
            _RESULTS["warnings"].append(f"{w.category.__name__}: {w.message}")

if not _RESULTS["summary"] and isinstance(sandbox_globals.get("result"), dict):
    _RESULTS["summary"] = _clean_json(sandbox_globals["result"])
for pkg_name in ("sklearn", "statsmodels", "polars", "plotly", "matplotlib"):
    if pkg_name in sys.modules:
        env_versions[pkg_name] = getattr(sys.modules[pkg_name], "__version__", "installed")

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fh:
    json.dump({
        "exit_code": exit_code, "error_type": error_type, "error_message": error_message,
        "summary": _clean_json(_RESULTS["summary"]), "records": _clean_json(_RESULTS["records"]),
        "charts": _clean_json(_RESULTS["charts"]), "tables": _clean_json(_RESULTS["tables"]),
        "assumptions": _RESULTS["assumptions"], "warnings": _RESULTS["warnings"],
        "generated_files": _RESULTS["generated_files"], "environment_versions": env_versions,
    }, fh, default=str)
sys.exit(exit_code)
'''


class SandboxProvider(ABC):
    @abstractmethod
    def create_workspace(self, run_id: str, task_id: str, limits: SandboxLimits | None = None) -> SandboxWorkspace: ...
    @abstractmethod
    def stage_dataset(self, workspace: SandboxWorkspace, name: str, rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]: ...
    @abstractmethod
    def execute_code(self, workspace: SandboxWorkspace, code: str, timeout_seconds: int | None = None) -> SandboxExecutionResult: ...
    @abstractmethod
    def destroy_workspace(self, workspace: SandboxWorkspace) -> None: ...


class IsolatedProcessSandboxProvider(SandboxProvider):
    """OS-process + AST + runtime audit-hook isolated Python sandbox (Windows & Linux)."""

    def __init__(self, base_dir: Path | None = None, python_executable: str | None = None):
        default_dir = Path(settings.dataset_store_path).resolve().parent / "sandboxes"
        self.base_dir = Path(base_dir or default_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self._lock = threading.Lock()
        self.cleanup_expired()

    def create_workspace(self, run_id: str, task_id: str, limits: SandboxLimits | None = None) -> SandboxWorkspace:
        ws_id = f"ws_{uuid.uuid4().hex[:12]}"
        ws_dir = (self.base_dir / ws_id).resolve()
        (ws_dir / "datasets").mkdir(parents=True, exist_ok=True)
        (ws_dir / ".cache").mkdir(parents=True, exist_ok=True)
        (ws_dir / "datasets_manifest.json").write_text("{}", encoding="utf-8")
        (ws_dir / "session_state.json").write_text("{}", encoding="utf-8")
        return SandboxWorkspace(
            workspace_id=ws_id, run_id=run_id, task_id=task_id, root_dir=ws_dir,
            limits=limits or SandboxLimits(timeout_seconds=settings.sandbox_timeout_seconds, max_memory_mb=settings.sandbox_max_memory_mb, max_output_bytes=settings.sandbox_max_output_bytes),
        )

    def stage_dataset(self, workspace: SandboxWorkspace, name: str, rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name).strip("_") or "dataset"
        rel_path = f"datasets/{safe_name}.json"
        (workspace.root_dir / rel_path).write_text(json.dumps({"columns": list(columns), "rows": _json_safe(rows)}, default=str), encoding="utf-8")
        manifest_path = workspace.root_dir / "datasets_manifest.json"
        with self._lock:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = {"rel_path": rel_path, "columns": list(columns), "row_count": len(rows)}
            manifest[name] = meta
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        workspace.staged_datasets[name] = meta
        return meta

    def execute_code(self, workspace: SandboxWorkspace, code: str, timeout_seconds: int | None = None) -> SandboxExecutionResult:
        exec_id = f"exec_{uuid.uuid4().hex[:10]}"
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
        eff_timeout = min(workspace.limits.timeout_seconds, max(1, timeout_seconds or workspace.limits.timeout_seconds))

        ok, err_type, err_msg = validate_python_code_ast(code)
        if not ok:
            res = SandboxExecutionResult(
                execution_id=exec_id, workspace_id=workspace.workspace_id, status="ERROR",
                exit_code=1, stdout="", stderr=err_msg or "", error_type=err_type, error_message=err_msg, code_hash=code_hash,
            )
            workspace.execution_history.append(res)
            return res

        step_num = len(workspace.execution_history) + 1
        user_code_path = workspace.root_dir / f"step_{step_num}_user.py"
        runner_path = workspace.root_dir / f"_runner_{step_num}.py"
        output_path = workspace.root_dir / f"_output_{step_num}.json"
        user_code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_runner_script_template(), encoding="utf-8")

        cache_dir = str(workspace.root_dir / ".cache")
        clean_env: dict[str, str] = {
            "PATH": str(Path(self.python_executable).resolve().parent),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": cache_dir, "POLARS_SKIP_CPU_CHECK": "1", "TEMP": cache_dir, "TMP": cache_dir,
            "HOME": cache_dir, "USERPROFILE": cache_dir, "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1",
        }
        if os.name == "nt":
            for k, d in (("SYSTEMROOT", r"C:\Windows"), ("COMSPEC", r"C:\Windows\System32\cmd.exe"), ("PROCESSOR_ARCHITECTURE", "AMD64"), ("NUMBER_OF_PROCESSORS", "4")):
                clean_env[k] = os.environ.get(k, d)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                [self.python_executable, "-I", "-B", str(runner_path), str(workspace.root_dir), str(user_code_path), str(output_path), str(PROJECT_ROOT)],
                cwd=str(workspace.root_dir), env=clean_env, capture_output=True, text=True, timeout=eff_timeout,
            )
            duration_ms = round((time.monotonic() - start) * 1000)
        except subprocess.TimeoutExpired as exc:
            duration_ms = round((time.monotonic() - start) * 1000)
            out_s = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout.decode("utf-8", "replace") if exc.stdout else "")
            err_s = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr.decode("utf-8", "replace") if exc.stderr else "")
            res = SandboxExecutionResult(
                execution_id=exec_id, workspace_id=workspace.workspace_id, status="TIMED_OUT", exit_code=-1,
                stdout=out_s[:workspace.limits.max_output_bytes], stderr=err_s[:workspace.limits.max_output_bytes],
                error_type="TimeoutExpired", error_message=f"Execution exceeded sandbox timeout limit ({eff_timeout}s).",
                duration_ms=duration_ms, code_hash=code_hash,
            )
            workspace.execution_history.append(res)
            return res

        max_b = workspace.limits.max_output_bytes
        stdout_trimmed, stderr_trimmed = (proc.stdout or "")[:max_b], (proc.stderr or "")[:max_b]
        out_data: dict[str, Any] = {}
        if output_path.exists():
            try:
                out_data = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not parse sandbox output JSON: %s", exc)

        err_t, err_m = out_data.get("error_type"), out_data.get("error_message")
        if proc.returncode == 2 or err_t == "SecurityViolation":
            raise SandboxSecurityError(err_m or "Sandbox runtime security policy violation.")

        status: Literal["COMPLETED", "ERROR", "TIMED_OUT", "SECURITY_VIOLATION"] = "COMPLETED" if proc.returncode == 0 and not err_t else "ERROR"
        if status == "ERROR" and not err_m:
            err_t, err_m = err_t or "ExecutionError", stderr_trimmed or f"Process exited with code {proc.returncode}"

        artifacts = [
            SandboxArtifact(name=fname, kind="json" if fname.endswith(".json") else "figure", path=str(fpath), sha256=hashlib.sha256(fpath.read_bytes()).hexdigest()[:16])
            for fname in out_data.get("generated_files", [])
            if (fpath := workspace.root_dir / fname).exists() and fpath.is_file()
        ]
        res = SandboxExecutionResult(
            execution_id=exec_id, workspace_id=workspace.workspace_id, status=status, exit_code=proc.returncode,
            stdout=stdout_trimmed, stderr=stderr_trimmed, error_type=err_t, error_message=err_m, duration_ms=duration_ms,
            summary=out_data.get("summary") or {}, records=out_data.get("records") or [], charts=out_data.get("charts") or [],
            tables=out_data.get("tables") or [], assumptions=out_data.get("assumptions") or [],
            warnings=out_data.get("warnings") or [], artifacts=artifacts, code_hash=code_hash,
            environment_versions=out_data.get("environment_versions") or {},
        )
        workspace.execution_history.append(res)
        return res

    def destroy_workspace(self, workspace: SandboxWorkspace) -> None:
        if workspace.root_dir.exists():
            shutil.rmtree(workspace.root_dir, ignore_errors=True)

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        removed, now = 0, time.time()
        if not self.base_dir.exists():
            return 0
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.name.startswith("ws_"):
                try:
                    if (now - child.stat().st_mtime) > max_age_seconds:
                        shutil.rmtree(child, ignore_errors=True)
                        removed += 1
                except Exception:
                    pass
        return removed


class ExternalSandboxProviderStub(SandboxProvider):
    def __init__(self, provider_name: str, policy: DataHandlingPolicy):
        self.provider_name = provider_name
        self.policy = policy

    def create_workspace(self, run_id: str, task_id: str, limits: SandboxLimits | None = None) -> SandboxWorkspace:
        if self.provider_name == "e2b" and not self.policy.allow_external_transfer:
            raise SandboxSecurityError("Data handling policy prohibits transferring database rows to external E2B sandbox when SANDBOX_ALLOW_EXTERNAL_DATA_TRANSFER=false.")
        raise SandboxUnavailableError(f"Sandbox provider '{self.provider_name}' is not configured or unavailable in this environment. No unsafe host fallback is permitted.")

    def stage_dataset(self, workspace: SandboxWorkspace, name: str, rows: list[dict[str, Any]], columns: list[str]) -> dict[str, Any]:
        raise SandboxUnavailableError(f"Sandbox provider '{self.provider_name}' unavailable.")

    def execute_code(self, workspace: SandboxWorkspace, code: str, timeout_seconds: int | None = None) -> SandboxExecutionResult:
        raise SandboxUnavailableError(f"Sandbox provider '{self.provider_name}' unavailable.")

    def destroy_workspace(self, workspace: SandboxWorkspace) -> None:
        return None


class SandboxExecutionService:
    """High-level facade managing sandbox lifecycle, dataset staging, and execution."""

    def __init__(self, provider: SandboxProvider | None = None, policy: DataHandlingPolicy | None = None):
        self.policy = policy or DataHandlingPolicy(allow_external_transfer=settings.sandbox_allow_external_data_transfer)
        if provider is not None:
            self.provider = provider
        elif settings.sandbox_provider == "isolated_process":
            self.provider = IsolatedProcessSandboxProvider()
        else:
            self.provider = ExternalSandboxProviderStub(settings.sandbox_provider, self.policy)

    def create_workspace(self, run_id: str, task_id: str, limits: SandboxLimits | None = None) -> SandboxWorkspace:
        return self.provider.create_workspace(run_id=run_id, task_id=task_id, limits=limits)

    def stage_dataframe(self, workspace: SandboxWorkspace, name: str, df: pd.DataFrame) -> dict[str, Any]:
        cols, rows = list(df.columns), (df.to_dict(orient="records") if not df.empty else [])
        if self.policy.redacted_columns:
            cols = [c for c in cols if c not in self.policy.redacted_columns]
            rows = [{k: v for k, v in r.items() if k not in self.policy.redacted_columns} for r in rows]
        return self.provider.stage_dataset(workspace, name, rows, cols)

    def stage_rows(self, workspace: SandboxWorkspace, name: str, columns: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
        return self.provider.stage_dataset(workspace, name, rows, columns)

    def execute(self, workspace: SandboxWorkspace, code: str, timeout_seconds: int | None = None) -> SandboxExecutionResult:
        return self.provider.execute_code(workspace, code, timeout_seconds=timeout_seconds)

    def destroy_workspace(self, workspace: SandboxWorkspace) -> None:
        self.provider.destroy_workspace(workspace)

    def cleanup_expired_workspaces(self, max_age_seconds: int = 3600) -> int:
        return self.provider.cleanup_expired(max_age_seconds=max_age_seconds) if isinstance(self.provider, IsolatedProcessSandboxProvider) else 0
