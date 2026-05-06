"""
pipeline.py
-----------
Self-healing code generation pipeline for AutoInsight.

Flow per run:
    1. Select starting model via keyword heuristic.
    2. Build a schema-enriched generation prompt.
    3. Call Claude to generate Python code (Haiku by default).
    4. Execute the code in a subprocess sandbox.
    5. On failure: diagnose the error type, inject targeted context,
       append to conversation history, and call Claude to repair.
    6. On the final repair attempt: escalate to Sonnet as a hard fallback.
    7. Yield a StepResult after every sub-step so the UI can render
       progress in real time using st.status / generator streaming.

Key design decisions:
- Full conversation history is passed on every repair call so the model
  can see all previous attempts and avoid repeating the same mistake.
- diagnose_error() injects targeted context (column names, dtypes, correct
  file path) based on the error type rather than dumping the full schema
  every time, keeping the repair prompt focused.
- max_tokens is increased for Sonnet repair calls because Sonnet tends to
  reason more verbosely before outputting code.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import anthropic

from model_selector import (
    MODEL_HAIKU,
    MODEL_SONNET,
    MAX_RETRIES,
    select_model_for_attempt,
    model_label,
)
from code_executor import execute_code


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Represents the outcome of one sub-step in the pipeline."""
    step:    str            # "generate" | "execute" | "repair"
    attempt: int            # 1-based attempt number
    status:  str            # "running" | "success" | "fail" | "done"
    model:   Optional[str] = None
    code:    Optional[str] = None
    stdout:  Optional[str] = None
    stderr:  Optional[str] = None
    images:  List[bytes]   = field(default_factory=list)
    note:    Optional[str] = None  # extra human-readable annotation


@dataclass
class PipelineResult:
    """Final result returned after the pipeline terminates."""
    success:  bool
    attempts: int
    code:     str
    stdout:   str
    stderr:   str
    images:   List[bytes] = field(default_factory=list)


# ── prompt builders ───────────────────────────────────────────────────────────

def _build_schema_context(schema: dict) -> str:
    if not schema:
        return ""
    return textwrap.dedent(f"""
        Dataset schema:
        - Shape   : {schema['shape'][0]} rows x {schema['shape'][1]} columns
        - Columns : {schema['columns']}
        - Dtypes  : {schema['dtypes']}
        - Sample  :
        {schema['sample_str']}
    """).strip()


def _build_system_prompt(schema: dict) -> str:
    schema_ctx = _build_schema_context(schema)
    return textwrap.dedent(f"""
        You are a Python data analysis assistant.

        The dataset is already loaded as a pandas DataFrame called `df`.
        You may also read it from 'data.csv' in the working directory.

        Available libraries: pandas, numpy, matplotlib, seaborn, sklearn.
        Call plt.show() to render each chart — figures are saved automatically.

        {schema_ctx}

        Rules:
        - Return only valid Python code.
        - No markdown fences, no explanation text.
        - Do not redefine `df` unless you need a transformed copy.
    """).strip()


def _build_repair_message(code: str, stderr: str, extra_context: str) -> str:
    parts = [
        "The following code failed:",
        f"```python\n{code}\n```",
        f"Error:\n{stderr}",
    ]
    if extra_context:
        parts.append(f"Additional context: {extra_context}")
    parts.append(
        "Think step by step about the root cause, "
        "then return only the corrected Python code."
    )
    return "\n\n".join(parts)


# ── error diagnosis ───────────────────────────────────────────────────────────

def _diagnose_error(stderr: str, schema: dict) -> str:
    """
    Returns a targeted hint string based on the error type.
    Keeping hints specific prevents the model from being distracted by
    irrelevant schema information.
    """
    if not schema:
        return ""
    if re.search(r"KeyError|Column not found", stderr):
        return f"Actual column names are: {schema['columns']}"
    if re.search(r"FileNotFoundError", stderr):
        return "The dataset is at 'data.csv' in the current directory, already loaded as `df`."
    if re.search(r"TypeError|ValueError", stderr):
        return f"Column dtypes are: {schema['dtypes']}"
    if re.search(r"ModuleNotFoundError|ImportError", stderr):
        return "Only use: pandas, numpy, matplotlib, seaborn, sklearn."
    return ""


# ── code cleaning ─────────────────────────────────────────────────────────────

def _clean_code(raw: str) -> str:
    """Strip markdown fences that models sometimes include despite instructions."""
    raw = re.sub(r"```[\w]*\n?", "", raw)
    raw = re.sub(r"```", "", raw)
    return raw.strip()


# ── api call ──────────────────────────────────────────────────────────────────

def _call_claude(
    client: anthropic.Anthropic,
    messages: list,
    model: str,
    max_tokens: int = 1500,
) -> str:
    response = client.messages.create(
        model      = model,
        max_tokens = max_tokens,
        messages   = messages,
    )
    return response.content[0].text


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    prompt:   str,
    schema:   dict,
    csv_path: str,
    api_key:  str,
) -> Generator[StepResult, None, PipelineResult]:
    """
    Generator that yields StepResult objects as the pipeline progresses.
    The caller (Streamlit) iterates over these to update the UI in real time.
    The generator's return value is the final PipelineResult.

    Usage:
        gen = run_pipeline(...)
        try:
            while True:
                step = next(gen)
                # update UI with step
        except StopIteration as e:
            result = e.value
    """
    client        = anthropic.Anthropic(api_key=api_key)
    system_prompt = _build_system_prompt(schema)

    # Conversation history — grows with each repair turn.
    history = [{"role": "user", "content": f"{system_prompt}\n\nTask: {prompt}"}]

    # -- Step 1: Generate (always Haiku) --
    yield StepResult(
        step="generate", attempt=1, status="running",
        model=MODEL_HAIKU,
        note="Haiku (default)",
    )

    raw_code = _call_claude(client, history, MODEL_HAIKU)
    code     = _clean_code(raw_code)
    history.append({"role": "assistant", "content": code})

    yield StepResult(step="generate", attempt=1, status="done", model=MODEL_HAIKU, code=code)

    # -- Steps 2+: Execute / Repair loop --
    for attempt in range(MAX_RETRIES):
        attempt_num = attempt + 1

        # Execute
        yield StepResult(step="execute", attempt=attempt_num, status="running")
        exec_result = execute_code(code, csv_path)

        if exec_result["success"]:
            yield StepResult(
                step="execute", attempt=attempt_num, status="success",
                stdout=exec_result["stdout"],
                stderr=exec_result["stderr"],
                images=exec_result["images"],
            )
            return PipelineResult(
                success  = True,
                attempts = attempt_num,
                code     = code,
                stdout   = exec_result["stdout"],
                stderr   = exec_result["stderr"],
                images   = exec_result["images"],
            )

        # Execution failed
        yield StepResult(
            step="execute", attempt=attempt_num, status="fail",
            stdout=exec_result["stdout"],
            stderr=exec_result["stderr"],
        )

        if attempt == MAX_RETRIES - 1:
            break

        # Repair
        repair_model  = select_model_for_attempt(attempt + 1)
        extra_context = _diagnose_error(exec_result["stderr"], schema)
        repair_msg    = _build_repair_message(code, exec_result["stderr"], extra_context)

        history.append({"role": "user", "content": repair_msg})

        is_escalation = (repair_model == MODEL_SONNET)
        yield StepResult(
            step="repair", attempt=attempt_num + 1, status="running",
            model=repair_model,
            note="Escalating to Sonnet (final attempt)" if is_escalation else None,
        )

        repaired_raw = _call_claude(
            client, history, repair_model,
            max_tokens=2048 if repair_model == MODEL_SONNET else 1500,
        )
        code = _clean_code(repaired_raw)
        history.append({"role": "assistant", "content": code})

        yield StepResult(step="repair", attempt=attempt_num + 1, status="done",
                         model=repair_model, code=code)

    # All attempts exhausted
    return PipelineResult(
        success  = False,
        attempts = MAX_RETRIES,
        code     = code,
        stdout   = exec_result.get("stdout", ""),
        stderr   = exec_result.get("stderr", "All attempts exhausted."),
        images   = [],
    )
