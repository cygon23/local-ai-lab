"""
observability.py
----------------
Real Phoenix OSS tracing via OpenTelemetry.

HOW IT WORKS:
  Phoenix runs locally at localhost:6006.
  We use arize-phoenix-otel to register a tracer provider that
  sends spans to Phoenix over HTTP.

  Every LLM call and tool call becomes a "span" inside a "trace".
  Phoenix displays them as a visual timeline.

SETUP (one time):
  pip install arize-phoenix-otel openinference-instrumentation

  Start Phoenix in a separate terminal:
  python -m phoenix.server.main   (or: pip install arize-phoenix then phoenix serve)

  Then run the agent normally — traces appear at localhost:6006 automatically.

WHAT WE INSTRUMENT MANUALLY:
  We don't use auto_instrument=True because our Ollama calls are
  plain HTTP requests, not an OpenAI SDK call.
  We create spans manually for:
    - Each full agent run      → root span
    - Each LLM call            → child span (kind=LLM)
    - Each tool call           → child span (kind=TOOL)

  This gives Phoenix the full nested trace it needs to render the timeline.
"""

import time
import json
import os
from datetime import datetime
from pathlib import Path

# ── Fallback file-based tracing (always available, no dependencies) ──
TRACES_FILE = "data/traces.jsonl"
Path("data").mkdir(exist_ok=True)

_phoenix_available = False
_tracer = None
_tracer_provider = None


def setup_phoenix(
    endpoint: str = "http://localhost:6006/v1/traces",
    project_name: str = "local-ai-lab-v4"
) -> bool:
    """
    Connect to Phoenix OSS via OpenTelemetry.
    Returns True if connected, False if Phoenix packages not installed.

    Based on: https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel
    """
    global _phoenix_available, _tracer, _tracer_provider

    try:
        from phoenix.otel import register
        from opentelemetry import trace

        # Set endpoint via env so phoenix.otel picks it up automatically
        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", endpoint)
        os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT_GRPC", "http://localhost:4317")

        # Register tracer provider pointing to local Phoenix
        # auto_instrument=False because we instrument manually (Ollama is not OpenAI SDK)
        _tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,          # http://localhost:6006/v1/traces
            auto_instrument=False,
            batch=False,
        )

        _tracer = trace.get_tracer(__name__)
        _phoenix_available = True
        return True

    except ImportError:
        # arize-phoenix-otel not installed — silent fallback to file logging
        _phoenix_available = False
        return False
    except Exception as e:
        # Phoenix server not running or connection refused — still fallback
        _phoenix_available = False
        return False


def is_phoenix_available() -> bool:
    return _phoenix_available


# ── OpenTelemetry span attributes (OpenInference semantic conventions) ──
# These are the standard keys Phoenix uses to render traces correctly.
# https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md

INPUT_VALUE       = "input.value"
OUTPUT_VALUE      = "output.value"
LLM_MODEL_NAME    = "llm.model_name"
LLM_TOKEN_COUNT   = "llm.token_count.total"
TOOL_NAME         = "tool.name"
TOOL_PARAMETERS   = "tool.parameters"
SESSION_ID_KEY    = "session.id"
SPAN_KIND_KEY     = "openinference.span.kind"


class AgentTrace:
    """
    Captures one complete agent run.

    Dual-write:
      1. OpenTelemetry spans → Phoenix UI at localhost:6006
      2. JSONL file → data/traces.jsonl (always, as fallback)

    Usage:
      trace = AgentTrace(session_id, user_message, model)
      trace.add_llm_span(...)
      trace.add_tool_span(...)
      trace.finish(final_answer)
      trace.save()
    """

    def __init__(self, session_id: str, user_message: str, model: str):
        self.trace_id = f"trace_{int(time.time() * 1000)}"
        self.session_id = session_id
        self.user_message = user_message
        self.model = model
        self.start_time = time.time()
        self.end_time = None
        self.spans = []
        self.total_tokens = 0
        self.llm_calls = 0
        self.tool_calls = []
        self.final_answer = ""
        self.error = None

        # Start root OTel span if Phoenix is available
        self._root_span = None
        self._root_ctx = None
        if _phoenix_available and _tracer:
            try:
                from opentelemetry import trace, context
                self._root_span = _tracer.start_span(
                    name="agent.run",
                    attributes={
                        SPAN_KIND_KEY: "AGENT",
                        INPUT_VALUE: user_message,
                        LLM_MODEL_NAME: model,
                        SESSION_ID_KEY: session_id,
                    }
                )
                self._root_ctx = trace.use_span(self._root_span, end_on_exit=False)
                self._root_ctx.__enter__()
            except Exception:
                self._root_span = None

    def add_llm_span(self, prompt_tokens: int, completion: str, duration_ms: int):
        """Record one LLM call as a child span."""
        self.llm_calls += 1
        completion_tokens = len(completion) // 4
        self.total_tokens += prompt_tokens + completion_tokens

        span_data = {
            "type": "llm",
            "call_number": self.llm_calls,
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "duration_ms": duration_ms,
            "output_preview": completion[:200],
        }
        self.spans.append(span_data)

        # OTel child span
        if _phoenix_available and _tracer and self._root_span:
            try:
                from opentelemetry import trace
                with _tracer.start_as_current_span(
                    name="llm.call",
                    attributes={
                        SPAN_KIND_KEY: "LLM",
                        LLM_MODEL_NAME: self.model,
                        OUTPUT_VALUE: completion[:500],
                        LLM_TOKEN_COUNT: prompt_tokens + completion_tokens,
                    }
                ):
                    pass  # span auto-ends on exit
            except Exception:
                pass

    def add_tool_span(self, tool_name: str, args: dict, result: str, duration_ms: int):
        """Record one tool call as a child span."""
        self.tool_calls.append(tool_name)
        span_data = {
            "type": "tool",
            "tool_name": tool_name,
            "args": args,
            "result_preview": result[:300],
            "duration_ms": duration_ms,
        }
        self.spans.append(span_data)

        # OTel child span
        if _phoenix_available and _tracer and self._root_span:
            try:
                with _tracer.start_as_current_span(
                    name=f"tool.{tool_name}",
                    attributes={
                        SPAN_KIND_KEY: "TOOL",
                        TOOL_NAME: tool_name,
                        TOOL_PARAMETERS: json.dumps(args),
                        INPUT_VALUE: json.dumps(args),
                        OUTPUT_VALUE: result[:500],
                    }
                ):
                    pass
            except Exception:
                pass

    def finish(self, final_answer: str, error: str = None):
        """Mark trace complete and close the root OTel span."""
        self.end_time = time.time()
        self.final_answer = final_answer
        self.error = error

        # Close root OTel span
        if self._root_span:
            try:
                self._root_span.set_attribute(OUTPUT_VALUE, final_answer[:500])
                self._root_span.set_attribute(
                    "llm.token_count.total", self.total_tokens
                )
                if error:
                    from opentelemetry.trace import StatusCode
                    self._root_span.set_status(StatusCode.ERROR, error)
                if self._root_ctx:
                    self._root_ctx.__exit__(None, None, None)
                self._root_span.end()
            except Exception:
                pass

    def save(self):
        """Write trace to JSONL file (always) and flush OTel (if Phoenix)."""
        trace_dict = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "timestamp": datetime.fromtimestamp(self.start_time).isoformat(),
            "user_message": self.user_message,
            "model": self.model,
            "duration_ms": int((self.end_time - self.start_time) * 1000) if self.end_time else None,
            "llm_calls": self.llm_calls,
            "tool_calls_made": self.tool_calls,
            "total_tokens_est": self.total_tokens,
            "final_answer_preview": self.final_answer[:300],
            "error": self.error,
            "spans": self.spans,
        }
        Path("data").mkdir(exist_ok=True)
        with open(TRACES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_dict, ensure_ascii=False) + "\n")

        # Flush OTel spans to Phoenix
        if _phoenix_available and _tracer_provider:
            try:
                _tracer_provider.force_flush()
            except Exception:
                pass


def load_recent_traces(n: int = 20) -> list[dict]:
    """Load the most recent N traces from the JSONL file."""
    if not Path(TRACES_FILE).exists():
        return []
    traces = []
    with open(TRACES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(traces))[:n]