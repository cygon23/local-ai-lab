"""
observability.py
----------------
Agent tracing with Arize Phoenix — running 100% locally.

WHAT IS OBSERVABILITY AND WHY DOES IT MATTER?

  Your agent makes decisions you can't always predict.
  It calls tools in unexpected orders. It misreads tool results.
  It loops when it shouldn't. It stops when it should keep going.

  Without observability, debugging means reading print statements.
  With observability, you get a visual timeline of every decision:

    ┌─────────────────────────────────────────────────────┐
    │ Run: "research fish prices and write a report"      │
    │                                                     │
    │  ├─ [0ms]  LLM call #1          → 847ms, 234 tokens│
    │  │    prompt: "You are an agent..."                 │
    │  │    output: <tool_call>web_search...</tool_call>  │
    │  │                                                  │
    │  ├─ [847ms] Tool: web_search    → 1.2s              │
    │  │    input: {"query": "fish prices Tanzania"}      │
    │  │    output: "[Result 1] ..."                      │
    │  │                                                  │
    │  ├─ [2.1s]  LLM call #2         → 623ms, 512 tokens│
    │  │    output: <tool_call>write_file...</tool_call>  │
    │  │                                                  │
    │  └─ [2.7s]  Tool: write_file    → 12ms              │
    │       input: {"filename": "report.txt", ...}        │
    │       output: "✅ File written"                     │
    └─────────────────────────────────────────────────────┘

  This is called a "trace" with nested "spans."
  Traces = full agent runs. Spans = individual steps within a run.
  This is the same concept as distributed tracing in microservices
  (OpenTelemetry, Jaeger, Zipkin) — Phoenix uses the same standard.

ARIZE PHOENIX:
  - Runs locally as a Python server (no cloud, no account needed)
  - Stores traces in a local SQLite database
  - Beautiful web UI at http://localhost:6006
  - Built on OpenTelemetry (industry standard)
  - Specifically designed for LLM/agent applications

HOW WE INSTRUMENT:
  We wrap key functions with "spans":
    - Each LLM call → one span
    - Each tool call → one span (child of the LLM span)
    - Full agent run → one root span containing all children
  
  This creates the nested tree structure you see in Phoenix UI.

SETUP:
  pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
  
  Then in a separate terminal:
  python -m phoenix.server.main
  
  Open: http://localhost:6006
"""

import time
import json
from typing import Any
from datetime import datetime

# We use a simple file-based trace log as fallback if Phoenix isn't installed
# This way V4 works even without Phoenix — observability is optional
TRACES_FILE = "data/traces.jsonl"

_phoenix_available = False
_tracer = None


def setup_phoenix() -> bool:
    """
    Initialize Arize Phoenix tracing.
    Returns True if Phoenix is available and set up, False otherwise.
    
    This is called once at app startup. If Phoenix isn't installed,
    we fall back to file-based logging — agent still works fine.
    """
    global _phoenix_available, _tracer

    try:
        import phoenix as px
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from openinference.instrumentation import OITracer

        # Start Phoenix server (or connect to existing)
        session = px.active_session()
        if session is None:
            session = px.launch_app()

        # Set up OpenTelemetry tracer pointing to Phoenix
        from phoenix.otel import register
        tracer_provider = register(
            project_name="local-ai-lab-v4",
            endpoint="http://localhost:6006/v1/traces"
        )

        _tracer = trace.get_tracer(__name__)
        _phoenix_available = True
        return True

    except ImportError:
        # Phoenix not installed — silent fallback
        _phoenix_available = False
        return False
    except Exception:
        _phoenix_available = False
        return False


def is_phoenix_available() -> bool:
    return _phoenix_available


# ─────────────────────────────────────────────
# SIMPLE FILE-BASED TRACING (always available)
# ─────────────────────────────────────────────
# Even without Phoenix, we log traces to a JSONL file.
# Each line = one complete agent run trace.
# You can read these with any JSON tool, or load them into Pandas.
# This is your "always-on" observability fallback.

class AgentTrace:
    """
    Captures a complete agent run as a structured trace.
    Written to disk as JSONL for inspection even without Phoenix.
    """
    def __init__(self, session_id: str, user_message: str, model: str):
        self.trace_id = f"trace_{int(time.time() * 1000)}"
        self.session_id = session_id
        self.user_message = user_message
        self.model = model
        self.start_time = time.time()
        self.end_time = None
        self.spans = []          # individual steps
        self.total_tokens = 0    # estimated
        self.llm_calls = 0
        self.tool_calls = []
        self.final_answer = ""
        self.error = None

    def add_llm_span(self, prompt_tokens: int, completion: str, duration_ms: int):
        """Record one LLM call."""
        self.llm_calls += 1
        completion_tokens = len(completion) // 4  # rough estimate
        self.total_tokens += prompt_tokens + completion_tokens

        self.spans.append({
            "type": "llm",
            "call_number": self.llm_calls,
            "prompt_tokens_est": prompt_tokens,
            "completion_tokens_est": completion_tokens,
            "duration_ms": duration_ms,
            "output_preview": completion[:200],
        })

    def add_tool_span(self, tool_name: str, args: dict, result: str, duration_ms: int):
        """Record one tool call."""
        self.tool_calls.append(tool_name)
        self.spans.append({
            "type": "tool",
            "tool_name": tool_name,
            "args": args,
            "result_preview": result[:300],
            "duration_ms": duration_ms,
        })

    def finish(self, final_answer: str, error: str = None):
        """Mark trace as complete."""
        self.end_time = time.time()
        self.final_answer = final_answer
        self.error = error

    def to_dict(self) -> dict:
        return {
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

    def save(self):
        """Append this trace to the JSONL file."""
        import os
        os.makedirs("data", exist_ok=True)
        with open(TRACES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), ensure_ascii=False) + "\n")


def load_recent_traces(n: int = 20) -> list[dict]:
    """Load the most recent N traces from the JSONL file."""
    import os
    if not os.path.exists(TRACES_FILE):
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
    # Return most recent first
    return list(reversed(traces))[-n:]