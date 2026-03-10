# V4 Extended — Real World Agent

Extends V4 with four production layers:
real API tools, observability, MCP, and long-term memory.

## What's New vs V4 Base

| Layer | What It Adds |
|---|---|
| Web Tools | web_search, fetch_webpage, call_api — agent reaches the internet |
| Observability | Every run logged as a trace — timing, tokens, tool calls |
| MCP Client | Connect any MCP server — auto-registers its tools |
| Long-term Memory | Agent stores + recalls facts across sessions |

## Run It

```bash
cd v4_extended
pip install -r requirements.txt
streamlit run app.py
```

## New Files

```
v4_extended/
├── tools/
│   └── web_tools.py       ← web_search, fetch_webpage, call_api
├── observability.py        ← AgentTrace, file-based + Phoenix tracing
├── mcp_client.py           ← MCP stdio client, auto tool registration
└── memory.py               ← SQLite-backed long-term memory
```

## Layer 1 — Real API Tools

Three new tools registered in the same registry as before.
The agent loop does NOT change — just new entries in TOOL_FUNCTIONS.

  web_search(query)              → DuckDuckGo, no API key
  fetch_webpage(url)             → read any public page or JSON API
  call_api(url, method, headers, body) → full REST API calls

Try: "Search for the latest news about AI agents and summarize it"
Try: "Fetch https://wttr.in/Arusha?format=3 and tell me the weather"

Pattern for adding YOUR API:
  1. Write a function in web_tools.py that calls requests.get/post
  2. Add to TOOL_FUNCTIONS dict
  3. Add schema to TOOL_SCHEMAS list
  Done. Agent uses it automatically.

## Layer 2 — Observability

Every agent run is saved to data/traces.jsonl as a structured trace.
Each trace records: duration, LLM calls, tool calls, token estimates.

View traces in the app's "🔍 Traces" tab.

For visual tracing with Arize Phoenix:
  pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation
  python -m phoenix.server.main     # in a separate terminal
  Open: http://localhost:6006

Phoenix shows a visual timeline of every decision — like DevTools for your agent.
LangSmith is the same concept but cloud-based (used in V5 with LangChain).

## Layer 3 — MCP Integration

Connect to any MCP server and its tools register automatically:

```python
from mcp_client import connect_mcp_server
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

# Connect to filesystem MCP server
success, mcp_tools = connect_mcp_server(
    "filesystem",
    ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./data/workspace"]
)
if success:
    for tool in mcp_tools:
        TOOL_SCHEMAS.append(tool)
        # TOOL_FUNCTIONS auto-routes via mcp_client.call_mcp_tool()
```

The agent calls MCP tools exactly like local tools — no difference from its perspective.

Free MCP servers to try (requires Node.js):
  npx @modelcontextprotocol/server-filesystem /path
  npx @modelcontextprotocol/server-fetch
  npx @modelcontextprotocol/server-memory
  npx @modelcontextprotocol/server-sqlite --db-path ./data/agent.db

## Layer 4 — Long-Term Memory

Two new tools: store_memory and recall_memory.
The agent can call these itself when it learns something important.

Also: build_memory_prompt() injects stored facts into every system prompt.
The agent always "remembers" what it stored — without being asked.

Memory is stored in data/memory.db (SQLite).
Human-readable, query-able with any SQLite browser.

Try: "Remember that I'm building FishHappy for the Zanzibar fish market"
Then start a NEW session and ask: "What project am I working on?"
The agent will know — from memory, not from conversation history.

## The Full Tool List (V4 Extended)

Local:      get_datetime, calculate, read_file, write_file, list_workspace, run_python
Knowledge:  search_knowledge_base
Web:        web_search, fetch_webpage, call_api
Memory:     store_memory, recall_memory
MCP:        any tools from connected MCP servers (auto-registered)

## What V5 Will Show You

In V5 you'll rewrite this same agent using LangChain.
You'll see: AgentExecutor = our run_agent loop
            Tool = our TOOL_FUNCTIONS entry + TOOL_SCHEMAS entry
            Memory = our memory.py
            Callbacks = our observability.py

Everything maps 1:1. The framework adds convenience + ecosystem,
but the concepts are identical to what you built here.