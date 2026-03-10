"""
tools/__init__.py
-----------------
The tool registry — the single source of truth for all available tools.

HOW TOOL CALLING WORKS WITH OLLAMA/QWEN:
  We don't use a special API feature here. Instead, we use a technique
  called "structured prompting" — we tell the model exactly what tools
  exist and what format to use when it wants to call one.

  The model outputs something like:
    <tool_call>
    {"tool": "calculate", "args": {"expression": "12 * 8"}}
    </tool_call>

  Our agent loop detects this pattern, extracts the JSON, calls the real
  Python function, and feeds the result back to the model.

  This is called "function calling via prompting" — it works with ANY
  model that follows instructions well, no special API needed.

WHY NOT USE OLLAMA'S NATIVE TOOL CALLING?
  Ollama does support native tool calling for some models.
  But building it via prompting teaches you the REAL mechanism.
  LangChain, AutoGen — they all do exactly this under the hood.
  Once you understand the pattern, native tool calling is trivial.

TOOL SCHEMA FORMAT:
  Each tool is a dict describing:
    name        → what the model calls it
    description → when and why to use it (model reads this)
    parameters  → what arguments to pass
    returns     → what the model will get back

  IMPORTANT: Write descriptions FOR THE MODEL, not for humans.
  The model decides which tool to use based on these descriptions.
  Vague descriptions = wrong tool selections.
"""

from tools.utils import get_datetime, calculate
from tools.filesystem import read_file, write_file, list_workspace
from tools.code_runner import run_python
from tools.knowledge import search_knowledge_base
from tools.web_tools import web_search, fetch_webpage, call_api
from memory import store_memory, recall_memory


# ── Tool function registry ──
TOOL_FUNCTIONS = {
    "get_datetime": get_datetime,
    "calculate": calculate,
    "read_file": read_file,
    "write_file": write_file,
    "list_workspace": list_workspace,
    "run_python": run_python,
    "search_knowledge_base": search_knowledge_base,
    # NEW — real world
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "call_api": call_api,
    # NEW — memory
    "store_memory": store_memory,
    "recall_memory": recall_memory,
}


# ── Tool schema ──
# This is what we inject into the system prompt so the model
# knows what tools exist and how to call them.
TOOL_SCHEMAS = [
    {
        "name": "get_datetime",
        "description": "Get the current date and time. Use when the user asks about the current date, time, day of the week, or needs a timestamp.",
        "parameters": {},
        "returns": "Current datetime as a formatted string."
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any arithmetic, percentage calculations, unit conversions, or numerical computations. Do NOT try to compute math in your head — always use this tool.",
        "parameters": {
            "expression": "A Python-compatible math expression string, e.g. '15 * 8 + 200' or '(100 / 3) * 2'"
        },
        "returns": "The numerical result as a string."
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace. Use when the user asks you to read, summarize, or analyze a specific file.",
        "parameters": {
            "filename": "The filename to read from the workspace folder (e.g. 'notes.txt', 'data.csv')"
        },
        "returns": "File contents as a string, or an error message if the file doesn't exist."
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace. Use when the user asks you to save, create, or write a file. Also use when you produce a report, summary, or code that should be saved.",
        "parameters": {
            "filename": "The filename to write (e.g. 'report.txt', 'analysis.py')",
            "content": "The full text content to write to the file."
        },
        "returns": "Confirmation message with the file path."
    },
    {
        "name": "list_workspace",
        "description": "List all files currently in the workspace folder. Use when the user asks what files exist, or before reading a file to confirm it exists.",
        "parameters": {},
        "returns": "List of filenames in the workspace."
    },
    {
        "name": "run_python",
        "description": "Execute a Python code snippet and return the output. Use for data processing, analysis, generating charts descriptions, or any computation that requires code. The code runs in an isolated environment.",
        "parameters": {
            "code": "The Python code to execute as a string."
        },
        "returns": "stdout output from the code execution, or error traceback."
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the local knowledge base (indexed documents) for information relevant to a query. Use when the user asks about something that might be in their uploaded documents, or when you need to retrieve specific information from indexed files.",
        "parameters": {
            "query": "A natural language search query describing what information you're looking for."
        },
        "returns": "Relevant text chunks from indexed documents with their source filenames."
    },
    {
        "name": "web_search",
        "description": "Search the internet for current information, news, facts, or anything not in your training data or local documents. Use when the user asks about recent events, current prices, live data, or anything you're uncertain about.",
        "parameters": {
            "query": "The search query string.",
            "max_results": "Number of results to return (default: 5, max: 10)."
        },
        "returns": "Search result titles, snippets, and URLs."
    },
    {
        "name": "fetch_webpage",
        "description": "Fetch and read the text content of a specific webpage or REST API endpoint URL. Use after web_search to read the full content of a result, or when the user provides a specific URL to analyze.",
        "parameters": {
            "url": "The full URL to fetch (must start with http:// or https://).",
            "max_chars": "Maximum characters to return (default: 3000)."
        },
        "returns": "The readable text content of the page, or JSON if the URL returns JSON."
    },
    {
        "name": "store_memory",
        "description": "Save an important fact to long-term memory so you remember it in future sessions. Use when the user tells you something important about themselves, their projects, or their preferences. Also use when you discover a key fact worth remembering.",
        "parameters": {
            "key": "A short identifier for this memory (e.g. 'user_name', 'project_fishhappy', 'user_prefers_python')",
            "value": "The fact to remember as a clear, complete sentence.",
            "category": "Category: 'user_profile', 'projects', 'preferences', or 'general'"
        },
        "returns": "Confirmation that the memory was stored."
    },
    {
        "name": "recall_memory",
        "description": "Search your long-term memory for facts you have stored in past sessions. Use when you need to remember something about the user or their projects.",
        "parameters": {
            "query": "What you want to remember — a keyword or topic."
        },
        "returns": "Matching memories from past sessions."
    },
    {
        "name": "call_api",
        "description": "Make a generic HTTP API call (GET, POST, PUT, DELETE) to any REST API endpoint. Use when you need to interact with an API that requires specific methods, headers, or a request body. For simple GET requests to public pages, prefer fetch_webpage instead.",
        "parameters": {
            "url": "The full API endpoint URL.",
            "method": "HTTP method: GET, POST, PUT, DELETE (default: GET).",
            "headers_json": "JSON string of HTTP headers, e.g. '{\"Authorization\": \"Bearer token\"}'. Use '{}' if none.",
            "body_json": "JSON string of the request body for POST/PUT. Use '{}' if none."
        },
        "returns": "HTTP status code and response body (JSON or text)."
    },
]


def get_tools_prompt() -> str:
    """
    Build the tool section of the system prompt.
    This is injected before every agent loop iteration.

    The model reads this and decides whether and how to call tools.
    The format tag <tool_call> is our custom protocol —
    the agent loop watches for this exact tag in the model output.
    """
    tools_description = "\n\n".join([
        f"**{t['name']}**\n"
        f"  When to use: {t['description']}\n"
        f"  Parameters: {t['parameters'] if t['parameters'] else 'none'}\n"
        f"  Returns: {t['returns']}"
        for t in TOOL_SCHEMAS
    ])

    return f"""
You are an autonomous AI agent with access to the following tools:

{tools_description}

---
TOOL CALLING PROTOCOL:
When you need to use a tool, output ONLY this exact format and nothing else:

<tool_call>
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

RULES:
- Use ONE tool at a time. After each tool result, decide the next step.
- If you have enough information to answer, do NOT call more tools.
- For tools with no parameters, use: {{"tool": "tool_name", "args": {{}}}}
- When you have a FINAL answer, write it normally — no tool_call tags.
- Think step by step before deciding which tool (if any) to use.
- If no tool is needed, answer directly.
"""


def call_tool(tool_name: str, args: dict) -> str:
    """
    Execute a tool by name with the given arguments.
    Returns the result as a string (always — the model reads strings).

    This is the bridge between the model's decision and real Python execution.
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_FUNCTIONS.keys())}"

    func = TOOL_FUNCTIONS[tool_name]

    try:
        result = func(**args)
        return str(result)
    except TypeError as e:
        return f"Error: Wrong arguments for tool '{tool_name}': {e}"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {e}"