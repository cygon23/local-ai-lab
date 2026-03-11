"""
tools.py
--------
All agent tools defined using LangChain's @tool decorator.

WHAT CHANGED FROM V4:
  V4: two separate dicts — TOOL_FUNCTIONS and TOOL_SCHEMAS
      Schema was manually written JSON describing each tool.
      The agent loop parsed <tool_call> tags and dispatched manually.

  V5: one @tool decorated function per tool.
      LangChain reads the function signature + docstring automatically
      to build the schema. No manual schema writing.
      AgentExecutor handles dispatch — no manual parsing.

  The actual tool logic (web search, file ops, etc.) is identical.
  Only the registration pattern changed.

HOW @tool WORKS:
  @tool
  def my_tool(param: str) -> str:
      "Description the LLM reads to decide when to use this tool."
      return do_something(param)

  LangChain extracts:
    - name       → function name
    - description → docstring (first line)
    - args schema → type hints on parameters

  That's it. No JSON schemas to maintain.
"""

import os
import re
import json
import subprocess
import requests
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from langchain_core.tools import tool

# Workspace for file operations
WORKSPACE = Path("data/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; LocalAIAgent/1.0)"
}


# ── Web Tools ────────────────────────────────────────────────────────


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information using DuckDuckGo.
    Use for news, recent events, or any factual question requiring live data.
    Returns titles, URLs, and snippets for the top results.
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        html = response.text
        results = []

        blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>',
            html, re.DOTALL
        )

        for href, title, snippet in blocks[:max_results]:
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            snippet_clean = re.sub(r"<[^>]+>", "", snippet).strip()
            real_url = re.sub(r"//duckduckgo\.com/l/\?uddg=", "", href)
            if real_url.startswith("//"):
                real_url = "https:" + real_url
            results.append(f"Title: {title_clean}\nURL: {real_url}\nSnippet: {snippet_clean}")

        if results:
            return f"Results for '{query}':\n\n" + "\n\n---\n\n".join(results)

        # Fallback to instant answer
        params = {"q": query, "format": "json", "no_html": "1"}
        r2 = requests.get("https://api.duckduckgo.com/", params=params, timeout=10, headers=HEADERS)
        data = r2.json()
        if data.get("AbstractText"):
            return data["AbstractText"]

        return (
            f"No results for '{query}'. "
            f"Try fetch_webpage with a direct URL instead."
        )
    except Exception as e:
        return f"Search error: {e}"


@tool
def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and return the readable text content of any public URL.
    Use for reading articles, documentation, JSON APIs, or any webpage.
    Strips HTML tags and returns clean text.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: URL must start with http:// or https://"

        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                data = response.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
                return (text[:max_chars] + "\n...[truncated]") if len(text) > max_chars else text
            except Exception:
                pass

        html = response.text
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<(p|div|br|li|h[1-6]|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return (text[:max_chars] + "\n...[truncated]") if len(text) > max_chars else text
    except Exception as e:
        return f"Fetch error: {e}"


@tool
def call_api(url: str, method: str = "GET", headers_json: str = "{}", body_json: str = "{}") -> str:
    """
    Make a generic HTTP API call to any REST endpoint.
    Use for APIs requiring specific methods, headers, or request bodies.
    headers_json and body_json must be valid JSON strings.
    """
    try:
        extra_headers = json.loads(headers_json) if headers_json.strip() != "{}" else {}
        body = json.loads(body_json) if body_json.strip() != "{}" else None
        merged = {**HEADERS, **extra_headers}

        response = requests.request(method.upper(), url, headers=merged, json=body, timeout=15)
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            text = json.dumps(response.json(), indent=2, ensure_ascii=False)
        else:
            text = response.text

        result = f"HTTP {response.status_code}:\n{text}"
        return result[:3000] + "\n...[truncated]" if len(result) > 3000 else result
    except Exception as e:
        return f"API error: {e}"


# ── Math ─────────────────────────────────────────────────────────────


@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the exact result.
    Use for any arithmetic, percentages, compound interest, unit conversions.
    Never guess math — always use this tool.
    Examples: '2 ** 10', '5000 * (1 + 0.08) ** 10', '100 / 3'
    """
    try:
        allowed = set("0123456789+-*/.() **eE")
        if not all(c in allowed for c in expression.replace(" ", "")):
            return f"Error: unsafe expression: {expression}"
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Math error: {e}"


# ── File Operations ───────────────────────────────────────────────────


@tool
def read_file(filename: str) -> str:
    """
    Read the contents of a file from the agent workspace (data/workspace/).
    Use to check existing file contents before writing or appending.
    """
    try:
        path = WORKSPACE / filename
        if not path.exists():
            files = [f.name for f in WORKSPACE.iterdir() if f.is_file()]
            return f"File '{filename}' not found. Available: {files}"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Read error: {e}"


@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the agent workspace (data/workspace/).
    Creates the file if it does not exist, overwrites if it does.
    Use for saving results, reports, code, or any output.
    """
    try:
        path = WORKSPACE / filename
        path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {filename}"
    except Exception as e:
        return f"Write error: {e}"


@tool
def list_files() -> str:
    """
    List all files currently in the agent workspace (data/workspace/).
    Use before reading or writing to see what already exists.
    """
    try:
        files = list(WORKSPACE.iterdir())
        if not files:
            return "Workspace is empty."
        return "\n".join(
            f"{f.name} ({f.stat().st_size} bytes)"
            for f in sorted(files) if f.is_file()
        )
    except Exception as e:
        return f"List error: {e}"


# ── Python Execution ──────────────────────────────────────────────────


@tool
def run_python(code: str) -> str:
    """
    Write and execute Python code. Returns stdout and stderr.
    Use for data processing, generating files, computations too complex for calculate.
    The code runs in a subprocess with a 30 second timeout.
    Always print() your results so they appear in the output.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout}"
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return output.strip() or "Code ran with no output."
    except subprocess.TimeoutExpired:
        return "Code timed out after 30 seconds."
    except Exception as e:
        return f"Execution error: {e}"


# ── All tools as a list for AgentExecutor ─────────────────────────────

ALL_TOOLS = [
    web_search,
    fetch_webpage,
    call_api,
    calculate,
    read_file,
    write_file,
    list_files,
    run_python,
]