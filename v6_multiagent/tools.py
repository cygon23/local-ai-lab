"""
tools.py
--------
All tools via @tool decorator — same as V5.
Split into two groups:
  RESEARCH_TOOLS  → used by the Researcher agent
  EXECUTOR_TOOLS  → used by the Executor agent
  ALL_TOOLS       → combined list
"""

import os
import re
import json
import subprocess
import requests
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from langchain_core.tools import tool

WORKSPACE = Path("data/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LocalAIAgent/1.0)"}


# ── Research Tools ────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for current information using DuckDuckGo.
    Use for news, recent events, or any factual question requiring live data.
    """
    # Strategy 1: DDG HTML scrape
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(url, headers=HEADERS, timeout=15)
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
            if title_clean and snippet_clean:
                results.append(f"Title: {title_clean}\nURL: {real_url}\nSnippet: {snippet_clean}")
        if results:
            return f"Results for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception:
        pass

    # Strategy 2: Hacker News for tech topics
    try:
        hn_url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(query)}&tags=story&hitsPerPage={max_results}"
        r = requests.get(hn_url, timeout=10)
        hits = r.json().get("hits", [])
        if hits:
            results = []
            for h in hits[:max_results]:
                title = h.get("title", "")
                url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
                results.append(f"Title: {title}\nURL: {url}")
            return f"Results for '{query}':\n\n" + "\n\n---\n\n".join(results)
    except Exception:
        pass

    return f"No results found for '{query}'. Try fetch_webpage with a direct URL."


@tool
def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and return the readable text content of any public URL.
    Use for reading articles, documentation, or JSON APIs.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: URL must start with http:// or https://"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            text = json.dumps(response.json(), indent=2, ensure_ascii=False)
        else:
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
def search_knowledge_base(query: str) -> str:
    """
    Search indexed documents for information relevant to the query.
    Use when the user asks about uploaded or indexed documents.
    """
    try:
        from rag import get_retriever
        retriever = get_retriever(k=4)
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"[{i}] Source: {source}\n{doc.page_content[:500]}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Knowledge base error: {e}"


# ── Executor Tools ────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the exact result.
    Never guess math — always use this for any arithmetic.
    Examples: '2 ** 10', '5000 * (1 + 0.08) ** 10'
    """
    try:
        allowed = set("0123456789+-*/.() **eE")
        if not all(c in allowed for c in expression.replace(" ", "")):
            return f"Error: unsafe expression: {expression}"
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Math error: {e}"


@tool
def run_python(code: str) -> str:
    """
    Write and execute Python code. Returns stdout and stderr.
    Use for data processing, file generation, or complex computations.
    Always print() your results so they appear in the output.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=30
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


@tool
def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the workspace (data/workspace/).
    Creates the file if it does not exist, overwrites if it does.
    """
    try:
        path = WORKSPACE / filename
        path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {filename}"
    except Exception as e:
        return f"Write error: {e}"


@tool
def read_file(filename: str) -> str:
    """
    Read the contents of a file from the workspace (data/workspace/).
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
def list_files() -> str:
    """List all files in the agent workspace (data/workspace/)."""
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


# ── Tool groups ───────────────────────────────────────────────────────

RESEARCH_TOOLS = [web_search, fetch_webpage, search_knowledge_base]
EXECUTOR_TOOLS = [calculate, run_python, write_file, read_file, list_files]
ALL_TOOLS = RESEARCH_TOOLS + EXECUTOR_TOOLS