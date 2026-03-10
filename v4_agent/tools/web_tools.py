"""
tools/web_tools.py
------------------
Real-world API tools: web search, HTTP fetch, page scraping.

NO API KEYS REQUIRED for these tools.
  - DuckDuckGo Instant Answer API: free, no auth, no rate limit for light use
  - HTTP fetch: just requests to any public URL
  - Scraper: requests + basic HTML stripping

THIS IS THE BRIDGE FROM LOCAL TO THE REAL WORLD.

Once you understand this pattern, every API becomes a tool:
  - Your own backend API → same pattern, different URL
  - OpenWeatherMap, NewsAPI, GitHub API → same pattern, add API key in header
  - Any REST service → same pattern

The agent loop does NOT change at all. You just add functions
to TOOL_FUNCTIONS and schemas to TOOL_SCHEMAS. That's it.
The ReAct loop is API-agnostic by design.

ARCHITECTURE LESSON:
  Notice all three functions follow the same contract:
    - Accept simple typed arguments (strings, ints)
    - Return a single string
    - Never raise exceptions — catch and return error strings
  
  This is the "tool contract." The model receives strings.
  It cannot receive Python dicts, lists, or objects.
  Everything must be serialized to string before returning.
  This constraint forces clean, simple tool interfaces.
"""

import requests
import json
import re
from urllib.parse import quote_plus, urlparse


# ── 1. DuckDuckGo Web Search ──

DDGO_URL = "https://api.duckduckgo.com/"

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo's Instant Answer API.
    Returns a formatted list of results with titles, URLs, and snippets.

    WHY DUCKDUCKGO?
      - No API key required
      - No rate limiting for reasonable use
      - Returns structured JSON
      - Privacy-respecting (good for local agents)

    LIMITATION:
      DDG Instant Answers is not a full search API — it returns
      the "instant answer" box + related topics, not a full SERP.
      For full search results, you'd use SerpAPI, Brave Search API,
      or Tavily (all have free tiers). Same pattern, different URL + key.
    """
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        response = requests.get(DDGO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        # Abstract (the main instant answer)
        if data.get("AbstractText"):
            results.append(
                f"[Instant Answer]\n"
                f"{data['AbstractText']}\n"
                f"Source: {data.get('AbstractURL', 'N/A')}"
            )

        # Related topics (actual search-like results)
        topics = data.get("RelatedTopics", [])
        count = 0
        for topic in topics:
            if count >= max_results:
                break
            # Topics can be nested (sub-categories)
            if "Topics" in topic:
                for sub in topic["Topics"]:
                    if count >= max_results:
                        break
                    text = sub.get("Text", "")
                    url = sub.get("FirstURL", "")
                    if text:
                        results.append(f"[Result {count+1}]\n{text}\nURL: {url}")
                        count += 1
            else:
                text = topic.get("Text", "")
                url = topic.get("FirstURL", "")
                if text:
                    results.append(f"[Result {count+1}]\n{text}\nURL: {url}")
                    count += 1

        if not results:
            return (
                f"No instant answer found for '{query}'.\n"
                f"Try fetch_webpage with a specific URL, or rephrase the query."
            )

        return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)

    except requests.exceptions.Timeout:
        return f"Search timed out for query: '{query}'"
    except requests.exceptions.ConnectionError:
        return "No internet connection available."
    except Exception as e:
        return f"Search error: {e}"


# ── 2. HTTP Fetch / Page Reader ──

def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """
    Fetch the text content of any public webpage or API endpoint.

    USE CASES:
      - Read a news article the agent found via web_search
      - Call a REST API that returns JSON
      - Read documentation pages
      - Fetch any public data source

    We strip HTML tags to return clean text.
    For JSON APIs, we return formatted JSON directly.

    max_chars: truncate output to avoid flooding the context window.
    """
    try:
        # Validate URL format
        parsed = urlparse(url)
        if not parsed.scheme in ("http", "https"):
            return f"Error: URL must start with http:// or https://"

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LocalAIAgent/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # JSON response — format it nicely
        if "application/json" in content_type:
            try:
                data = response.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n... [truncated]"
                return f"JSON response from {url}:\n{text}"
            except Exception:
                pass

        # HTML response — strip tags
        html = response.text

        # Remove script and style blocks entirely
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)

        # Replace block elements with newlines
        html = re.sub(r"<(p|div|br|li|h[1-6]|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)

        # Strip remaining tags
        text = re.sub(r"<[^>]+>", "", html)

        # Clean whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated, use a smaller page or specific section]"

        if not text:
            return f"Page fetched but no readable text found at {url}"

        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Timeout fetching {url}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP error {e.response.status_code} for {url}"
    except requests.exceptions.ConnectionError:
        return f"Cannot reach {url} — check internet connection."
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ── 3. Generic HTTP API Call ──

def call_api(
    url: str,
    method: str = "GET",
    headers_json: str = "{}",
    body_json: str = "{}"
) -> str:
    """
    Make a generic HTTP API call and return the response.

    This tool lets the agent call ANY REST API:
      - Your own backend APIs
      - Public APIs (OpenWeatherMap, GitHub, etc.)
      - Internal microservices

    Parameters use JSON strings (not dicts) because the model
    can only pass string arguments through the tool call format.

    EXAMPLE TOOL CALLS FROM THE AGENT:
      call_api(
        url="https://wttr.in/Arusha?format=j1",
        method="GET",
        headers_json="{}",
        body_json="{}"
      )

      call_api(
        url="https://api.example.com/data",
        method="POST",
        headers_json='{"Authorization": "Bearer mytoken", "Content-Type": "application/json"}',
        body_json='{"query": "fish prices"}'
      )
    """
    try:
        # Parse JSON strings back to dicts
        try:
            headers = json.loads(headers_json) if headers_json.strip() not in ("{}", "") else {}
        except json.JSONDecodeError:
            return f"Error: headers_json is not valid JSON: {headers_json}"

        try:
            body = json.loads(body_json) if body_json.strip() not in ("{}", "") else None
        except json.JSONDecodeError:
            return f"Error: body_json is not valid JSON: {body_json}"

        # Add a default user agent
        headers.setdefault("User-Agent", "LocalAIAgent/1.0")

        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=body,
            timeout=15
        )

        status = response.status_code
        content_type = response.headers.get("content-type", "")

        # Try to parse as JSON first
        if "application/json" in content_type:
            try:
                data = response.json()
                body_str = json.dumps(data, indent=2, ensure_ascii=False)
                if len(body_str) > 3000:
                    body_str = body_str[:3000] + "\n... [truncated]"
                return f"HTTP {status} from {url}:\n{body_str}"
            except Exception:
                pass

        # Fall back to text
        text = response.text[:3000]
        return f"HTTP {status} from {url}:\n{text}"

    except requests.exceptions.Timeout:
        return f"Timeout calling {url}"
    except requests.exceptions.ConnectionError:
        return f"Cannot reach {url}"
    except Exception as e:
        return f"API call error: {e}"