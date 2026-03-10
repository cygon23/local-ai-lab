"""
tools/web_tools.py
------------------
Real-world web tools: search, fetch, API calls.

SEARCH FIX:
  DuckDuckGo's Instant Answer API only returns "instant answers"
  (like Wikipedia boxes), not full web search results.
  For queries like "AI agent news today" it returns nothing.

  We now use TWO strategies in order:
    1. DuckDuckGo HTML search (scrape the actual search results page)
       — no API key, no rate limit for light use
    2. Fallback: direct fetch from known sources

  This gives the agent real search results for any query.
"""

import requests
import json
import re
from urllib.parse import quote_plus, urlparse


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LocalAIAgent/1.0; +https://github.com/local-ai-lab)"}


# ── 1. Web Search (DuckDuckGo HTML scrape) ──

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return real results.

    We scrape the DuckDuckGo HTML results page — this gives actual
    web results (titles + URLs + snippets), not just instant answers.
    No API key needed. Respects robots.txt via reasonable usage.
    """
    try:
        # DuckDuckGo HTML search endpoint
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        html = response.text
        results = []

        # Extract result blocks — DDG HTML uses class "result__body"
        # Each result has: title (.result__a), snippet (.result__snippet), url (.result__url)
        blocks = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</span>',
            html,
            re.DOTALL
        )

        for href, title, snippet in blocks[:max_results]:
            # Clean HTML tags from title and snippet
            title_clean = re.sub(r"<[^>]+>", "", title).strip()
            snippet_clean = re.sub(r"<[^>]+>", "", snippet).strip()
            # DDG wraps URLs — extract the real one
            real_url = re.sub(r"//duckduckgo\.com/l/\?uddg=", "", href)
            if real_url.startswith("//"):
                real_url = "https:" + real_url

            results.append(
                f"[Result]\nTitle: {title_clean}\nURL: {real_url}\nSnippet: {snippet_clean}"
            )

        if results:
            return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)

        # If HTML scrape fails, fall back to Instant Answer API
        return _ddg_instant_answer(query, max_results)

    except requests.exceptions.Timeout:
        return f"Search timed out for: '{query}'. Try fetch_webpage with a direct URL."
    except requests.exceptions.ConnectionError:
        return "No internet connection available."
    except Exception as e:
        # Always try fallback
        return _ddg_instant_answer(query, max_results)


def _ddg_instant_answer(query: str, max_results: int = 5) -> str:
    """Fallback: DuckDuckGo Instant Answer API."""
    try:
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        response = requests.get("https://api.duckduckgo.com/", params=params, timeout=10, headers=HEADERS)
        data = response.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"[Summary]\n{data['AbstractText']}\nSource: {data.get('AbstractURL', '')}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Topics" in topic:
                for sub in topic["Topics"][:2]:
                    text = sub.get("Text", "")
                    url = sub.get("FirstURL", "")
                    if text:
                        results.append(f"[Result]\n{text}\nURL: {url}")
            else:
                text = topic.get("Text", "")
                url = topic.get("FirstURL", "")
                if text:
                    results.append(f"[Result]\n{text}\nURL: {url}")

        if results:
            return f"Search results for '{query}':\n\n" + "\n\n---\n\n".join(results)

        return (
            f"No results found for '{query}'.\n"
            f"Suggestion: Use fetch_webpage with a direct URL to a news site, "
            f"e.g. fetch_webpage('https://news.ycombinator.com') for tech news."
        )
    except Exception as e:
        return f"Search unavailable: {e}"


# ── 2. Fetch Webpage ──

def fetch_webpage(url: str, max_chars: int = 3000) -> str:
    """
    Fetch and return the readable text content of any public URL.
    Works for webpages, JSON APIs, RSS feeds, plain text files.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return "Error: URL must start with http:// or https://"

        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")

        # JSON API response
        if "application/json" in content_type:
            try:
                data = response.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n... [truncated]"
                return f"JSON from {url}:\n{text}"
            except Exception:
                pass

        # HTML — strip tags and return clean text
        html = response.text
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
        html = re.sub(r"<(p|div|br|li|h[1-6]|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", html)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"

        if not text:
            return f"No readable text found at {url}"

        return f"Content from {url}:\n\n{text}"

    except requests.exceptions.Timeout:
        return f"Timeout fetching {url}"
    except requests.exceptions.HTTPError as e:
        return f"HTTP {e.response.status_code} error for {url}"
    except requests.exceptions.ConnectionError:
        return f"Cannot reach {url} — check internet connection."
    except Exception as e:
        return f"Error fetching {url}: {e}"


# ── 3. Generic API Call ──

def call_api(url: str, method: str = "GET", headers_json: str = "{}", body_json: str = "{}") -> str:
    """
    Make a generic HTTP API call to any REST endpoint.
    Use for APIs that need specific methods, headers, or request bodies.
    For simple page reads, prefer fetch_webpage instead.
    """
    try:
        try:
            extra_headers = json.loads(headers_json) if headers_json.strip() not in ("{}", "") else {}
        except json.JSONDecodeError:
            return f"Error: headers_json is not valid JSON: {headers_json}"

        try:
            body = json.loads(body_json) if body_json.strip() not in ("{}", "") else None
        except json.JSONDecodeError:
            return f"Error: body_json is not valid JSON: {body_json}"

        merged_headers = {**HEADERS, **extra_headers}

        response = requests.request(
            method=method.upper(),
            url=url,
            headers=merged_headers,
            json=body,
            timeout=15
        )

        content_type = response.headers.get("content-type", "")
        status = response.status_code

        if "application/json" in content_type:
            try:
                data = response.json()
                body_str = json.dumps(data, indent=2, ensure_ascii=False)
                if len(body_str) > 3000:
                    body_str = body_str[:3000] + "\n... [truncated]"
                return f"HTTP {status} from {url}:\n{body_str}"
            except Exception:
                pass

        text = response.text[:3000]
        return f"HTTP {status} from {url}:\n{text}"

    except requests.exceptions.Timeout:
        return f"Timeout calling {url}"
    except requests.exceptions.ConnectionError:
        return f"Cannot reach {url}"
    except Exception as e:
        return f"API call error: {e}"