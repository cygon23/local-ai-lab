"""
tools/filesystem.py
-------------------
Tools for reading and writing files in the agent's workspace.

WORKSPACE CONCEPT:
  The agent has a sandboxed folder: data/workspace/
  It can read and write files there freely.
  It cannot access files outside this folder (path traversal is blocked).

  This is a fundamental safety principle for agents:
  Give them a sandbox, not the whole filesystem.

WHY THIS MATTERS FOR REAL AGENTS:
  An agent that can write files can:
  - Save research results for later
  - Create reports, summaries, code
  - Build up a "memory" of completed work
  - Pass outputs to other agents (multi-agent, V6)

  Without file tools, the agent's work disappears after each response.
  With file tools, it can produce persistent artifacts.
"""

from pathlib import Path

WORKSPACE = Path("data/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Max file size to read (prevents accidentally reading huge files)
MAX_READ_BYTES = 50_000  # 50KB


def _safe_path(filename: str) -> Path:
    """
    Resolve a filename to an absolute path inside the workspace.
    Blocks path traversal attacks like '../../etc/passwd'.

    ALWAYS sanitize paths when letting an AI (or user) specify filenames.
    """
    # Strip any directory components — only allow plain filenames
    safe_name = Path(filename).name
    return WORKSPACE / safe_name


def read_file(filename: str) -> str:
    """
    Read and return the contents of a file from the workspace.
    Returns an error string if the file doesn't exist or is too large.
    """
    path = _safe_path(filename)

    if not path.exists():
        files = list_workspace()
        return (
            f"Error: File '{filename}' not found in workspace.\n"
            f"Available files: {files}"
        )

    if path.stat().st_size > MAX_READ_BYTES:
        return (
            f"Error: File '{filename}' is too large to read directly "
            f"({path.stat().st_size} bytes). Consider reading specific sections."
        )

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(filename: str, content: str) -> str:
    """
    Write content to a file in the workspace.
    Creates the file if it doesn't exist, overwrites if it does.
    Returns a confirmation message.
    """
    path = _safe_path(filename)

    try:
        path.write_text(content, encoding="utf-8")
        size = path.stat().st_size
        return f"✅ File '{filename}' written successfully ({size} bytes) at {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_workspace() -> str:
    """
    List all files in the workspace folder.
    Returns a formatted string listing files and their sizes.
    """
    files = sorted(WORKSPACE.glob("*"))

    if not files:
        return "Workspace is empty — no files yet."

    lines = [f"Files in workspace ({len(files)} total):"]
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            size_str = f"{size}B" if size < 1024 else f"{size//1024}KB"
            lines.append(f"  - {f.name} ({size_str})")

    return "\n".join(lines)