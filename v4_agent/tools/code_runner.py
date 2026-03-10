"""
tools/code_runner.py
--------------------
Execute Python code in a subprocess and return its output.

WHY A SUBPROCESS AND NOT exec()?
  exec() runs code IN the same Python process as our agent.
  If the code does `import os; os.system("rm -rf /")` — that's a problem.
  A subprocess is isolated:
    - Has its own memory space
    - Can be killed if it runs too long (timeout)
    - Its crashes don't crash our agent

  This is the same principle Docker uses — isolation via process boundaries.
  Real production agents (like Claude's computer use) use full containers.
  For local learning, subprocess is the right level of isolation.

WHAT THE AGENT CAN DO WITH THIS:
  - Analyze data (CSV parsing, statistics)
  - Generate computed results (fibonacci, prime numbers)
  - String manipulation, text processing
  - Anything pure Python can do

WHAT WE BLOCK:
  We don't explicitly block imports — we rely on the timeout and
  the fact that the subprocess has no special permissions.
  For production agents, you'd add import whitelisting.
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

TIMEOUT_SECONDS = 10  # Kill the process if it runs longer than this


def run_python(code: str) -> str:
    """
    Execute Python code in an isolated subprocess.
    Returns stdout output, or stderr if execution fails.

    The code has access to:
      - Python standard library
      - Any packages installed in the current environment
      - Current working directory (read-only in practice)

    Timeout: 10 seconds — prevents infinite loops from hanging the agent.
    """
    # Write code to a temp file (cleaner than passing via -c flag)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(Path("data/workspace"))  # run from workspace dir
        )

        # Combine stdout and stderr for the model to see
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"

        if not output.strip():
            return "Code executed successfully (no output produced)."

        # Truncate very long outputs so we don't overflow the context
        if len(output) > 3000:
            output = output[:3000] + "\n... [output truncated at 3000 chars]"

        return output.strip()

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {TIMEOUT_SECONDS} seconds. Check for infinite loops."
    except Exception as e:
        return f"Error running code: {e}"
    finally:
        # Always clean up the temp file
        try:
            os.unlink(temp_path)
        except Exception:
            pass