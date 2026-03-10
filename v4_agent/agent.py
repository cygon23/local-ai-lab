"""
agent.py
--------
The ReAct agent loop — built from scratch, no frameworks.

READ THIS FILE CAREFULLY. Every line matters.

THIS IS WHAT EVERY AGENT FRAMEWORK HIDES FROM YOU.
LangChain's AgentExecutor, AutoGen's ConversableAgent, CrewAI's Agent —
they all do exactly what this file does, wrapped in abstractions.

Once you understand this loop, you can:
  - Debug any agent framework
  - Build agents frameworks don't support
  - Know exactly why your agent is making a decision
  - Fix failures that black-box frameworks hide

THE LOOP IN PLAIN ENGLISH:
  1. Build a prompt: system instructions + tool schemas + conversation history
  2. Ask the model: "What should I do next?"
  3. If the model outputs <tool_call>...</tool_call>:
       a. Parse the JSON inside the tags
       b. Call the real Python function
       c. Add the result to context as an "observation"
       d. Go back to step 2
  4. If the model outputs normal text (no tool call):
       That's the final answer. Return it.
  5. If we've looped MAX_ITERATIONS times, stop (prevents infinite loops).

CONTEXT BUILDING — HOW THE MODEL "REMEMBERS" TOOL RESULTS:
  Each iteration the model sees:
    [system: tools + instructions]
    [user: original question]
    [assistant: "I'll use the calculate tool"]
    [tool: <tool_call>{"tool": "calculate", ...}</tool_call>]
    [user: "Tool result: 320"]        ← observation injected as user message
    [assistant: decides next step]

  The model sees its own previous tool calls and their results.
  This is how it chains multiple tool calls intelligently.
"""

import json
import re
import requests
from typing import Generator

import time
from tools import get_tools_prompt, call_tool, TOOL_SCHEMAS
from observability import AgentTrace
from memory import build_memory_prompt

OLLAMA_BASE_URL = "http://localhost:11434"
MAX_ITERATIONS = 8       # Maximum tool calls per agent run
TOOL_CALL_TAG = "<tool_call>"
TOOL_END_TAG = "</tool_call>"


# ── Data structures ──

class AgentStep:
    """
    Represents one step in the agent's reasoning process.
    Stored and displayed in the UI to show the agent's "thinking".

    Types:
      "thinking"    → model's reasoning before a tool call
      "tool_call"   → the tool being called + arguments
      "observation" → result returned by the tool
      "answer"      → final response to the user
      "error"       → something went wrong
    """
    def __init__(self, step_type: str, content: str, tool_name: str = None):
        self.type = step_type
        self.content = content
        self.tool_name = tool_name

    def __repr__(self):
        return f"AgentStep({self.type}: {self.content[:60]}...)"


# ── Tool call parsing ──

def extract_tool_call(text: str) -> dict | None:
    """
    Parse a tool call from the model's output.

    We look for the pattern:
      <tool_call>
      {"tool": "...", "args": {...}}
      </tool_call>

    Returns the parsed dict, or None if no tool call found.

    WHY REGEX + JSON AND NOT JUST JSON?
      The model sometimes adds extra text before/after the JSON.
      We use regex to find the tags, then parse what's inside.
      This is more robust than expecting perfect JSON output.
    """
    # Find content between tool call tags
    pattern = rf"{re.escape(TOOL_CALL_TAG)}\s*(.*?)\s*{re.escape(TOOL_END_TAG)}"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    raw_json = match.group(1).strip()

    try:
        parsed = json.loads(raw_json)
        # Validate expected structure
        if "tool" in parsed and "args" in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        # Model produced malformed JSON — try to extract just the tool name
        tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', raw_json)
        if tool_match:
            return {"tool": tool_match.group(1), "args": {}}
        return None


def has_tool_call(text: str) -> bool:
    """Check if text contains a tool call."""
    return TOOL_CALL_TAG in text


# ── Context building ──

def build_system_prompt(base_system_prompt: str) -> str:
    """
    Build the full system prompt: base + long-term memory + tools.
    ORDER: role instructions → memory facts → tool schemas
    """
    memory_block = build_memory_prompt()
    return base_system_prompt + chr(10) + chr(10) + memory_block + chr(10) + get_tools_prompt()


def build_messages(
    conversation_history: list[dict],
    agent_scratchpad: list[dict]
) -> list[dict]:
    """
    Assemble the full message list to send to Ollama.

    Structure:
      conversation_history: the ongoing chat (user ↔ assistant turns)
      agent_scratchpad: this iteration's tool calls and observations

    The scratchpad is APPENDED to the history each iteration.
    It shows the model its own previous tool calls and results.

    WHY SEPARATE HISTORY AND SCRATCHPAD?
      History is persistent — it spans multiple user messages.
      Scratchpad is ephemeral — it's built fresh for each user message.
      Mixing them would confuse the model about what's "old" vs "current task".
    """
    return conversation_history + agent_scratchpad


# ── The core loop ──

def run_agent(
    user_message: str,
    conversation_history: list[dict],
    model: str,
    base_system_prompt: str,
    session_id: str = "default",
) -> Generator[AgentStep, None, None]:
    """
    Run the ReAct agent loop for a single user message.

    This is a GENERATOR — it yields AgentStep objects one by one.
    The UI consumes these steps to show the agent's thinking in real time.

    Why a generator?
      The agent loop can take many iterations.
      Yielding each step immediately keeps the UI responsive.
      The user sees: "Thinking... → calling tool X → got result → answer"
      Without a generator they'd see nothing until the final answer.

    FLOW:
      iteration 0: build context, ask model
      model says: <tool_call>{"tool": "calculate", "args": {"expression": "15*8"}}
      → yield AgentStep("tool_call", ...)
      → call calculate(expression="15*8") → "120"
      → yield AgentStep("observation", "120")
      → add to scratchpad, loop back

      iteration 1: model sees the result
      model says: "The answer is 120."
      → yield AgentStep("answer", "The answer is 120.")
      → done
    """
    system_prompt = build_system_prompt(base_system_prompt)
    scratchpad = []

    # Start a trace for this agent run
    trace = AgentTrace(session_id=session_id, user_message=user_message, model=model)

    # Add user message to scratchpad
    scratchpad.append({"role": "user", "content": user_message})

    for iteration in range(MAX_ITERATIONS):

        # Build the full message list for this iteration
        messages = build_messages(conversation_history, scratchpad)

        # Ask the model what to do next
        response_text = _call_model_sync(messages, model, system_prompt)

        if not response_text:
            yield AgentStep("error", "Model returned empty response.")
            return

        # Check if the model wants to call a tool
        if has_tool_call(response_text):
            tool_call = extract_tool_call(response_text)

            if tool_call is None:
                yield AgentStep("error", f"Model produced malformed tool call: {response_text}")
                # Add the malformed response to scratchpad so model can recover
                scratchpad.append({"role": "assistant", "content": response_text})
                scratchpad.append({
                    "role": "user",
                    "content": "Tool call format was invalid. Please try again with valid JSON."
                })
                continue

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            # Yield the "thinking" — any text before the tool call tag
            thinking = response_text[:response_text.find(TOOL_CALL_TAG)].strip()
            if thinking:
                yield AgentStep("thinking", thinking)

            # Yield the tool call itself (for UI display)
            yield AgentStep(
                "tool_call",
                json.dumps(tool_call, indent=2),
                tool_name=tool_name
            )

            # Execute the tool — record timing for trace
            t_tool_start = time.time()
            observation = call_tool(tool_name, tool_args)
            t_tool_ms = int((time.time() - t_tool_start) * 1000)

            # Record tool span in trace
            trace.add_tool_span(tool_name, tool_args, observation, t_tool_ms)

            # Yield the observation (tool result)
            yield AgentStep("observation", observation, tool_name=tool_name)

            # Add this tool call + result to the scratchpad
            scratchpad.append({
                "role": "assistant",
                "content": response_text
            })
            scratchpad.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{observation}"
            })

        else:
            # No tool call → this is the final answer
            final = response_text.strip()
            trace.finish(final_answer=final)
            trace.save()     # write trace to disk
            yield AgentStep("answer", final)
            return

    # Max iterations reached
    final = f"[Reached maximum {MAX_ITERATIONS} iterations] Completed as much as possible."
    trace.finish(final_answer=final)
    trace.save()
    yield AgentStep("answer", final)


def _call_model_sync(
    messages: list[dict],
    model: str,
    system_prompt: str
) -> str:
    """
    Call Ollama synchronously (non-streaming) and return the full response.

    WHY SYNC AND NOT STREAMING HERE?
      The agent loop needs the COMPLETE response to check for tool calls.
      We can't parse a tool call from half a response.
      We stream to the USER in app.py (the final answer).
      We use sync calls INTERNALLY within the loop.

      This is a common pattern: sync internally, stream externally.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": full_messages,
                "stream": False,
                "options": {
                    "temperature": 0.3,   # Lower temp for agents — more deterministic
                    "num_ctx": 4096,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama."
    except Exception as e:
        return f"Error: {e}"