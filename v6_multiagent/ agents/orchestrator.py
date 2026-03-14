"""
agents/orchestrator.py
----------------------
The Orchestrator agent.

ROLE:
  - Receives the user's goal
  - Analyzes it and decides which agent(s) to involve
  - Routes to Researcher, Executor, or both
  - Collects their outputs and writes the final answer

ROUTING LOGIC:
  The orchestrator outputs a routing decision as part of its response.
  LangGraph reads this decision and follows the correct edge in the graph.

  Routes:
    "researcher"  → needs web search / document retrieval
    "executor"    → needs code execution / file ops / math
    "both"        → needs research then execution
    "done"        → can answer directly, no tools needed
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


ORCHESTRATOR_SYSTEM = """You are the Orchestrator in a multi-agent AI system.

Your job:
1. Analyze the user's goal carefully
2. Decide which specialist agent(s) to involve
3. After specialists report back, synthesize a clear final answer

Available agents:
- Researcher: web search, reading webpages, searching indexed documents
- Executor: math calculations, Python code execution, file read/write

Routing rules:
- If the goal needs current information, facts, or document search → route to Researcher
- If the goal needs computation, code, or file operations → route to Executor  
- If it needs both (e.g. search then save results) → route to both, Researcher first
- If you can answer directly from knowledge → answer immediately

When routing, state clearly:
  ROUTE: researcher
  ROUTE: executor  
  ROUTE: both
  ROUTE: done

When writing the final answer after agents have reported, be concise and clear.
Do not repeat what the agents said verbatim — synthesize it."""


def get_orchestrator_prompt() -> str:
    return ORCHESTRATOR_SYSTEM


def parse_route(response: str) -> str:
    """
    Extract routing decision from orchestrator response.
    Returns: 'researcher' | 'executor' | 'both' | 'done'
    """
    response_lower = response.lower()

    if "route: both" in response_lower:
        return "both"
    if "route: researcher" in response_lower:
        return "researcher"
    if "route: executor" in response_lower:
        return "executor"
    if "route: done" in response_lower:
        return "done"

    # Infer from content if no explicit route tag
    if any(w in response_lower for w in ["search", "fetch", "look up", "find information", "research"]):
        return "researcher"
    if any(w in response_lower for w in ["calculate", "compute", "run", "execute", "write file", "code"]):
        return "executor"

    return "done"