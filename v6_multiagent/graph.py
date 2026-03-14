"""
graph.py
--------
The V6 multi-agent LangGraph graph.

ARCHITECTURE:
                    ┌─────────────────┐
  User goal ──────► │   Orchestrator  │
                    └────────┬────────┘
                             │ decides route
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        researcher       executor         done
              │              │              │
              └──────────────┘              │
                     │                     │
                     ▼                     ▼
              ┌─────────────────┐    Final answer
              │   Orchestrator  │
              │  (synthesizes)  │
              └─────────────────┘
"""

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from tools import RESEARCH_TOOLS, EXECUTOR_TOOLS
from memory import build_memory_prompt


# ── Agent prompts (inlined to avoid import path issues) ───────────────

ORCHESTRATOR_SYSTEM = """You are the Orchestrator in a multi-agent AI system.

Your job:
1. Analyze the user's goal carefully
2. Decide which specialist agent(s) to involve
3. After specialists report back, synthesize a clear final answer

Available agents:
- Researcher: web search, reading webpages, searching INDEXED DOCUMENTS (PDFs, uploaded files)
- Executor: math calculations, Python code execution, file read/write

Routing rules:
- Questions about uploaded documents, PDFs, or indexed content → ROUTE: researcher
  (The Researcher uses search_knowledge_base tool to search indexed documents)
- If the goal needs current web information or facts → ROUTE: researcher
- If the goal needs computation, code, or file operations → ROUTE: executor
- If it needs both (e.g. search then save results) → ROUTE: both
- If you can answer directly from knowledge → ROUTE: done

IMPORTANT document routing examples:
  "summarize nevland.pdf" → ROUTE: researcher (search knowledge base)
  "what is in this document" → ROUTE: researcher (search knowledge base)
  "what is the theme of the uploaded doc" → ROUTE: researcher (search knowledge base)

When routing, state clearly:
  ROUTE: researcher
  ROUTE: executor
  ROUTE: both
  ROUTE: done

IMPORTANT: When you choose ROUTE: done, you MUST immediately answer the user in the same response.
Do not just explain why you chose done — actually answer the question or respond to the message.

Example for "hi":
  ROUTE: done
  Hi there! How can I help you today?

When writing the final answer after agents have reported, be concise and clear."""


RESEARCHER_SYSTEM = """You are the Researcher agent in a multi-agent AI system.

Your role: gather information accurately and efficiently.

You have EXACTLY 3 tools. Use ONLY these — calling anything else will cause an error:
  1. search_knowledge_base(query) — search indexed/uploaded documents (PDFs, files)
  2. web_search(query, max_results) — search the web for current information
  3. fetch_webpage(url) — read a specific URL directly

NEVER call: brave_search, google_search, bing_search, or any other tool.

Decision strategy — pick the right tool first:
  - User asks about an uploaded file, PDF, or document → use search_knowledge_base FIRST
  - User asks about web news, current events → use web_search or fetch_webpage
  - For LangGraph news → fetch_webpage('https://github.com/langchain-ai/langgraph/releases')
  - For LangChain docs → fetch_webpage('https://python.langchain.com/docs/')
  - If web_search returns no results → try fetch_webpage with a direct URL

Examples:
  "summarize nevland.pdf" → search_knowledge_base('nevland')
  "what is the theme of the document" → search_knowledge_base('theme main topic')
  "latest LangGraph news" → fetch_webpage('https://github.com/langchain-ai/langgraph/releases')

Instructions:
- For document questions, ALWAYS try search_knowledge_base before web search
- Return findings clearly and concisely
- Do not make up facts"""


EXECUTOR_SYSTEM = """You are the Executor agent in a multi-agent AI system.

Your role: perform computations, run code, and manage files accurately.

Tools available to you:
- calculate: evaluate mathematical expressions exactly
- run_python: write and execute Python code
- write_file: save content to files in the workspace
- read_file: read existing files from the workspace
- list_files: see what files exist in the workspace

Instructions:
- Always use calculate for math — never guess numbers
- When writing code, always print() results so they appear in output
- Return your results clearly so the Orchestrator can synthesize them"""


def parse_route(response: str) -> str:
    r = response.lower()
    if "route: both" in r:
        return "both"
    if "route: researcher" in r:
        return "researcher"
    if "route: executor" in r:
        return "executor"
    if "route: done" in r:
        return "done"
    if any(w in r for w in ["search", "fetch", "look up", "find information", "research"]):
        return "researcher"
    if any(w in r for w in ["calculate", "compute", "run", "execute", "write file", "code"]):
        return "executor"
    return "done"


# ── Shared State ──────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_goal: str
    orchestrator_plan: str
    research_results: str
    execution_results: str
    final_answer: str
    route: str
    messages: list
    steps: list


# ── Node functions ────────────────────────────────────────────────────

def orchestrator_node(state: AgentState, llm) -> AgentState:
    memory_block = build_memory_prompt()
    system = ORCHESTRATOR_SYSTEM
    if memory_block:
        system = f"{system}\n\n{memory_block}"

    messages = [SystemMessage(content=system)]

    if state.get("research_results") or state.get("execution_results"):
        synthesis_prompt = f"""Original goal: {state['user_goal']}

Research findings:
{state.get('research_results', 'No research was done.')}

Execution results:
{state.get('execution_results', 'No execution was done.')}

Now write the final answer to the user's goal based on these findings.
Be clear, concise, and helpful. Do not use ROUTE tags."""
        messages.append(HumanMessage(content=synthesis_prompt))
    else:
        messages.append(HumanMessage(content=(
            f"User goal: {state['user_goal']}\n\n"
            f"Analyze this goal and decide which agent(s) to involve. "
            f"State your routing decision clearly using ROUTE: tag."
        )))

    response = llm.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    if state.get("research_results") or state.get("execution_results"):
        return {
            **state,
            "final_answer": content,
            "steps": state.get("steps", []) + [{
                "agent": "orchestrator",
                "type": "answer",
                "content": content,
            }]
        }
    else:
        route = parse_route(content)
        return {
            **state,
            "orchestrator_plan": content,
            "route": route,
            "steps": state.get("steps", []) + [{
                "agent": "orchestrator",
                "type": "routing",
                "content": f"Route: {route}\n\n{content}",
            }]
        }


def researcher_node(state: AgentState, llm) -> AgentState:
    agent = create_react_agent(model=llm, tools=RESEARCH_TOOLS)

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=(
            f"Research task from Orchestrator:\n{state['orchestrator_plan']}\n\n"
            f"Original user goal: {state['user_goal']}\n\n"
            f"Use your tools to find the information needed."
        ))
    ]

    result = agent.invoke({"messages": messages})
    result_messages = result.get("messages", [])

    steps = list(state.get("steps", []))
    for msg in result_messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "agent": "researcher",
                    "type": "tool_call",
                    "content": str(tc.get("args", {})),
                    "tool_name": tc.get("name", ""),
                })
        elif msg_type == "ToolMessage":
            steps.append({
                "agent": "researcher",
                "type": "observation",
                "content": str(msg.content)[:500],
                "tool_name": getattr(msg, "name", None),
            })

    research_output = ""
    for msg in reversed(result_messages):
        if type(msg).__name__ == "AIMessage":
            if not getattr(msg, "tool_calls", []) and msg.content:
                research_output = msg.content
                break

    steps.append({"agent": "researcher", "type": "answer", "content": research_output})

    return {**state, "research_results": research_output, "steps": steps}


def executor_node(state: AgentState, llm) -> AgentState:
    llm_with_tools = llm.bind_tools(EXECUTOR_TOOLS)
    agent = create_react_agent(model=llm_with_tools, tools=EXECUTOR_TOOLS)

    context = (
        f"Execution task from Orchestrator:\n{state['orchestrator_plan']}\n\n"
        f"Original user goal: {state['user_goal']}"
    )
    if state.get("research_results"):
        context += f"\n\nResearch findings to work with:\n{state['research_results']}"

    messages = [
        SystemMessage(content=EXECUTOR_SYSTEM),
        HumanMessage(content=context)
    ]

    result = agent.invoke({"messages": messages})
    result_messages = result.get("messages", [])

    steps = list(state.get("steps", []))
    for msg in result_messages:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "agent": "executor",
                    "type": "tool_call",
                    "content": str(tc.get("args", {})),
                    "tool_name": tc.get("name", ""),
                })
        elif msg_type == "ToolMessage":
            steps.append({
                "agent": "executor",
                "type": "observation",
                "content": str(msg.content)[:500],
                "tool_name": getattr(msg, "name", None),
            })

    execution_output = ""
    for msg in reversed(result_messages):
        if type(msg).__name__ == "AIMessage":
            if not getattr(msg, "tool_calls", []) and msg.content:
                execution_output = msg.content
                break

    steps.append({"agent": "executor", "type": "answer", "content": execution_output})

    return {**state, "execution_results": execution_output, "steps": steps}


# ── Routing functions ─────────────────────────────────────────────────

def route_after_orchestrator(state: AgentState) -> Literal["researcher", "executor", "both_research_first", "end"]:
    route = state.get("route", "done")
    if route == "researcher":
        return "researcher"
    if route == "executor":
        return "executor"
    if route == "both":
        return "both_research_first"
    return "end"


def route_after_researcher(state: AgentState) -> Literal["executor", "orchestrator"]:
    if state.get("route") == "both":
        return "executor"
    return "orchestrator"


# ── Build the graph ───────────────────────────────────────────────────

def build_graph(llm):
    def _orchestrator(state):
        return orchestrator_node(state, llm)

    def _researcher(state):
        return researcher_node(state, llm)

    def _executor(state):
        return executor_node(state, llm)

    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", _orchestrator)
    builder.add_node("researcher", _researcher)
    builder.add_node("executor", _executor)

    builder.add_edge(START, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "researcher": "researcher",
            "executor": "executor",
            "both_research_first": "researcher",
            "end": END,
        }
    )

    builder.add_conditional_edges(
        "researcher",
        route_after_researcher,
        {
            "executor": "executor",
            "orchestrator": "orchestrator",
        }
    )

    builder.add_edge("executor", "orchestrator")

    return builder.compile()


def run_graph(user_goal: str, llm) -> dict:
    graph = build_graph(llm)

    initial_state: AgentState = {
        "user_goal": user_goal,
        "orchestrator_plan": "",
        "research_results": "",
        "execution_results": "",
        "final_answer": "",
        "route": "",
        "messages": [],
        "steps": [],
    }

    return graph.invoke(initial_state)


def stream_graph(user_goal: str, llm):
    """
    Stream graph execution step by step.
    Yields dicts: {"type": "step"|"done", "step": {...}|None, "state": {...}|None}
    This allows the UI to render each step as it happens instead of waiting for full completion.
    """
    graph = build_graph(llm)

    initial_state: AgentState = {
        "user_goal": user_goal,
        "orchestrator_plan": "",
        "research_results": "",
        "execution_results": "",
        "final_answer": "",
        "route": "",
        "messages": [],
        "steps": [],
    }

    last_steps_count = 0

    for state_update in graph.stream(initial_state, stream_mode="values"):
        current_steps = state_update.get("steps", [])

        # Yield any new steps since last update
        if len(current_steps) > last_steps_count:
            for step in current_steps[last_steps_count:]:
                yield {"type": "step", "step": step, "state": state_update}
            last_steps_count = len(current_steps)

        # If final answer is ready, yield done
        if state_update.get("final_answer"):
            yield {"type": "done", "step": None, "state": state_update}
            return