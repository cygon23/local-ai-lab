"""
agent.py
--------
V5 LangChain agent using the modern API (LangChain 1.x / LangGraph).

WHAT CHANGED:
  LangChain 1.x removed AgentExecutor and create_react_agent.
  The new way is langgraph.prebuilt.create_react_agent — which is
  actually cleaner and more powerful.

  Same ReAct loop. Same tools. Different entry point.

  V4: hand-written while loop
  V5: langgraph.prebuilt.create_react_agent (the loop LangChain now recommends)
"""

from dataclasses import dataclass
from typing import Generator, Optional

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from ollama_llm import get_llm
from memory import build_memory_prompt, store_memory, recall_memory
from tools import ALL_TOOLS
from rag import search_knowledge_base
from langchain_core.tools import tool


@tool
def store_memory_tool(key: str, value: str, category: str = "general") -> str:
    """
    Store an important fact in long-term memory to remember across sessions.
    Use when the user shares something important about themselves, their projects,
    or their preferences. Category options: user_profile, projects, preferences, general.
    """
    return store_memory(key, value, category)


@tool
def recall_memory_tool(query: str) -> str:
    """
    Search long-term memory for previously stored facts.
    Use when the user references something from a past session,
    or when context from memory would help answer the question.
    """
    return recall_memory(query)


AGENT_TOOLS = ALL_TOOLS + [search_knowledge_base, store_memory_tool, recall_memory_tool]


@dataclass
class AgentStep:
    type: str          # tool_call | observation | answer | error
    content: str
    tool_name: Optional[str] = None


def build_system_prompt(base_prompt: str) -> str:
    memory_block = build_memory_prompt()
    parts = [base_prompt]
    if memory_block:
        parts.append(f"\n\n{memory_block}")
    return "\n".join(parts)


def run_agent(
    user_message: str,
    model: str = "qwen3:1.7b",
    base_system_prompt: str = "",
    memory=None,
) -> Generator[AgentStep, None, None]:
    """
    Runs the LangGraph ReAct agent and yields AgentStep objects.

    create_react_agent from langgraph.prebuilt is the modern replacement
    for LangChain's AgentExecutor. It implements the same Think → Act →
    Observe loop, now backed by a LangGraph state machine.
    """
    try:
        llm = get_llm(model=model)

        system_content = build_system_prompt(
            base_system_prompt or (
                "You are a capable AI agent. Think step by step. "
                "Use tools when needed. Be concise in your final answers. "
                "When you learn something important about the user or their projects, "
                "use store_memory_tool to remember it."
            )
        )

        # Build message history from LangChain memory if available
        messages = [SystemMessage(content=system_content)]

        if memory:
            # Add chat history from ConversationBufferMemory
            try:
                history = memory.chat_memory.messages
                messages.extend(history)
            except Exception:
                pass

        messages.append(HumanMessage(content=user_message))

        # create_react_agent is the new recommended API in LangChain 1.x
        agent = create_react_agent(
            model=llm,
            tools=AGENT_TOOLS,
        )

        # Run the agent
        result = agent.invoke({"messages": messages})

        # Parse the result messages to extract tool calls and observations
        result_messages = result.get("messages", [])

        for msg in result_messages:
            msg_type = type(msg).__name__

            # Tool call messages
            if msg_type == "AIMessage" and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    yield AgentStep(
                        type="tool_call",
                        content=str(tc.get("args", {})),
                        tool_name=tc.get("name", "unknown"),
                    )

            # Tool result messages
            elif msg_type == "ToolMessage":
                yield AgentStep(
                    type="observation",
                    content=str(msg.content),
                    tool_name=getattr(msg, "name", None),
                )

        # Final answer — last AIMessage with no tool calls
        final = ""
        for msg in reversed(result_messages):
            if type(msg).__name__ == "AIMessage":
                tool_calls = getattr(msg, "tool_calls", [])
                if not tool_calls and msg.content:
                    final = msg.content
                    break

        yield AgentStep(type="answer", content=final or "Agent completed.")

    except Exception as e:
        yield AgentStep(type="error", content=f"Agent error: {e}")