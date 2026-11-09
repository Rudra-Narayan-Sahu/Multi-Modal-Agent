import logfire
from typing import Dict, Any, List, Union, AsyncGenerator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_agent
from app.agents.rag_agent import rag_node
from app.agents.web_agent import web_node
from app.agents.tool_agent import tool_node
from app.agents.research_agent import research_node
from app.agents.response_agent import response_node


def route_supervisor_initial(state: AgentState) -> Union[str, List[Send]]:
    """Conditional router for the initial Supervisor planning output."""
    next_step = state.get("next_step")
    
    if next_step == "parallel_rag_web":
        # Dynamic parallel dispatch to both RAG and Web agents
        return [
            Send("rag_agent", state),
            Send("web_agent", state)
        ]
    elif next_step == "rag_agent":
        return "rag_agent"
    elif next_step == "web_agent":
        return "web_agent"
    elif next_step == "tool_agent":
        return "tool_agent"
    elif next_step == "research_agent":
        return "research_agent"
    else:
        return "response_agent"


def route_supervisor_eval(state: AgentState) -> str:
    """Conditional router after intermediate agent results have been evaluated."""
    next_step = state.get("next_step", "response_agent")
    if next_step in {"research_agent", "rag_agent", "web_agent", "tool_agent"}:
        return next_step
    return "response_agent"


def build_multi_agent_graph():
    """Constructs and compiles the complete multi-agent LangGraph workflow."""
    builder = StateGraph(AgentState)

    # Register all 6 specialized agent nodes (+ supervisor evaluation)
    builder.add_node("supervisor", supervisor_agent.plan_and_route)
    builder.add_node("supervisor_eval", supervisor_agent.evaluate_progress)
    builder.add_node("rag_agent", rag_node)
    builder.add_node("web_agent", web_node)
    builder.add_node("tool_agent", tool_node)
    builder.add_node("research_agent", research_node)
    builder.add_node("response_agent", response_node)

    # Define edges
    # START -> Supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor -> Dynamic Route
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_initial,
        ["rag_agent", "web_agent", "tool_agent", "research_agent", "response_agent"]
    )

    # Intermediate workers -> Supervisor Evaluation
    builder.add_edge("rag_agent", "supervisor_eval")
    builder.add_edge("web_agent", "supervisor_eval")
    builder.add_edge("tool_agent", "supervisor_eval")

    # Supervisor Evaluation -> Research or Response
    builder.add_conditional_edges(
        "supervisor_eval",
        route_supervisor_eval,
        ["research_agent", "rag_agent", "web_agent", "tool_agent", "response_agent"]
    )

    # Research -> Response
    builder.add_edge("research_agent", "response_agent")

    # Response -> END
    builder.add_edge("response_agent", END)

    # Checkpointing for conversation-level persistence
    checkpointer = MemorySaver()
    compiled_app = builder.compile(checkpointer=checkpointer)
    return compiled_app


# Singleton compiled multi-agent application
multi_agent_app = build_multi_agent_graph()


def run_agent_workflow(
    query: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """Synchronous invocation of the multi-agent graph with thread checkpointing & memory."""
    from app.memory.conversation import conversation_manager
    from app.memory.retriever import memory_retriever

    # Load short-term history and relevant long-term memories
    history_context = conversation_manager.format_history_context(conversation_id, user_id=user_id)
    memories = memory_retriever.retrieve_relevant_memories(user_id=user_id, query=query, limit=5)

    initial_state: AgentState = {
        "user_query": query,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "conversation_context": history_context,
        "relevant_memories": memories,
        "execution_plan": [],
        "selected_agents": [],
        "retrieved_documents": [],
        "web_search_results": [],
        "tool_results": [],
        "agent_messages": [],
        "final_context": "",
        "source_references": [],
        "execution_status": "started",
        "errors": [],
        "iteration_count": 0
    }

    config = {"configurable": {"thread_id": conversation_id}}

    with logfire.span("Multi-Agent Workflow Execution", query=query, conversation_id=conversation_id):
        final_state = multi_agent_app.invoke(initial_state, config=config)
        return final_state


def stream_agent_workflow(
    query: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
):
    """Streaming generator yielding updates as nodes complete execution."""
    from app.memory.conversation import conversation_manager
    from app.memory.retriever import memory_retriever

    history_context = conversation_manager.format_history_context(conversation_id, user_id=user_id)
    memories = memory_retriever.retrieve_relevant_memories(user_id=user_id, query=query, limit=5)

    initial_state: AgentState = {
        "user_query": query,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "conversation_context": history_context,
        "relevant_memories": memories,
        "execution_plan": [],
        "selected_agents": [],
        "retrieved_documents": [],
        "web_search_results": [],
        "tool_results": [],
        "agent_messages": [],
        "final_context": "",
        "source_references": [],
        "execution_status": "started",
        "errors": [],
        "iteration_count": 0
    }

    config = {"configurable": {"thread_id": conversation_id}}
    for event in multi_agent_app.stream(initial_state, config=config):
        yield event

