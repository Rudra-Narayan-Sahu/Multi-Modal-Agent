import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


def merge_documents(existing: Optional[List[Dict[str, Any]]], new: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merges document chunks while deduplicating by text or chunk id."""
    existing_list = existing or []
    new_list = new or []
    if not new_list:
        return existing_list
    seen = {doc.get("text", "")[:100] for doc in existing_list}
    merged = list(existing_list)
    for doc in new_list:
        key = doc.get("text", "")[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged


def merge_web_results(existing: Optional[List[Dict[str, Any]]], new: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merges web search items while deduplicating by URL."""
    existing_list = existing or []
    new_list = new or []
    if not new_list:
        return existing_list
    seen = {item.get("url", "") for item in existing_list}
    merged = list(existing_list)
    for item in new_list:
        url = item.get("url", "")
        if url and url not in seen:
            seen.add(url)
            merged.append(item)
    return merged


class AgentState(TypedDict, total=False):
    """
    Shared typed state for the Multi-Agent RAG Assistant LangGraph workflow.
    All agents read from and write to this state.
    """
    user_query: str
    conversation_id: str
    user_id: Optional[str]
    conversation_context: Optional[str]          # Recent dialogue history for follow-up resolution
    relevant_memories: List[Dict[str, Any]]      # Injected persistent user preferences & facts
    query_category: str                          # DOCUMENT_QUERY, WEB_SEARCH_QUERY, COMPARISON_QUERY, etc.
    execution_plan: List[str]                    # Ordered list of planned agent steps
    selected_agents: List[str]                   # Names of agents activated for this query
    retrieved_documents: Annotated[List[Dict[str, Any]], merge_documents]
    web_search_results: Annotated[List[Dict[str, Any]], merge_web_results]
    research_findings: Optional[str]             # Comparative synthesis and key insights
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    agent_messages: Annotated[List[Dict[str, Any]], operator.add]  # Agent audit & diagnostic trail
    final_context: str                           # Unified context assembled for Response Agent
    source_references: Annotated[List[Dict[str, Any]], operator.add] # Structured citations
    execution_status: str                        # "planning", "executing", "completed", "fallback"
    errors: Annotated[List[str], operator.add]   # Accumulated errors/warnings across nodes
    final_response: str                          # User-facing answer formatted in Markdown
    iteration_count: int                         # Cycle guard to prevent infinite loops (max 5)
    next_step: Optional[str]                     # Routing destination: "rag", "web", "tool", "research", "response"

