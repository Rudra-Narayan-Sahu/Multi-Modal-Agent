import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import run_agent_workflow, stream_agent_workflow
from app.memory.conversation import conversation_manager


router = APIRouter(prefix="/api", tags=["Multi-Agent Chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., description="User question or research query")
    conversation_id: Optional[str] = Field("default_session", description="Unique conversation session ID")
    user_id: Optional[str] = Field("default_user", description="Identifier for authorized user")


class ChatResponse(BaseModel):
    query: str
    conversation_id: str
    user_id: str
    final_response: str
    query_category: Optional[str] = None
    selected_agents: List[str] = Field(default_factory=list)
    execution_plan: List[str] = Field(default_factory=list)
    source_references: List[Dict[str, Any]] = Field(default_factory=list)
    execution_status: str
    errors: List[str] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
def execute_chat(request: ChatRequest):
    """
    Executes the complete Multi-Agent LangGraph workflow.
    Supervisor analyzes the query, dynamically routes across RAG, Web, Tool,
    and Research agents, and synthesizes the grounded final response.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        final_state = run_agent_workflow(
            query=request.query.strip(),
            conversation_id=request.conversation_id or "default_session",
            user_id=request.user_id or "default_user"
        )

        final_response = final_state.get("final_response", "No response generated.")
        selected_agents = final_state.get("selected_agents", [])
        category = final_state.get("query_category", "UNKNOWN")
        citations = final_state.get("source_references", [])
        errors = final_state.get("errors", [])
        status = final_state.get("execution_status", "completed")

        # Save turn to conversation memory
        conversation_manager.add_turn(
            conversation_id=request.conversation_id or "default_session",
            user_id=request.user_id or "default_user",
            query=request.query,
            response=final_response,
            category=category,
            agents_used=selected_agents,
            citations=citations
        )

        return ChatResponse(
            query=request.query,
            conversation_id=request.conversation_id or "default_session",
            user_id=request.user_id or "default_user",
            final_response=final_response,
            query_category=category,
            selected_agents=selected_agents,
            execution_plan=final_state.get("execution_plan", []),
            source_references=citations,
            execution_status=status,
            errors=errors
        )
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-Agent Workflow Error: {str(e)}")


@router.post("/chat/stream")
def stream_chat(request: ChatRequest):
    """
    Streams multi-agent progress events via Server-Sent Events (SSE)
    as nodes complete execution, followed by the final answer.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def event_generator():
        try:
            for event in stream_agent_workflow(
                query=request.query.strip(),
                conversation_id=request.conversation_id or "default_session",
                user_id=request.user_id or "default_user"
            ):
                for node_name, node_output in event.items():
                    payload = {
                        "event": "node_complete",
                        "node": node_name,
                        "data": {
                            "status": node_output.get("execution_status"),
                            "agent_messages": node_output.get("agent_messages", []),
                            "errors": node_output.get("errors", [])
                        }
                    }
                    if "final_response" in node_output:
                        payload["final_response"] = node_output["final_response"]
                    yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as e:
            err_payload = {"event": "error", "error": str(e)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

