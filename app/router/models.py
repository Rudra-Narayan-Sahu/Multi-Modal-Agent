from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Categorized user intents."""
    DOCUMENT_RAG = "document_rag"
    SUMMARIZATION = "summarization"
    GENERAL_QA = "general_qa"
    INFORMATION_EXTRACTION = "information_extraction"
    DATA_ANALYSIS = "data_analysis"
    TECHNICAL_CODE = "technical_code"
    CONVERSATIONAL = "conversational"
    OUT_OF_SCOPE = "out_of_scope"


class RouteDestination(str, Enum):
    """Execution targets corresponding to categorized intents."""
    DOCUMENT_RAG_PIPELINE = "document_retrieval_rag"
    SUMMARIZATION_PIPELINE = "document_summarization"
    GENERAL_LLM_PIPELINE = "direct_llm_response"
    EXTRACTION_PIPELINE = "information_extraction_pipeline"
    DATA_ANALYSIS_PIPELINE = "data_analysis_pipeline"
    TECHNICAL_HANDLER = "technical_coding_handler"
    CONVERSATIONAL_HANDLER = "conversational_dialogue"
    FALLBACK_HANDLER = "fallback_clarification_handler"


class IntentClassificationResult(BaseModel):
    """Output from the classification engine."""
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence between 0.0 and 1.0")
    reasoning: str = Field(description="Explanation of why this intent was selected")
    requires_retrieval: bool = Field(description="True if Qdrant knowledge base retrieval is required")
    suggested_route: RouteDestination
    target_topic_or_entity: Optional[str] = None


class RoutingDecision(BaseModel):
    """Final decision including confidence evaluation and fallback gating."""
    query: str
    conversation_id: str
    intent: QueryIntent
    confidence: float
    route: RouteDestination
    is_fallback: bool = False
    requires_retrieval: bool = False
    reasoning: str
    context_used: bool = False


class RouterRequest(BaseModel):
    """API payload for query classification and routing."""
    query: str
    conversation_id: Optional[str] = "default_session"
    execute_handler: bool = False  # If True, runs the downstream pipeline and returns response


class RouterResponse(BaseModel):
    """API response model."""
    query: str
    conversation_id: str
    intent: str
    confidence: float
    route: str
    status: str = "success"
    requires_retrieval: bool
    reasoning: str
    is_fallback: bool = False
    handler_response: Optional[str] = None
    citations: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

