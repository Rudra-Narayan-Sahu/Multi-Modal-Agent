import time
import logfire
from typing import Dict, Any, Optional

from app.router.models import (
    QueryIntent,
    RouteDestination,
    RoutingDecision,
    RouterRequest,
    RouterResponse,
)
from app.router.classifier import IntentClassifier, INTENT_TO_ROUTE
from app.router.context_store import conversation_store
from app.router.handlers import (
    handle_document_rag,
    handle_summarization,
    handle_general_qa,
    handle_technical_code,
    handle_data_analysis,
    handle_conversational,
    handle_fallback,
)
from app.services.retrieval.pipeline import AdvancedRetriever


class QueryRouter:
    """
    Production-grade Intelligent Query Router.
    Analyzes intent, evaluates confidence, enforces routing thresholds,
    maintains conversation history, and dispatches to specialized handlers.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.68,
        retriever: Optional[AdvancedRetriever] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.classifier = IntentClassifier()
        self.retriever = retriever or AdvancedRetriever()

    def route(self, query: str, conversation_id: str = "default_session") -> RoutingDecision:
        """
        Classifies incoming query and determines the destination pipeline
        applying confidence thresholds and fallback gating.
        """
        query_clean = query.strip()
        with logfire.span("Query Routing", query=query_clean, session=conversation_id):
            classification = self.classifier.classify(query=query_clean, conversation_id=conversation_id)

            # Confidence-Based Routing Gating
            is_fallback = False
            route = classification.suggested_route

            if classification.confidence < self.confidence_threshold:
                logfire.warning(
                    f"Low classification confidence ({classification.confidence:.2f} < {self.confidence_threshold:.2f}). "
                    f"Gating to Fallback Handler."
                )
                route = RouteDestination.FALLBACK_HANDLER
                is_fallback = True

            history_turns = conversation_store.get_history(conversation_id)
            context_used = len(history_turns) > 0

            decision = RoutingDecision(
                query=query_clean,
                conversation_id=conversation_id,
                intent=classification.intent,
                confidence=round(classification.confidence, 4),
                route=route,
                is_fallback=is_fallback,
                requires_retrieval=classification.requires_retrieval and not is_fallback,
                reasoning=classification.reasoning,
                context_used=context_used
            )
            return decision

    def execute_route(
        self,
        decision: RoutingDecision,
        conversation_id: str = "default_session"
    ) -> Dict[str, Any]:
        """Dispatches query to the determined specialized handler based on the routing decision."""
        history_prompt = conversation_store.format_history_prompt(conversation_id)

        if decision.is_fallback or decision.route == RouteDestination.FALLBACK_HANDLER:
            return handle_fallback(
                query=decision.query,
                confidence=decision.confidence,
                reasoning=decision.reasoning
            )

        elif decision.route == RouteDestination.DOCUMENT_RAG_PIPELINE:
            return handle_document_rag(
                query=decision.query,
                retriever=self.retriever,
                conversation_history=history_prompt
            )

        elif decision.route == RouteDestination.SUMMARIZATION_PIPELINE:
            return handle_summarization(
                query=decision.query,
                retriever=self.retriever,
                conversation_history=history_prompt
            )

        elif decision.route == RouteDestination.GENERAL_LLM_PIPELINE:
            return handle_general_qa(
                query=decision.query,
                conversation_history=history_prompt
            )

        elif decision.route == RouteDestination.TECHNICAL_HANDLER:
            return handle_technical_code(
                query=decision.query,
                conversation_history=history_prompt
            )

        elif decision.route in (RouteDestination.DATA_ANALYSIS_PIPELINE, RouteDestination.EXTRACTION_PIPELINE):
            return handle_data_analysis(
                query=decision.query,
                retriever=self.retriever,
                conversation_history=history_prompt
            )

        elif decision.route == RouteDestination.CONVERSATIONAL_HANDLER:
            return handle_conversational(
                query=decision.query,
                conversation_history=history_prompt
            )

        return handle_fallback(
            query=decision.query,
            confidence=decision.confidence,
            reasoning="Unmapped route fallback."
        )

    def process(self, request: RouterRequest) -> RouterResponse:
        """End-to-end processing: classification, routing, and optional execution."""
        start_time = time.time()
        conv_id = request.conversation_id or "default_session"

        # 1. Routing Decision
        decision = self.route(query=request.query, conversation_id=conv_id)

        # 2. Execution if requested
        handler_response = None
        citations = None
        docs = []

        if request.execute_handler:
            result = self.execute_route(decision=decision, conversation_id=conv_id)
            handler_response = result.get("answer")
            citations = result.get("citations")
            docs = result.get("documents", [])

            # Update conversation memory
            conversation_store.add_turn(
                conversation_id=conv_id,
                role="user",
                content=request.query,
                intent=decision.intent.value
            )
            if handler_response:
                conversation_store.add_turn(
                    conversation_id=conv_id,
                    role="assistant",
                    content=handler_response,
                    intent=decision.intent.value
                )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return RouterResponse(
            query=decision.query,
            conversation_id=conv_id,
            intent=decision.intent.value,
            confidence=decision.confidence,
            route=decision.route.value,
            status="success",
            requires_retrieval=decision.requires_retrieval,
            reasoning=decision.reasoning,
            is_fallback=decision.is_fallback,
            handler_response=handler_response,
            citations=citations,
            metadata={
                "latency_ms": latency_ms,
                "context_used": decision.context_used,
                "retrieved_document_count": len(docs)
            }
        )


# Global singleton instance
default_router = QueryRouter()

