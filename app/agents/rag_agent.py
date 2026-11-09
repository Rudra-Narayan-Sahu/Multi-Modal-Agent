import logfire
from typing import Dict, Any, List
from app.agents.state import AgentState
from app.services.retrieval.retriever import retrieve_documents
from app.agents.llm import agent_llm
from app.agents.prompts.rag_prompt import RAG_ANALYSIS_PROMPT


def rag_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: RAG Knowledge Base Agent.
    Retrieves relevant chunks from Qdrant via Advanced Hybrid Pipeline,
    validates relevance, extracts metadata and citations.
    """
    query = state.get("user_query", "")
    with logfire.span("RAG Agent Node", query=query):
        try:
            # 1. Retrieve top chunks using hybrid dense + sparse + reranker
            retrieved_chunks = retrieve_documents(
                query=query,
                top_k=5,
                search_type="hybrid",
                enable_reranker=True
            )
        except Exception as e:
            logfire.error(f"RAG retrieval failed: {e}")
            retrieved_chunks = []

        if not retrieved_chunks:
            return {
                "retrieved_documents": [],
                "agent_messages": [{
                    "agent": "rag_agent",
                    "action": "retrieve",
                    "status": "warning",
                    "message": "No matching documents found in Qdrant knowledge base."
                }],
                "errors": ["RAG Agent: No relevant chunks found in internal knowledge base."]
            }

        # 2. Extract structured citations
        citations: List[Dict[str, Any]] = []
        formatted_chunks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            file_name = chunk.get("file_name", "Unknown File")
            chunk_idx = chunk.get("chunk_index", 0)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")

            citations.append({
                "type": "document",
                "file_name": file_name,
                "chunk_index": chunk_idx,
                "score": round(score, 4),
                "preview": text[:150] + "..." if len(text) > 150 else text
            })

            formatted_chunks.append(
                f"[Chunk {i}] File: {file_name} (Chunk #{chunk_idx}, Relevance Score: {score:.3f})\n{text}"
            )

        chunks_text = "\n\n".join(formatted_chunks)

        # 3. Grounding check via LLM
        prompt = RAG_ANALYSIS_PROMPT.format(
            query=query,
            chunks_formatted=chunks_text
        )
        analysis = agent_llm.call_json([
            {"role": "system", "content": "You are a factual RAG validation agent."},
            {"role": "user", "content": prompt}
        ])

        has_sufficient_context = analysis.get("has_sufficient_context", True)
        synthesized_context = analysis.get("synthesized_context", chunks_text[:1500])

        return {
            "retrieved_documents": retrieved_chunks,
            "source_references": citations,
            "final_context": f"### Document Findings\n{synthesized_context}\n",
            "agent_messages": [{
                "agent": "rag_agent",
                "action": "retrieval_and_analysis",
                "status": "success",
                "chunks_found": len(retrieved_chunks),
                "has_sufficient_context": has_sufficient_context
            }]
        }

