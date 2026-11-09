from typing import Optional
from langchain.tools import tool
from app.services.retrieval.retriever import retrieve_context


@tool
def search_knowledge_base(query: str) -> str:
    """
    Use this tool to search the internal knowledge base for information 
    about enterprise documents, research papers, and company data.
    Uses multi-stage hybrid search (dense + BM25 keyword) with cross-encoder reranking
    and returns contextual evidence along with document citations.
    """
    return retrieve_context(query, top_k=5, include_citations=True)


@tool
def search_knowledge_base_filtered(query: str, file_name: Optional[str] = None) -> str:
    """
    Use this tool to search the internal knowledge base filtered to a specific document or file name.
    """
    filters = {"file_name": file_name} if file_name else None
    return retrieve_context(query, top_k=5, include_citations=True, filters=filters)
