from langchain.tools import tool
from app.services.retrieval.retriever import retrieve_context

@tool
def search_knowledge_base(query: str) -> str:
    """
    Use this tool to search the internal knowledge base for information 
    about enterprise documents, research papers, and company data.
    """
    # This automatically embeds the query and searches Qdrant!
    return retrieve_context(query, top_k=5)

# Now you can pass `search_knowledge_base` to your LangGraph or LangChain agent!
