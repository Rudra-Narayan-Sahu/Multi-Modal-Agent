import sys
from app.services.retrieval import (
    retrieve_documents,
    retrieve_context,
    AdvancedRetriever,
    RetrievalConfig,
)

if __name__ == "__main__":
    query = input("Enter a search query: ").strip() or "knowledge graph and multi-agent system"
    print(f"\nEvaluating query: '{query}'")

    retriever = AdvancedRetriever()

    # 1. Hybrid + Reranked search
    print("\n" + "=" * 60)
    print(" 1. ADVANCED HYBRID + FLASH RANK RERANKED RESULTS ")
    print("=" * 60)
    config = RetrievalConfig(
        search_type="hybrid",
        enable_reranker=True,
        top_k=3
    )
    docs = retriever.retrieve(query=query, config=config)
    print(f"Retrieved documents: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Rerank Score : {doc.get('score', 0):.4f}")
        print(f"Dense Score  : {doc.get('dense_score', 0):.4f}")
        print(f"Sparse Score : {doc.get('sparse_score', 0):.4f}")
        print(f"File         : {doc.get('file_name')}")
        print(f"Chunk Index  : {doc.get('chunk_index')}")
        print(f"Snippet      : {doc.get('text', '')[:250]}...")

    # 2. Context with Citations
    print("\n" + "=" * 60)
    print(" 2. SYNTHESIZED CONTEXT WITH CITATIONS ")
    print("=" * 60)
    context_data = retriever.retrieve_context_with_citations(query=query, config=config)
    print(context_data["context"][:600] + "\n...")
    print("\n" + context_data["citations"])
