import time
from typing import List, Dict, Any
from app.services.retrieval.pipeline import AdvancedRetriever, RetrievalConfig
from app.services.retrieval.reranker import rerank_documents


def evaluate_queries(queries: List[str]):
    """
    Benchmarks and evaluates different retrieval strategies across a set of test queries:
    1. Dense Only
    2. Sparse (BM25) Only
    3. Hybrid (Dense + BM25 via RRF)
    4. Hybrid + Cross-Encoder Reranking
    5. Hybrid + Cross-Encoder + MMR
    """
    retriever = AdvancedRetriever()
    print("=" * 80)
    print(" ADVANCED RAG RETRIEVAL PIPELINE EVALUATION & BENCHMARK ")
    print("=" * 80)

    strategies = [
        ("Dense Vector Search", RetrievalConfig(search_type="dense", enable_reranker=False, top_k=5)),
        ("Sparse BM25 Search", RetrievalConfig(search_type="sparse", enable_reranker=False, top_k=5)),
        ("Hybrid (RRF Fusion)", RetrievalConfig(search_type="hybrid", enable_reranker=False, top_k=5)),
        ("Hybrid + FlashRank Rerank", RetrievalConfig(search_type="hybrid", enable_reranker=True, rerank_top_k=5, top_k=5)),
        ("Hybrid + FlashRank + MMR", RetrievalConfig(search_type="hybrid", enable_reranker=True, enable_mmr=True, top_k=5)),
    ]

    for q_idx, query in enumerate(queries, start=1):
        print(f"\n[Test Query {q_idx}]: \"{query}\"")
        print("-" * 80)
        print(f"{'Strategy':<28} | {'Latency':<9} | {'Docs':<5} | {'Avg Score':<10} | {'Sources'}")
        print("-" * 80)

        for name, config in strategies:
            start_time = time.time()
            results = retriever.retrieve(query=query, config=config)
            latency_ms = (time.time() - start_time) * 1000

            doc_count = len(results)
            scores = [r.get("score", 0.0) for r in results]
            avg_score = sum(scores) / doc_count if doc_count > 0 else 0.0
            sources = set(r.get("file_name", "N/A") for r in results)
            sources_summary = f"{len(sources)} files ({', '.join(list(sources)[:2])}{'...' if len(sources) > 2 else ''})"

            print(f"{name:<28} | {latency_ms:6.1f} ms | {doc_count:<5} | {avg_score:10.4f} | {sources_summary}")

    print("\n" + "=" * 80)
    print("Evaluation benchmark completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    test_queries = [
        "knowledge graph and reasoning in artificial intelligence",
        "multi-agent orchestration and large language models",
        "retrieval augmented generation benchmark evaluation metrics"
    ]
    evaluate_queries(test_queries)

