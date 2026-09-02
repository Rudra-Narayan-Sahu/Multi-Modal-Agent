import sys

# Configure UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

from app.services.retrieval import retrieve_documents, retrieve_context

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Enter a search query (default: 'knowledge graph'): ").strip() or "knowledge graph"

    print("=" * 60)
    print(f"🔍 SIMILARITY SEARCH ON QDRANT")
    print(f"Query: {query}")
    print("=" * 60)

    docs = retrieve_documents(query, top_k=3)
    print(f"Retrieved Chunks: {len(docs)}\n")

    for i, doc in enumerate(docs, start=1):
        print(f"--- [Result {i}] Score: {doc.get('score')} ---")
        print(f"Document: {doc.get('file_name')} (Chunk #{doc.get('chunk_index')})")
        snippet = doc.get('text', '').replace('\n', ' ')[:300]
        print(f"Content:  {snippet}...\n")

    if not docs:
        print("No documents found matching the query. Make sure documents are ingested via 'python ingest.py'.")

if __name__ == "__main__":
    main()

