from app.services.retrieval import retrieve_documents, retrieve_context

if __name__ == "__main__":
    query = input("Enter a search query: ").strip() or "test"
    docs = retrieve_documents(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Retrieved documents: {len(docs)}")

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {doc.get('score')}")
        print(f"File: {doc.get('file_name')}")
        print(f"Source: {doc.get('source_name')}")
        print(f"Text: {doc.get('text', '')[:500]}")

    print("\nContext preview:")
    print(retrieve_context(query, top_k=3)[:1000])
