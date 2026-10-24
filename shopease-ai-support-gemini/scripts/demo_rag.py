"""Small interview/demo utility for proving the RAG retrieval path.

Run from any working directory:
    python scripts/demo_rag.py

It prints the user's query, the actual Chroma/KB retrieval metadata, the selected
source/section, and a short evidence excerpt. It does not generate answers and
never hard-codes customer-facing policy text.
"""

from app.services.rag_service import get_rag_service


DEMO_QUERIES = [
    "How long does express delivery take?",
    "Can I get my order delivered today?",
    "My card was charged twice for the same order.",
    "My tracking hasn't moved for four consecutive business days.",
    "How long will my refund take if I paid by card?",
    "Do you offer 5% cashback on every order?",
]


def main() -> None:
    rag = get_rag_service()
    status = rag.get_status()
    if not status["is_indexed"]:
        print("Knowledge base is not indexed; indexing now...\n")
        print(rag.index_documents(force_reindex=False))
        status = rag.get_status()

    print(f"Indexed chunks: {status['total_chunks']}")
    print(f"Embedding model: {status['embedding_model']}\n")

    for query in DEMO_QUERIES:
        result = rag.search(query, top_k=3)
        print("=" * 78)
        print(f"QUERY: {query}")
        print(f"Sufficient context: {result.has_sufficient_context}")
        if not result.results:
            print("RETRIEVAL: no sufficiently supported evidence")
            continue

        top = result.results[0]
        print(f"SOURCE: {top.source}")
        print(f"SECTION: {top.topic}")
        print(f"SIMILARITY: {top.metadata.get('semantic_similarity')}")
        print(f"HYBRID SCORE: {top.similarity_score}")
        print("EVIDENCE:")
        lines = [line.strip() for line in top.content.splitlines() if line.strip()]
        for line in lines[:8]:
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
