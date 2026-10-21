from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.query.generate import generate_answer


embedding_model: str = "all-MiniLM-L6-v2"
qdrant_path = str(Path(__file__).resolve().parent / "data" / "qdrant")


def retrieve(
    query: str,
    client: QdrantClient,
    embedder: SentenceTransformer,
    top_k: int = 3,
    *,
    collection_name: str | list[str],
) -> list[dict]:
    """Embed the query and return the top k most relevant chunks."""
    query_vector = embedder.encode(query)

    collections = [collection_name] if isinstance(collection_name, str) else collection_name
    if not collections or any(not name.strip() for name in collections):
        raise ValueError("Provide at least one non-empty collection name.")

    results = []
    for target_collection in dict.fromkeys(collections):
        hits = client.query_points(
            collection_name=target_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        results.extend(
            {
                **(hit.payload or {}),
                "score": round(hit.score, 2),
                "collection_name": target_collection,
            }
            for hit in hits.points
        )

    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        parts.append(f"[Source {i}]\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def main() -> None:
    query = "what is the case selection principle?"
    client = QdrantClient(path=qdrant_path)
    try:
        embedder = SentenceTransformer(embedding_model)
        retrieved_chunks = retrieve(
            client=client,
            embedder=embedder,
            query=query,
            collection_name=["legal_cases", "citizenship_guides"],
        )
        context = build_context(retrieved_chunks)
        answer = generate_answer(context=context, query=query)
        print(answer)
    finally:
        client.close()


if __name__ == "__main__":
    main()