import os
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.config import settings

# ================= Config ===============
load_dotenv(Path(__file__).resolve().parent / ".env")

embedding_model = settings.embedding_model
qdrant_path = str(Path(settings.qdrant_path).expanduser())
collection_name = settings.collection_name


def retrieve(
    query: str,
    client: QdrantClient,
    embedder: SentenceTransformer,
    top_k: int = 3,
    target_collection: str | None = None,
) -> list[dict]:
    """Search one requested collection or all collections for this corpus."""
    if top_k < 1:
        raise ValueError("top_k must be greater than zero.")

    available = {
        item.name for item in client.get_collections().collections
    }
    if target_collection is not None:
        if target_collection not in available:
            raise ValueError(f"Qdrant collection does not exist: {target_collection}")
        collections = [target_collection]
    else:
        prefix = f"{collection_name}_"
        collections = sorted(
            name
            for name in available
            if name.startswith(prefix)
        )
        if not collections and collection_name in available:
            collections = [collection_name]
    if not collections:
        raise ValueError("No document collections found. Ingest documents first.")

    query_vector = embedder.encode(query)

    results = []
    for collection in collections:
        hits = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        for hit in hits.points:
            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            results.append(
                {
                    **metadata,
                    "chunk_text": payload.get("content", payload.get("chunk_text", "")),
                    "score": round(hit.score, 2),
                    "collection_name": collection,
                }
            )

    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source_name") or chunk.get("collection_name", "unknown")
        pages = chunk.get("page_numbers") or []
        citation = f"[Source {i}: {source}"
        if pages:
            citation += f", page(s) {', '.join(map(str, pages))}"
        citation += "]"
        parts.append(f"{citation}\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(context: str, query: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    if not model:
        raise ValueError("GROQ_MODEL is not set.")

    system_prompt = (
        "You are a legal information assistant specializing in Indian citizenship law.\n"
        "Answer using ONLY the provided context; do not make up facts or legal conclusions.\n"
        "If the context does not contain enough information, say so clearly.\n"
        "Cite the relevant case, section, or source when the context identifies one."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0.2,
    )
    messages = [
        ("system", system_prompt),
        ("user", user_message),
    ]
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    query = "what is the case selection principle?"
    client = QdrantClient(path=qdrant_path)
    try:
        embedder = SentenceTransformer(embedding_model)
        retrieved_chunks = retrieve(client=client, embedder=embedder, query=query)
        context = build_context(retrieved_chunks)
        answer = generate_answer(context=context, query=query)
        print(answer)
    finally:
        client.close()


if __name__ == "__main__":
    main()