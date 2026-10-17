from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ================= Config ===============
embedding_model: str = "all-MiniLM-L6-v2"
qdrant_path = str(Path(__file__).resolve().parent / "data" / "qdrant")
collection_name: str = "legal_docs"
source_file = Path(__file__).resolve().parent / "parsed_results" / "cases.md"


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def split_text(text, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return chunks


def imbed_text(chunk_text: list[str], embedding_model: str, collection_name: str):
    if not chunk_text:
        raise ValueError("No text chunks were created; check the source document.")

    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(chunk_text , show_progress_bar=True)

    client = QdrantClient(path=qdrant_path)
    dim = embedder.get_embedding_dimension()

    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    "content": chunk,
                    "chunk_text": chunk,
                },
            )
            for idx, (chunk, embedding) in enumerate(zip(chunk_text, embeddings))
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)

        info = client.get_collection(collection_name)
        print(
            f"[OK] Indexed {info.points_count} points "
            f"(dim={info.config.params.vectors.size})"
        )
    finally:
        client.close()
    print("\n[DONE] Ingestion complete.")


if __name__ == "__main__":
    text = source_file.read_text(encoding="utf-8")
    chunks = split_text(text=text, chunk_size=500, chunk_overlap=50)
    imbed_text(
        chunk_text=chunks,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
