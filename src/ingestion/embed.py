import pandas as pd
from uuid import uuid4

from src.database.vector_store import VectorStore
from src.ingestion.chunkers import ChunkRecord


def embed_and_store(
    chunks: list[ChunkRecord],
    vector_store: VectorStore,
) -> None:
    """Embed Docling chunks and persist them through the shared vector store."""
    if not chunks:
        raise ValueError("No text chunks were created; check the source document.")

    embeddings = vector_store.get_embeddings(
        [chunk["chunk_text"] for chunk in chunks]
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks."
        )

    records = [
        {
            "id": str(uuid4()),
            "metadata": {
                "headings": chunk["headings"],
                "content": chunk["content"],
                "source_name": chunk["source_name"],
                "source_uri": chunk["source_uri"],
                "page_numbers": chunk["page_numbers"],
            },
            "contents": chunk["chunk_text"],
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    vector_store.upsert(pd.DataFrame.from_records(records))
