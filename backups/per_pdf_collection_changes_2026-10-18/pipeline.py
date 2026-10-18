from pathlib import Path
import re

from src.config import settings
from src.database.vector_store import VectorStore
from src.ingestion.chunkers import ChunkRecord, hierarchical_chunking
from src.ingestion.embed import embed_and_store
from src.ingestion.parsers import load_document


def _collection_name_for_file(source_file: Path) -> str:
    document_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", source_file.stem).strip("_")
    if not document_name:
        raise ValueError(f"Cannot derive a collection name from: {source_file.name}")
    return f"{settings.collection_name}_{document_name.lower()}"


def run_ingestion() -> None:
    """Store each source document in its own Qdrant collection."""
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = Path(settings.docs_dir).expanduser()
    if not docs_dir.is_absolute():
        docs_dir = project_root / docs_dir

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    source_files = sorted(path for path in docs_dir.iterdir() if path.is_file())
    if not source_files:
        raise ValueError(f"No documents found in: {docs_dir}")

    documents: list[tuple[str, list[ChunkRecord]]] = []
    used_collections: set[str] = set()
    for source_file in source_files:
        print(f"Processing: {source_file}")
        collection_name = _collection_name_for_file(source_file)
        if collection_name in used_collections:
            raise ValueError(
                f"Multiple source files map to the collection '{collection_name}'. "
                "Rename the files so each document has a unique name."
            )
        used_collections.add(collection_name)

        document = load_document(source=str(source_file))
        chunks = hierarchical_chunking(document)
        if not chunks:
            raise ValueError(f"Docling did not produce chunks for: {source_file}")
        documents.append((collection_name, chunks))

    store = VectorStore()
    try:
        for collection_name, chunks in documents:
            print(f"Storing {len(chunks)} chunks in: {collection_name}")
            embed_and_store(chunks, store, collection_name=collection_name)
    finally:
        store.close()


if __name__ == "__main__":
    run_ingestion()
