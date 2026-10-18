from pathlib import Path

from src.config import settings
from src.database.vector_store import VectorStore
from src.ingestion.chunkers import ChunkRecord, hierarchical_chunking
from src.ingestion.embed import embed_and_store
from src.ingestion.parsers import load_document


def run_ingestion() -> None:
    """Parse, chunk, embed, and store every file in the configured documents folder."""
    project_root = Path(__file__).resolve().parents[2]
    docs_dir = Path(settings.docs_dir).expanduser()
    if not docs_dir.is_absolute():
        docs_dir = project_root / docs_dir

    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {docs_dir}")

    source_files = sorted(path for path in docs_dir.iterdir() if path.is_file())
    if not source_files:
        raise ValueError(f"No documents found in: {docs_dir}")

    chunks: list[ChunkRecord] = []
    for source_file in source_files:
        print(f"Processing: {source_file}")
        document = load_document(source=str(source_file))
        chunks.extend(hierarchical_chunking(document))

    if not chunks:
        raise ValueError("Docling did not produce any chunks from the source documents.")

    store = VectorStore()
    try:
        embed_and_store(chunks, store)
    finally:
        store.close()


if __name__ == "__main__":
    run_ingestion()
