import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.database.vector_store import VectorStore
from src.ingestion.chunkers import hierarchical_chunking
from src.ingestion.embed import embed_and_store
from src.ingestion.parsers import load_document


@dataclass(frozen=True)
class DocumentSet:
    input_dir: Path
    collection_name: str


def discover_files(input_dir: Path) -> list[Path]:
    """Return files directly inside a document-set directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Document-set directory does not exist: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file())


def ingest_document_set(
    document_set: DocumentSet,
    vector_store: VectorStore,
) -> None:
    """Parse, chunk, embed, and store every file in one document set."""
    collection_name = document_set.collection_name.strip()
    if not collection_name:
        raise ValueError("Document-set collection name cannot be empty.")

    project_root = Path(__file__).resolve().parents[2]
    input_dir = document_set.input_dir.expanduser()
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir

    files = discover_files(input_dir)
    if not files:
        raise ValueError(f"No documents found in: {input_dir}")

    vector_store.ensure_collection(collection_name)
    for file_path in files:
        print(f"Processing {file_path} into collection '{collection_name}'")
        document = load_document(source=str(file_path))
        chunks = hierarchical_chunking(document)
        if not chunks:
            raise ValueError(f"Docling did not produce chunks for: {file_path}")
        embed_and_store(
            chunks,
            vector_store,
            collection_name=collection_name,
        )


def run_ingestion(document_sets: Sequence[DocumentSet]) -> None:
    """Ingest document sets using their explicitly provided collection names."""
    if not document_sets:
        raise ValueError("Configure at least one document set for ingestion.")

    store = VectorStore()
    try:
        for document_set in document_sets:
            ingest_document_set(document_set, store)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest document directories into explicitly named Qdrant collections."
    )
    parser.add_argument(
        "--document-set",
        action="append",
        metavar="INPUT_DIR=COLLECTION_NAME",
        help=(
            "Document directory and its collection name; repeat for each set. "
            "Without this option, the configured cases and guides directories are used."
        ),
    )
    arguments = parser.parse_args()

    document_sets = []
    for value in arguments.document_set or []:
        input_dir, separator, collection_name = value.partition("=")
        if not separator or not input_dir.strip() or not collection_name.strip():
            parser.error(
                f"Invalid document set {value!r}; expected INPUT_DIR=COLLECTION_NAME."
            )
        document_sets.append(
            DocumentSet(Path(input_dir.strip()), collection_name.strip())
        )

    if not document_sets:
        document_sets = [
            DocumentSet(Path("docs/cases"), "legal_cases"),
            DocumentSet(Path("docs/guides"), "citizenship_guides"),
        ]

    run_ingestion(document_sets)


if __name__ == "__main__":
    main()
