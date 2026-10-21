curate your data if possible.

The parser's job is to extract the structure. The chunker's job is to decide how to make that structure searchable.


Use `DocumentConverter()` to convert entire documents into structured content for RAG; 
use `DocumentExtractor()` to extract specific fields, such as case name, court, judgment date, and legal provisions.


# Chunking

All chunking functions take a `DoclingDocument` and return records containing:

- `headings`: detected section headings
- `content`: clean chunk text
- `chunk_text`: Docling's contextualized text, ready for embedding
- `source_name` and `source_uri`: source information when Docling provides it
- `page_numbers`: page numbers from the chunk's source provenance

Choose the method that fits the document:

- `hybrid_chunking(document, tokenizer_model="all-MiniLM-L6-v2")` uses Docling's token-aware `HybridChunker`. Set `tokenizer_model` to the same Hugging Face tokenizer used by the embedding model so chunk limits align with embedding tokens. `max_tokens` can override the tokenizer's configured limit; it requires `tokenizer_model`.
- `hierarchical_chunking(document)` follows Docling's document elements and headings without token-based merging and splitting.
- `page_based_chunking(document)` creates page-level chunks for page-focused review and citations. It does not enforce model token limits.
- `line_based_token_chunking(document, prefix="...")` uses Docling's `LineBasedTokenChunker`, which tries to keep lines together. This is useful for tables and lists.
- `advanced_chunking_and_serialization(document, output_format="markdown")` returns hybrid chunks and a full-document Markdown or HTML serialization. Set `image_placeholder` to change Markdown image placeholders or `triplet_tables=True` to serialize Markdown tables as triplets. HTML output does not support triplet table serialization.
- `trivial_chunking(document)` serializes the full document as one unsplit chunk. It may be too large for embedding models, so prefer the other methods for long documents.

## Ingesting document sets

`DocumentSet` groups an input directory with an explicitly provided Qdrant collection name. `ingest_document_set()` parses and chunks each file, then calls `embed_and_store()` to embed and upsert it into that collection. `run_ingestion()` requires one or more document sets. Collection names are not configured globally; provide them when ingesting, searching, or deleting.

To use different collections, put each group of files in its own directory and pass multiple sets:

```python
from pathlib import Path

from src.ingestion.pipeline import DocumentSet, run_ingestion

run_ingestion(
    [
        DocumentSet(Path("docs/cases"), "legal_cases"),
        DocumentSet(Path("docs/guides"), "citizenship_guides"),
    ]
)
```

You can also provide the collection names on the command line:

```powershell
python -m src.ingestion.pipeline `
  --document-set "docs/cases=legal_cases" `
  --document-set "docs/guides=citizenship_guides"
```

Running `python -m src.ingestion.pipeline` without options uses those two
document sets by default: `docs/cases` goes to `legal_cases`, and `docs/guides`
goes to `citizenship_guides`.

Search one collection with `VectorStore.search(query_text, collection_name="legal_cases")`, or search across both and get the best combined matches with:

```python
results = store.search(
    "What documents are required?",
    collection_name=["legal_cases", "citizenship_guides"],
    limit=5,
)
```

`limit` is the total number of results returned across the requested collections. Each result's metadata includes its `collection_name`.
