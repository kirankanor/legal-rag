import json
from pathlib import Path
from ingestion.ingest import load_document

doc_dir = Path(r"C:\Users\91932\OneDrive\Documents\Desktop\legal-rag\docs")
output_dir = Path("parsed_results")

for doc in doc_dir.iterdir():
    parsed_doc = load_document(doc)

    output_file = output_dir / f"{doc.stem}.md"
    output_file.write_text(
        parsed_doc.export_to_markdown(),
        encoding="utf-8"
    )
    print(f"Saved: {output_file}")

