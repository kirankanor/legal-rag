from collections.abc import Sequence

import pandas as pd

from src.database.vector_store import VectorStore


def _format_table(results: pd.DataFrame) -> str:
    columns = list(results.columns)
    rows = [
        [
            " ".join(str(value).split())
            for value in row
        ]
        for row in results.itertuples(index=False, name=None)
    ]
    widths = [
        max([len(str(column)), *(len(row[index]) for row in rows)])
        for index, column in enumerate(columns)
    ]

    def format_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    header = format_row([str(column) for column in columns])
    body = [format_row(row) for row in rows]
    return "\n".join([border, header, border, *body, border])


def retrieve(
    query_text: str,
    *,
    collection_name: str | Sequence[str],
    limit: int = 3,
) -> pd.DataFrame:
    """Search one or more Qdrant collections for relevant chunks."""
    store = VectorStore()
    try:
        return store.search(
            query_text,
            collection_name=collection_name,
            limit=limit,
        )
    finally:
        store.close()


def main() -> None:
    results = retrieve(
        "What documents are required?",
        collection_name=["legal_cases", "citizenship_guides"],
    )
    if results.empty:
        print("No matching chunks found.")
    else:
        display_results = results.drop(columns=["embedding"], errors="ignore")
        print(_format_table(display_results))


if __name__ == "__main__":
    main()
