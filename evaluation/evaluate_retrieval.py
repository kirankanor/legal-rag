"""Retrieve chunks for golden questions and calculate retrieval Recall@k."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.vector_store import VectorStore  # noqa: E402
from evaluation.metrices.retrieval_metrics import recall_at_k  # noqa: E402


GOLDENS_PATH = PROJECT_ROOT / "dataset" / "goldens.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "retrieval_results.json"
COLLECTIONS = ("legal_cases", "citizenship_guides")


def _load_rows(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{description} not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return data


def _reference_contexts(golden: dict[str, Any], index: int) -> list[str] | None:
    references = golden.get("relevant_contexts", golden.get("context"))
    if references is None:
        return None
    if not isinstance(references, list) or any(
        not isinstance(reference, str) for reference in references
    ):
        raise ValueError(
            f"Golden example {index + 1} needs a list of string "
            "'relevant_contexts' or 'context' values."
        )
    return references or None


def _retrieve_contexts(
    store: VectorStore,
    question: str,
    top_k: int,
) -> list[str]:
    retrieved = store.search(
        question,
        limit=top_k,
        collection_name=COLLECTIONS,
    )
    if retrieved.empty:
        return []
    columns = retrieved.columns.tolist()
    if "content" not in columns:
        raise ValueError("Retrieval results do not contain a content column.")
    content = retrieved.iloc[:, columns.index("content")]
    return [str(value) for value in content.dropna().tolist()]


def _save_results(path: Path, results: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            [results[index] for index in sorted(results)],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def evaluate_retrieval(
    goldens_path: Path,
    output_path: Path,
    *,
    top_k: int,
    limit: int | None,
) -> None:
    goldens = _load_rows(goldens_path, "Golden dataset")
    if limit is not None:
        goldens = goldens[:limit]

    results: dict[int, dict[str, Any]] = {}
    store = VectorStore()
    try:
        for index, golden in enumerate(goldens):
            question = golden.get("input")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Golden example {index + 1} needs a non-empty string input."
                )

            print(f"Retrieving example {index + 1}/{len(goldens)}...", flush=True)
            contexts = _retrieve_contexts(store, question, top_k)
            references = _reference_contexts(golden, index)
            recall = (
                recall_at_k(references, contexts)
                if references is not None
                else None
            )
            results[index] = {
                "index": index,
                "input": question,
                "top_k": top_k,
                "relevant_contexts": references,
                "retrieved_contexts": contexts,
                "retrieval_recall_at_k": recall,
            }
    finally:
        store.close()

    _save_results(output_path, results)
    recall_values = [
        row["retrieval_recall_at_k"]
        for row in results.values()
        if isinstance(row["retrieval_recall_at_k"], (int, float))
    ]
    print(f"Retrieval results saved to {output_path}")
    if recall_values:
        mean_recall = sum(recall_values) / len(recall_values)
        print(f"Mean retrieval Recall@{top_k}: {mean_recall:.3f}")
    else:
        print(
            "Retrieval Recall@k was not calculated; add relevant_contexts "
            "or context passages to the golden rows."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve chunks for golden questions, save them, and calculate "
            "Recall@k without generating answers."
        )
    )
    parser.add_argument("--goldens", type=Path, default=GOLDENS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.output.resolve() == args.goldens.resolve():
        parser.error("--output must not overwrite the golden dataset.")

    try:
        evaluate_retrieval(
            args.goldens,
            args.output,
            top_k=args.top_k,
            limit=args.limit,
        )
    except (ValueError, RuntimeError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
