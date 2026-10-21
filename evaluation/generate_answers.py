"""Generate and save RAG answers for golden questions."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.vector_store import VectorStore  # noqa: E402
from src.query.generate import generate_answer_with_provider  # noqa: E402


GOLDENS_PATH = PROJECT_ROOT / "dataset" / "goldens.json"
ANSWERS_PATH = PROJECT_ROOT / "dataset" / "generated_answers.json"
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


def _save_answers(path: Path, answers: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            [answers[index] for index in sorted(answers)],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _retrieve_contexts(store: VectorStore, question: str, top_k: int) -> list[str]:
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


def generate_answers(
    goldens_path: Path,
    answers_path: Path,
    *,
    top_k: int,
    limit: int | None,
    provider: str,
) -> None:
    goldens = _load_rows(goldens_path, "Golden dataset")
    if limit is not None:
        goldens = goldens[:limit]
    answers: dict[int, dict[str, Any]] = {}
    if answers_path.exists():
        for row in _load_rows(answers_path, "Generated answers"):
            index = row.get("index")
            if isinstance(index, int):
                answers[index] = row

    store = VectorStore()
    try:
        for index, golden in enumerate(goldens):
            question = golden.get("input")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Golden example {index + 1} needs a non-empty string input."
                )

            contexts = _retrieve_contexts(store, question, top_k)
            if not contexts:
                raise ValueError(
                    f"No chunks retrieved for golden example {index + 1}."
                )

            existing = answers.get(index)
            if (
                existing
                and existing.get("input") == question
                and existing.get("retrieved_contexts") == contexts
                and existing.get("top_k") == top_k
                and existing.get("requested_provider") == provider
                and isinstance(existing.get("generated_answer"), str)
            ):
                print(f"Skipping example {index + 1}/{len(goldens)} (already generated)")
                continue

            print(f"Generating example {index + 1}/{len(goldens)}...", flush=True)
            answer, used_provider = generate_answer_with_provider(
                context="\n\n".join(contexts),
                query=question,
                provider=provider,
            )
            answers[index] = {
                "index": index,
                "input": question,
                "family": golden.get("family"),
                "top_k": top_k,
                "requested_provider": provider,
                "provider": used_provider,
                "retrieved_contexts": contexts,
                "generated_answer": answer,
            }
            _save_answers(answers_path, answers)
    finally:
        store.close()

    print(f"Generated answers saved to {answers_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers from golden questions and retrieved context, "
            "then save them separately from the golden dataset."
        )
    )
    parser.add_argument("--goldens", type=Path, default=GOLDENS_PATH)
    parser.add_argument("--output", type=Path, default=ANSWERS_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--provider",
        choices=("auto", "groq", "ollama"),
        default="auto",
        help="Use Groq, local Ollama, or try Groq and fall back to Ollama.",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.output.resolve() in {args.goldens.resolve(), GOLDENS_PATH.resolve()}:
        parser.error("--output must not overwrite the golden dataset.")

    try:
        generate_answers(
            args.goldens,
            args.output,
            top_k=args.top_k,
            limit=args.limit,
            provider=args.provider,
        )
    except (ValueError, RuntimeError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
