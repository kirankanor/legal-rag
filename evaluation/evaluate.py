"""Run RAG metrics against reviewed golden examples."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from deepeval.test_case import LLMTestCase  # noqa: E402

from evaluation.metrices.answer_metrics import build_answer_metrics  # noqa: E402
from evaluation.metrices.ollama_judge import OllamaJudge  # noqa: E402
from evaluation.metrices.retrieval_metrics import recall_at_k  # noqa: E402


GOLDENS_PATH = PROJECT_ROOT / "dataset" / "goldens.json"
ANSWERS_PATH = PROJECT_ROOT / "dataset" / "generated_answers.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "results.json"


def _fingerprint(
    row: dict[str, Any],
    answer_row: dict[str, Any],
) -> str:
    content = {
        "input": row.get("input"),
        "expected_output": row.get("expected_output"),
        "relevant_contexts": row.get("relevant_contexts"),
        "context": row.get("context"),
        "generated_answer": answer_row.get("generated_answer"),
        "retrieved_contexts": answer_row.get("retrieved_contexts"),
        "top_k": answer_row.get("top_k"),
        "provider": answer_row.get("provider"),
    }
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _metric_result(metric: Any) -> dict[str, Any]:
    return {"score": metric.score, "reason": metric.reason}


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


def _load_results(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return {
        row["index"]: row
        for row in data
        if isinstance(row.get("index"), int)
    }


def evaluate(
    goldens_path: Path,
    answers_path: Path,
    results_path: Path,
    *,
    limit: int | None,
) -> None:
    try:
        goldens: Any = json.loads(goldens_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Golden dataset not found: {goldens_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid golden dataset JSON: {error}") from error
    if not isinstance(goldens, list) or any(not isinstance(row, dict) for row in goldens):
        raise ValueError(f"Expected a JSON array of golden examples in {goldens_path}")
    answers = _load_results(answers_path)
    if limit is not None:
        goldens = goldens[:limit]
    for index, golden in enumerate(goldens):
        answer_row = answers.get(index)
        if answer_row is None:
            raise ValueError(
                f"No generated answer for example {index + 1}. "
                "Run `uv run python evaluation\\generate_answers.py` first."
            )
        if answer_row.get("input") != golden.get("input"):
            raise ValueError(
                f"Generated answer {index + 1} does not match its golden question."
            )

    saved_results = _load_results(results_path)
    judge = OllamaJudge()
    for index, golden in enumerate(goldens):
        answer_row = answers[index]

        question = golden.get("input")
        answer = answer_row.get("generated_answer")
        retrieved_contexts = answer_row.get("retrieved_contexts")
        reference_answer = golden.get("expected_output")
        if not isinstance(question, str):
            raise ValueError(f"Golden example {index + 1} has an invalid input.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Generated answer {index + 1} is empty or invalid.")
        if not isinstance(retrieved_contexts, list) or not all(
            isinstance(context, str) for context in retrieved_contexts
        ):
            raise ValueError(
                f"Generated answer {index + 1} has invalid retrieved_contexts."
            )
        if not retrieved_contexts:
            raise ValueError(f"Generated answer {index + 1} has no retrieved context.")
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise ValueError(f"Golden example {index + 1} has invalid expected_output.")

        fingerprint = _fingerprint(golden, answer_row)
        existing = saved_results.get(index)
        if existing and existing.get("fingerprint") == fingerprint:
            print(f"Skipping example {index + 1}/{len(goldens)} (already evaluated)")
            continue

        print(f"Evaluating example {index + 1}/{len(goldens)}...", flush=True)
        has_reference = isinstance(reference_answer, str) and bool(reference_answer.strip())
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=reference_answer if has_reference else None,
            context=retrieved_contexts,
            retrieval_context=retrieved_contexts,
        )

        metrics = build_answer_metrics(
            judge,
            include_reference_metrics=has_reference,
        )
        metric_results = {}
        for name, metric in metrics.items():
            metric.measure(test_case)
            metric_results[name] = _metric_result(metric)

        relevant_contexts = golden.get(
            "relevant_contexts",
            golden.get("context"),
        )
        recall_value = None
        if isinstance(relevant_contexts, list) and all(
            isinstance(context, str) for context in relevant_contexts
        ) and relevant_contexts:
            recall_value = recall_at_k(relevant_contexts, retrieved_contexts)

        saved_results[index] = {
            "index": index,
            "fingerprint": fingerprint,
            "input": question,
            "family": golden.get("family"),
            "top_k": answer_row.get("top_k"),
            "provider": answer_row.get("provider"),
            "retrieved_contexts": retrieved_contexts,
            "generated_answer": answer,
            "retrieval_recall_at_k": recall_value,
            "metrics": metric_results,
        }
        _save_results(results_path, saved_results)

    evaluated = [
        row
        for row in saved_results.values()
        if isinstance(row.get("index"), int)
        and 0 <= row["index"] < len(goldens)
        and row.get("fingerprint")
        == _fingerprint(goldens[row["index"]], answers[row["index"]])
    ]
    print(f"Evaluation results saved to {results_path}")
    print(f"Examples evaluated or resumed: {len(evaluated)}")

    recall_values = [
        row["retrieval_recall_at_k"]
        for row in evaluated
        if isinstance(row.get("retrieval_recall_at_k"), (int, float))
    ]
    if recall_values:
        mean_recall = sum(recall_values) / len(recall_values)
        print(f"Mean retrieval Recall@k: {mean_recall:.3f}")
    else:
        print(
            "Retrieval Recall@k: not calculated; add a human-reviewed "
            "'relevant_contexts' or 'context' list to golden rows."
        )

    metric_names = {
        name
        for row in evaluated
        for name in row.get("metrics", {})
    }
    for name in sorted(metric_names):
        values = [
            row["metrics"][name]["score"]
            for row in evaluated
            if isinstance(
                row.get("metrics", {}).get(name, {}).get("score"),
                (int, float),
            )
        ]
        if values:
            print(f"Mean {name}: {sum(values) / len(values):.3f}")
    if any(
        not isinstance(row.get("expected_output"), str)
        or not row["expected_output"].strip()
        for row in goldens
    ):
        print(
            "Contextual recall and answer correctness are skipped for examples "
            "without a reference expected_output."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate previously generated answers against golden questions."
    )
    parser.add_argument("--goldens", type=Path, default=GOLDENS_PATH)
    parser.add_argument("--answers", type=Path, default=ANSWERS_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N examples (useful for a small initial run).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.output.resolve() in {args.goldens.resolve(), args.answers.resolve()}:
        parser.error("--output must not overwrite the golden or generated answers.")

    try:
        evaluate(
            args.goldens,
            args.answers,
            args.output,
            limit=args.limit,
        )
    except (ValueError, RuntimeError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
