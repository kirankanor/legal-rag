"""Judge silver questions with GEval using the local Ollama model."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

try:
    from build_dataset.generate_dataset import OllamaLLM
except ModuleNotFoundError:
    from generate_dataset import OllamaLLM


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "silvers.json"
JUDGMENTS_PATH = DATASET_PATH.with_name("silver_judgments.json")


def judge_silvers(
    rows: list[dict[str, Any]],
    output_path: Path = JUDGMENTS_PATH,
) -> None:
    """Save question-quality and answer-grounding scores, resuming saved work."""
    model = OllamaLLM()
    question_quality = GEval(
        name="Question Quality",
        criteria=(
            "Score how clear, self-contained, and answerable the question is. "
            "A high score means it has a clear objective and does not rely on "
            "unstated references; a low score means it is vague or malformed."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT],
        model=model,
        async_mode=False,
    )
    answer_grounding = GEval(
        name="Answer Grounding",
        criteria=(
            "Judge whether the proposed answer is factually supported by the "
            "provided context and answers the question. Penalize claims not "
            "supported by the context or contradictions."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        model=model,
        async_mode=False,
    )

    completed: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        try:
            existing: Any = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid judgments JSON in {output_path}: {error}") from error
        if not isinstance(existing, list) or any(not isinstance(row, dict) for row in existing):
            raise ValueError(f"Expected a JSON array of judgments in {output_path}")
        completed = {
            row["index"]: row
            for row in existing
            if isinstance(row.get("index"), int)
            and isinstance(row.get("fingerprint"), str)
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    for index, row in enumerate(rows):
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "input": row.get("input"),
                    "expected_output": row.get("expected_output"),
                    "context": row.get("context"),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        previous = completed.get(index)
        if previous and previous["fingerprint"] == fingerprint:
            print(f"Skipping silver {index + 1}/{len(rows)} (already judged)")
            continue

        question = row.get("input")
        answer = row.get("expected_output")
        context = row.get("context")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(
                f"Silver {index + 1} must have string input and expected_output."
            )
        if not isinstance(context, list) or any(
            not isinstance(part, str) for part in context
        ):
            raise ValueError(
                f"Silver {index + 1} must have a list of string context passages."
            )

        print(f"Judging silver {index + 1}/{len(rows)}...", flush=True)
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            context=context,
        )
        question_quality.measure(test_case, _show_indicator=False)
        answer_grounding.measure(test_case, _show_indicator=False)
        completed[index] = {
            "index": index,
            "fingerprint": fingerprint,
            "input": question,
            "family": row.get("family"),
            "question_quality": {
                "score": question_quality.score,
                "reason": question_quality.reason,
            },
            "answer_grounding": {
                "score": answer_grounding.score,
                "reason": answer_grounding.reason,
            },
        }
        temporary_path.write_text(
            json.dumps(
                [completed[key] for key in sorted(completed)],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(output_path)

    print(f"Judgments saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge silver question quality and answer grounding with GEval."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Silver dataset JSON file (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=JUDGMENTS_PATH,
        help=f"Judgment output file (default: {JUDGMENTS_PATH})",
    )
    args = parser.parse_args()

    try:
        rows: Any = json.loads(args.dataset.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"Dataset file not found: {args.dataset}")
    except json.JSONDecodeError as error:
        parser.error(f"Invalid JSON in {args.dataset}: {error}")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        parser.error(f"Expected a JSON array of question objects in {args.dataset}")

    try:
        judge_silvers(rows, args.output)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
