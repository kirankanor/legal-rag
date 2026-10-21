"""Report silver question counts by family and quality score.
uv run python build_dataset\helper.py --list
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "silvers.json"


def summarize_dataset(path: Path, list_questions: bool = False) -> None:
    """Print total questions and counts grouped by family and quality score."""
    try:
        rows: Any = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Dataset file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of question objects in {path}")

    family_counts: Counter[str] = Counter()
    score_counts: Counter[float] = Counter()
    missing_scores = 0

    for row in rows:
        family = row.get("family")
        family_counts[family if isinstance(family, str) else "unknown"] += 1

        metadata = row.get("additional_metadata")
        score = (
            metadata.get("synthetic_input_quality")
            if isinstance(metadata, dict)
            else None
        )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            missing_scores += 1
        else:
            score_counts[float(score)] += 1

    print(f"Total questions: {len(rows)}")
    print("\nQuestions by family:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")

    print("\nQuestions by quality score:")
    for score, count in sorted(score_counts.items()):
        print(f"  {score:g}: {count}")
    if missing_scores:
        print(f"  missing score: {missing_scores}")

    if list_questions:
        print("\nQuestions:")
        for index, row in enumerate(rows, start=1):
            metadata = row.get("additional_metadata")
            score = (
                metadata.get("synthetic_input_quality")
                if isinstance(metadata, dict)
                else None
            )
            score_label = (
                f"{score:g}"
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else "missing"
            )
            family = row.get("family")
            family_label = family if isinstance(family, str) else "unknown"
            print(f"{index}. family={family_label} | quality={score_label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count silver questions by family and quality score."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset JSON file (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List every question with its family and quality score",
    )
    args = parser.parse_args()
    try:
        summarize_dataset(args.dataset, list_questions=args.list)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
