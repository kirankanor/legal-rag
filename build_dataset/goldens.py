"""Create a quality-filtered golden dataset from generated silvers.
uv run python build_dataset\goldens.py 0.8
"""

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
SILVERS_PATH = DATASET_DIR / "silvers.json"
GOLDENS_PATH = DATASET_DIR / "goldens.json"
QUALITY_FIELD = "synthetic_input_quality"


def filter_silvers(
    rows: list[dict[str, Any]],
    minimum_score: float,
) -> list[dict[str, Any]]:
    """Return silver rows with a numeric quality score at or above the minimum."""
    selected = []
    for row in rows:
        metadata = row.get("additional_metadata")
        score = metadata.get(QUALITY_FIELD) if isinstance(metadata, dict) else None
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and score >= minimum_score
        ):
            selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create goldens.json from silvers meeting a minimum quality score."
    )
    parser.add_argument(
        "minimum_score",
        type=float,
        help="Include silvers with synthetic_input_quality >= this value (0 to 1)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SILVERS_PATH,
        help=f"Source silver dataset (default: {SILVERS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDENS_PATH,
        help=f"Filtered dataset output (default: {GOLDENS_PATH})",
    )
    args = parser.parse_args()

    if not 0 <= args.minimum_score <= 1:
        parser.error("minimum_score must be between 0 and 1.")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the source silvers dataset.")

    try:
        rows: Any = json.loads(args.input.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"Silver dataset not found: {args.input}")
    except json.JSONDecodeError as error:
        parser.error(f"Invalid JSON in {args.input}: {error}")

    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        parser.error(f"Expected a JSON array of silver objects in {args.input}")

    goldens = filter_silvers(rows, args.minimum_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(goldens, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(args.output)

    missing_scores = len(rows) - sum(
        1
        for row in rows
        if isinstance(row.get("additional_metadata"), dict)
        and isinstance(row["additional_metadata"].get(QUALITY_FIELD), (int, float))
        and not isinstance(row["additional_metadata"].get(QUALITY_FIELD), bool)
    )
    print(f"Source silvers: {len(rows)}")
    print(f"Minimum quality score: {args.minimum_score:g}")
    print(f"Goldens saved: {len(goldens)} -> {args.output}")
    if missing_scores:
        print(f"Skipped rows with missing/invalid quality score: {missing_scores}")


if __name__ == "__main__":
    main()
