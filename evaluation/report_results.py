"""Build a Markdown report from saved answer and retrieval evaluation results."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWER_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "results.json"
RETRIEVAL_RESULTS_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "retrieval_results.json"
)
REPORT_PATH = PROJECT_ROOT / "evaluation" / "results" / "report.md"
GROUP_FIELDS = ("family", "provider", "top_k")


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


def _index_rows(
    rows: list[dict[str, Any]],
    description: str,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("index")
        if not isinstance(index, int):
            raise ValueError(f"{description} contains a row without an integer index.")
        if index in indexed:
            raise ValueError(f"{description} contains duplicate index {index}.")
        indexed[index] = row
    return indexed


def _combined_results(
    answer_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    retrieval_by_index = _index_rows(retrieval_rows, "Retrieval results")
    combined: list[dict[str, Any]] = []
    metric_names: list[str] = []

    for answer_row in answer_rows:
        index = answer_row.get("index")
        if not isinstance(index, int):
            raise ValueError("Answer results contain a row without an integer index.")

        retrieval_row = retrieval_by_index.get(index)
        if retrieval_row is not None and retrieval_row.get("input") != answer_row.get(
            "input"
        ):
            raise ValueError(
                f"Answer and retrieval results do not match at question {index + 1}."
            )

        scores: dict[str, Any] = {}
        metrics = answer_row.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"Answer results for question {index + 1} have invalid metrics.")
        for name, result in metrics.items():
            if name not in metric_names:
                metric_names.append(name)
            scores[name] = result.get("score") if isinstance(result, dict) else None

        recall = (
            retrieval_row.get("retrieval_recall_at_k")
            if retrieval_row is not None
            else answer_row.get("retrieval_recall_at_k")
        )
        if "retrieval_recall_at_k" not in metric_names:
            metric_names.append("retrieval_recall_at_k")
        scores["retrieval_recall_at_k"] = recall

        combined.append(
            {
                "index": index,
                "input": answer_row.get("input", ""),
                "family": answer_row.get("family", "unknown"),
                "provider": answer_row.get("provider", "unknown"),
                "top_k": answer_row.get("top_k", "unknown"),
                "scores": scores,
            }
        )
    return combined, metric_names


def _numeric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [row["scores"].get(metric) for row in rows]
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}"
    return "—"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _metric_summary(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[str]:
    lines = [
        "| Metric | Mean score | Scored / total |",
        "|---|---:|---:|",
    ]
    for metric in metric_names:
        values = _numeric_values(rows, metric)
        mean = sum(values) / len(values) if values else None
        lines.append(
            f"| `{metric}` | {_format_score(mean)} | {len(values)} / {len(rows)} |"
        )
    return lines


def _group_summary(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    field: str,
) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)

    headers = ["Group", "Questions", *metric_names]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", "---:"] + ["---:"] * len(metric_names)) + "|",
    ]
    for label, group_rows in sorted(groups.items()):
        means = []
        for metric in metric_names:
            values = _numeric_values(group_rows, metric)
            means.append(_format_score(sum(values) / len(values) if values else None))
        lines.append(
            "| "
            + " | ".join(
                [_escape_cell(label), str(len(group_rows)), *means]
            )
            + " |"
        )
    return lines


def build_report(
    answer_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    *,
    group_by: list[str],
) -> str:
    if not answer_rows:
        raise ValueError("Answer results are empty; run the answer evaluator first.")
    rows, metric_names = _combined_results(answer_rows, retrieval_rows)

    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions with answer metrics: **{len(rows)}**",
        f"- Answer metrics: `{ANSWER_RESULTS_PATH.name}`",
        f"- Retrieval metrics: `{RETRIEVAL_RESULTS_PATH.name}`",
        "",
        "## Overall metric summary",
        "",
        *_metric_summary(rows, metric_names),
        "",
    ]

    for field in group_by:
        lines.extend(
            [
                f"## Mean scores grouped by {field}",
                "",
                *_group_summary(rows, metric_names, field),
                "",
            ]
        )

    headers = [
        "ID",
        "Family",
        "Provider",
        "Top-k",
        "Question",
        *metric_names,
    ]
    lines.extend(
        [
            "## Scores by question",
            "",
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---:"] * len(headers)) + "|",
        ]
    )
    for row in rows:
        values = [
            str(row["index"] + 1),
            _escape_cell(row["family"]),
            _escape_cell(row["provider"]),
            _escape_cell(row["top_k"]),
            _escape_cell(row["input"]),
            *[
                _format_score(row["scores"].get(metric))
                for metric in metric_names
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Markdown report with per-question evaluation scores, "
            "overall means, and grouped summaries."
        )
    )
    parser.add_argument("--answers", type=Path, default=ANSWER_RESULTS_PATH)
    parser.add_argument("--retrieval", type=Path, default=RETRIEVAL_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--group-by",
        nargs="+",
        choices=GROUP_FIELDS,
        default=["family", "provider"],
        help="Fields to summarize separately; choose family, provider, and/or top_k.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print the full report, including every per-question score, to the terminal.",
    )
    args = parser.parse_args()

    try:
        answer_rows = _load_rows(args.answers, "Answer evaluation results")
        retrieval_rows = (
            _load_rows(args.retrieval, "Retrieval evaluation results")
            if args.retrieval.exists()
            else []
        )
        report = build_report(
            answer_rows,
            retrieval_rows,
            group_by=args.group_by,
        )
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    terminal_report = (
        report
        if args.show_all
        else report.partition("## Scores by question")[0].rstrip()
    )
    print(terminal_report)
    print(f"Report saved to {args.output}")
    print(f"Questions included: {len(answer_rows)}")


if __name__ == "__main__":
    main()
