"""Standalone retrieval metrics that use human-labeled relevant passages."""


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def recall_at_k(relevant_contexts: list[str], retrieved_contexts: list[str]) -> float:
    """Compute passage Recall@k using normalized exact/contained text matches."""
    if not relevant_contexts:
        raise ValueError("Recall@k requires at least one labeled relevant passage.")
    retrieved = [_normalized(text) for text in retrieved_contexts]
    found = sum(
        any(_normalized(reference) in passage for passage in retrieved)
        for reference in relevant_contexts
    )
    return found / len(relevant_contexts)
