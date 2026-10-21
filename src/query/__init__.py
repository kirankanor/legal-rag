"""Query and retrieval utilities for legal RAG."""

from .generate import generate_answer
from .retrieve import retrieve

__all__ = ["generate_answer", "retrieve"]
