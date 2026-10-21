"""Factory for the answer-quality metrics used by the evaluator."""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams

from evaluation.metrices.ollama_judge import OllamaJudge


def build_answer_metrics(
    judge: OllamaJudge,
    *,
    include_reference_metrics: bool = True,
) -> dict[str, object]:
    """Create DeepEval metrics for retrieval context and generated answers."""
    metrics: dict[str, object] = {
        "faithfulness": FaithfulnessMetric(
            model=judge,
            async_mode=False,
            include_reason=True,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            model=judge,
            async_mode=False,
            include_reason=True,
        ),
    }
    if include_reference_metrics:
        metrics["contextual_recall"] = ContextualRecallMetric(
            model=judge,
            async_mode=False,
            include_reason=True,
        )
        metrics["answer_correctness"] = GEval(
            name="Answer Correctness",
            criteria=(
                "Compare the actual answer with the expected answer for the question. "
                "Score factual agreement and completeness; do not reward unsupported "
                "details or penalize equivalent wording."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            async_mode=False,
        )
    metrics["citation_correctness"] = GEval(
        name="Legal Citation Correctness",
        criteria=(
            "Check whether each case, statute section, or article cited in the "
            "actual answer appears in the retrieved context and whether that "
            "context supports the claim attached to it. Penalize unsupported or "
            "contradicted citations. If the answer makes no citation, score it "
            "low and explain that no citation was provided."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        async_mode=False,
    )
    return metrics
