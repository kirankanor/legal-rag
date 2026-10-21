# RAG evaluation metrics

Start with a small, human-reviewed set of questions, trusted reference answers,
and relevant source passages. Generated silver questions and answers should not
be treated as ground truth until they have been reviewed for legal accuracy.

## Core answer metrics

| Metric | What it checks | Needed for each example |
|---|---|---|
| **Contextual Recall** | Whether the retrieved context contains the information needed to answer the question. | Question, trusted reference answer, retrieved context |
| **Faithfulness** | Whether claims in the generated answer are supported by the retrieved context. | Generated answer, retrieved context |
| **Answer Relevancy** | Whether the generated answer addresses the question instead of going off-topic. | Question, generated answer |
| **Answer Correctness** | Whether the generated answer agrees with a trusted reference answer. | Generated answer, trusted reference answer |

These checks evaluate different failure modes. For example, low contextual
recall points to retrieval, while low faithfulness points to unsupported claims
in generation.

## Legal citation correctness

Also check that each cited case, statutory section, or article exists in the
provided source and actually supports the claim attached to it. This is
especially important for legal answers. Treat it as a separate citation review
(manual initially, or a specifically designed evaluator); a high faithfulness
or answer-correctness score alone does not guarantee that citations are valid.

## Retrieval Recall@k

Use this to evaluate retrieval independently of answer generation:

```text
Recall@k = relevant reference passages found in the top k results
           / total relevant reference passages for the question
```

For every test question, first label the source passage(s) that are relevant.
Then retrieve the top `k` chunks and compare them with those labels. Without
human-identified relevant passages, Recall@k cannot be measured reliably.

## Run the evaluator

Generate answers from the golden questions first. The answer generator uses
each golden row's `input` and retrieved Qdrant context; it does not send the
golden reference answer to the answer model. It saves answers and their
retrieved contexts to `dataset/generated_answers.json` without changing
`dataset/goldens.json`:

```powershell
uv run python evaluation\generate_answers.py --limit 1
```

The default `--provider auto` tries Groq first and falls back to local Ollama
if Groq is not configured or its request fails. To use Ollama directly:

```powershell
uv run python evaluation\generate_answers.py --provider ollama --limit 1
```

Once answers are saved, run the metrics separately:

```powershell
uv run python evaluation\evaluate.py --limit 1
```

Remove `--limit 1` to run the full dataset. The final metrics are saved in
`evaluation/results/results.json`. Use `--answers` on the evaluator or
`--output` on the generator to select different files.

The scripts are `evaluation/generate_answers.py` and
`evaluation/evaluate.py`. Metric definitions and reusable tools live in this
`metrices` directory:

- `answer_metrics.py`: Contextual Recall, Faithfulness, Answer Relevancy,
  Answer Correctness, and Legal Citation Correctness.
- `retrieval_metrics.py`: labeled passage Recall@k.
- `ollama_judge.py`: local Ollama adapter for DeepEval.

The generator retrieves the top three chunks by default from `legal_cases` and
`citizenship_guides`. Configure `GROQ_API_KEY` and `GROQ_MODEL` for Groq, and
`OLLAMA_MODEL` and `OLLAMA_HOST` for Ollama answer generation or judging in
the root `.env`. If the generator falls back to Ollama, ensure that Ollama is
running and the configured model is installed (for example,
`ollama pull qwen3:4b`). The evaluator reads the saved answer/context pairs
and uses local Ollama as the DeepEval judge.

Generated answers are checkpointed after each question and skipped on reruns
when the question, retrieved chunks, `top_k`, and requested provider are
unchanged. Evaluation results are checkpointed after every example in
`evaluation/results/results.json`; reruns skip unchanged questions, answers,
contexts, references, and provider metadata. Use `--top-k` on
`generate_answers.py` to change the retrieval count, `--output` on that script
to choose the answer file, or `--output` on `evaluate.py` to choose the metrics
file.

The result contains per-example scores and reasons for Contextual Recall,
Faithfulness, Answer Relevancy, Answer Correctness, and Legal Citation
Correctness, as well as the retrieved chunks and generated answer. The command
also prints the mean score for each metric.

Recall@k is calculated only when a golden row contains a human-reviewed
`relevant_contexts` list of passage strings. The current generated `goldens.json`
does not include those labels, so the evaluator will report Recall@k as
unavailable until they are added. Do not treat generated answers as trusted
references until a human has checked their legal accuracy.

Use the scores to find cases for review, not as a substitute for legal review.
