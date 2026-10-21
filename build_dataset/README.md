#Evaluation
(1) data
(2) document parsing with metadata
(3) Chunking
(4) embeddings
        |
    Context:
    filter para 
    like models
    #prompt template
        |
    Silver
        |
    Filter based on various parameters
    {
        - rule based filter
        - llm judge score GEval
        - human spot check
        - decide: keep|drop|need_review
    }
        |
    Golden


# Evaluation dataset generation

[`generate_dataset.py`](./generate_dataset.py) parses two documents, builds contexts, and uses a local Ollama model with DeepEval's synthesizer to generate **silver** question-and-answer examples. These generated examples are candidates for evaluation, not reviewed goldens.

## Configure
- `DOC_A` and `DOC_B`: paths to the two source documents. They currently point to `docs/cases/cases.pdf` and `docs/guides/indian_citizenship_guide.pdf`.
- `OUT_PATH`: default output file, `dataset/silvers.json`.
- `OLLAMA_MODEL`: local model name, default `qwen3:4b`.
- `OLLAMA_HOST`: Ollama API URL, default `http://localhost:11434`.

Make sure Ollama is running and the selected model is installed (`ollama pull qwen3:4b` for the default). Dataset generation does not require a Groq API key.

Install the project dependencies, then run from the repository root:

```powershell
uv sync
uv run python evaluation\generate_dataset.py
```

To choose another output path, pass `--out`. To discard saved silver results and generate from scratch, pass `--force`:

```powershell
uv run python evaluation\generate_dataset.py --out dataset\trial_silvers.json
uv run python evaluation\generate_dataset.py --force
```

## What it generates

The script creates three question families:

- `single`: questions answerable from one document chunk.
- `cross`: questions intended to require related chunks from both documents.
- `summary`: overview questions about either document or both.

Chunk counts, merge behavior, context counts, samples per context, batch size, and random seed are configurable in the same `CONFIG` section.

Each JSON row contains the generated `input`, `expected_output`, source `context`, source document names, question `family`, and `batch_id`. `load_silvers()` converts these rows to DeepEval `Golden` objects for use by DeepEval evaluation APIs; this does not mean the examples have been filtered or human-verified.

## Cache, checkpoints, and resuming

- Parsed chunks are cached under `cache/` as `<document>.<hash>.chunks.json`. The cache key includes the document contents and the merge/headings chunk settings. A complete cache is reused on later runs; an interrupted parse has not yet written a cache for that document.
- Silver rows are checkpointed to the output JSON after each completed context. Malformed structured output is retried once and then skipped; other errors stop the run. Re-run without `--force` to keep successful rows and retry contexts that did not produce a row.
- `--force` starts with an empty silver list but keeps parsed chunk caches. The existing output file is replaced when the first new batch is saved; if generation fails before that save, the old file remains.
- Resume identifies completed work by batch IDs and context indexes, not by comparing the contexts or configuration. If you change the input documents, random seed, context/count settings, chunk settings, or batch size, use `--force` (or choose a new `--out`) to avoid mixing results from different runs.

The cache and default output paths are relative to the current working directory, so run the command from the repository root as shown above.

## Count silvers by quality score

[`helper.py`](./helper.py) lists the number of silver examples for every quality score:

```powershell
uv run python evaluation\helper.py
```

To use another dataset file, pass `--dataset`.

To print every question together with its family and quality score, use `--list`:

```powershell
uv run python evaluation\helper.py --list
```

## Create a quality-filtered golden dataset

Use `goldens.py` with a minimum quality score. It copies silver rows whose `additional_metadata.synthetic_input_quality` is greater than or equal to the chosen score into `dataset/goldens.json`; it does not modify `silvers.json`.

```powershell
uv run python evaluation\goldens.py 0.8
```

Optionally choose another source or output path with `--input` and `--output`. This is a quality-score filter only; review the selected examples before treating them as verified goldens.

## Judge silvers with GEval

Run two local Ollama GEval checks for every silver: question clarity/answerability and whether its expected answer is supported by its context:

```powershell
uv run python evaluation\judge_silvers.py
```

Scores and reasons are saved separately to `dataset/silver_judgments.json`; the source silver dataset is not modified. The judge checkpoints each silver and skips unchanged rows on a later run. These scores are for review only and do not automatically keep or discard examples. Use `--output` to choose another results file.

## Next step: review silver examples

Generated silvers should be filtered before being treated as goldens. The intended evaluation workflow is:

1. Apply rule-based filters.
2. Score examples with an LLM judge (for example, GEval).
3. Perform a human spot check.
4. Mark each example `keep`, `drop`, or `need_review`.
5. Promote only accepted examples to the golden evaluation dataset.



#Done. 74 silvers -> dataset/silvers.json
  single: 32
  cross: 24
  summary: 18