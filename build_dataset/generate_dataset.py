"""
generate_dataset.py  --  parse ONCE, generate ONCE, evaluate MANY times.

Pipeline:
    Docling (parse) -> HierarchicalChunker (chunk) -> merge small neighbours
    -> contexts -> DeepEval Synthesizer.generate_goldens_from_contexts()
    -> dataset/silvers.json

Three question families from TWO documents:
    single   questions answerable from one chunk of one doc
    cross    questions that need a chunk from A AND a chunk from B
    summary  "summarize doc A / doc B / both" style overview questions

Cost-saving layers:
    1. Docling parse+chunk is cached per file (cache/*.chunks.json)
    2. Silvers are saved after EVERY batch -> crash/rate-limit loses nothing
    3. Re-running skips finished batches; your eval runs just load the JSON

Install:
    Install project dependencies with `uv sync` and run a local Ollama server.

Usage:
    python generate_dataset.py
    python generate_dataset.py --force         # regenerate silvers (chunk cache is kept)

Set DOC_A and DOC_B in the CONFIG section below to choose the input documents.

Load in your eval script:
    from generate_dataset import load_silvers
    silvers = load_silvers("dataset/silvers.json")
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

load_dotenv()
# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_CONNECT_TIMEOUT = 5
OLLAMA_REQUEST_TIMEOUT = 180
OLLAMA_MAX_OUTPUT_TOKENS = 512
OLLAMA_SCHEMA_RETRIES = 1
DOC_A = Path(__file__).resolve().parents[1] / "docs" / "cases" / "cases.pdf"
DOC_B = Path(__file__).resolve().parents[1] / "docs" / "guides" / "indian_citizenship_guide.pdf"
OUT_PATH = "dataset/silvers.json"
CACHE_DIR = "cache"

# Chunking (HierarchicalChunker emits ONE chunk per document element, so
# chunks are often tiny. We merge neighbours that share the same heading.)
MERGE_MAX_CHARS = 1200       # merge same-section neighbours up to this size; 0 = no merging
MIN_CHUNK_CHARS = 250         # drop fragments shorter than this (page numbers, stray lines)
USE_HEADINGS_IN_CONTEXT = True   # prepend "Heading > Subheading." to each chunk's text

# Counts (tokens spent scale with these)
N_SINGLE_CONTEXTS_PER_DOC = 8
N_CROSS_CONTEXTS = 12
N_SUMMARY_CONTEXTS = 3                           # per doc, and same for "both docs"
SUMMARY_CHUNKS_PER_CONTEXT = 6
SILVERS_PER_CONTEXT = 2
BATCH_SIZE = 3                                   # contexts per synthesizer call = checkpoint size
SEED = 42


class InvalidStructuredOutputError(ValueError):
    """Raised when Ollama cannot produce a valid response for a requested schema."""


# ----------------------------------------------------------------------
# LOCAL OLLAMA LLM
# ----------------------------------------------------------------------
class OllamaLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.ollama_host = OLLAMA_HOST
        self._check_server(model_name)
        super().__init__(model_name=model_name)

    def _check_server(self, model_name: str):
        request = Request(f"{self.ollama_host}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=OLLAMA_CONNECT_TIMEOUT) as response:
                result = json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama health check failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(
                f"Ollama is not running or not reachable at {self.ollama_host}. "
                "Start Ollama and verify OLLAMA_HOST."
            ) from error

        if not isinstance(result, dict) or not isinstance(result.get("models"), list):
            raise ValueError("Ollama /api/tags returned an invalid model list.")
        installed_models = {
            model.get("name")
            for model in result["models"]
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }
        if model_name not in installed_models:
            raise RuntimeError(
                f"Ollama model '{model_name}' is not installed. "
                f"Install it with `ollama pull {model_name}`."
            )

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        schema_json = None
        if schema is not None:
            schema_json = schema.model_json_schema()
        started_at = time.monotonic()
        max_attempts = 1 + (OLLAMA_SCHEMA_RETRIES if schema is not None else 0)
        for attempt in range(max_attempts):
            message = f"/no_think\n{prompt}"
            if schema_json is not None:
                message += (
                    "\n\nReturn one JSON object matching this schema exactly. Use "
                    "the exact property names, include every required property, and "
                    "do not add other properties:\n"
                    f"{json.dumps(schema_json, ensure_ascii=False)}"
                )
                if attempt:
                    message += (
                        "\nYour previous response was incomplete or invalid. "
                        "Return a complete, valid JSON object only."
                    )

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": message}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": OLLAMA_MAX_OUTPUT_TOKENS * (attempt + 1),
                },
                "think": False,
            }
            if schema_json is not None:
                payload["format"] = schema_json

            request = Request(
                f"{self.ollama_host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            print(
                f"  Ollama: generating with {self.model_name}"
                f"{' (retry)' if attempt else ''}...",
                flush=True,
            )
            try:
                with urlopen(request, timeout=OLLAMA_REQUEST_TIMEOUT) as response:
                    result = json.loads(response.read())
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Ollama returned HTTP {error.code}: {detail}"
                ) from error
            except URLError as error:
                raise RuntimeError(
                    f"Could not connect to Ollama at {self.ollama_host}. "
                    "Start Ollama and verify OLLAMA_HOST."
                ) from error
            except TimeoutError as error:
                raise TimeoutError(
                    f"Ollama generation exceeded the "
                    f"{OLLAMA_REQUEST_TIMEOUT}-second request timeout."
                ) from error

            print(
                f"  Ollama: response received in "
                f"{time.monotonic() - started_at:.1f}s",
                flush=True,
            )
            if not isinstance(result, dict):
                raise ValueError("Ollama response was not a JSON object.")
            response_message = result.get("message")
            if not isinstance(response_message, dict):
                raise ValueError("Ollama response did not contain a message object.")
            text = response_message.get("content")
            if not isinstance(text, str):
                raise ValueError("Ollama response message did not contain text content.")
            if not text.strip():
                parse_error = ValueError(
                    "Ollama returned empty content "
                    f"(done_reason={result.get('done_reason', 'unknown')})."
                )
            elif schema_json is None:
                return text
            else:
                text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                try:
                    return schema(**json.loads(text))
                except (json.JSONDecodeError, ValueError) as error:
                    parse_error = error

            if attempt + 1 == max_attempts:
                done_reason = result.get("done_reason", "unknown")
                raise InvalidStructuredOutputError(
                    f"Ollama returned invalid structured output after "
                    f"{max_attempts} attempt(s) (done_reason={done_reason}): "
                    f"{parse_error}"
                ) from parse_error
            print(
                f"  Ollama: structured output was invalid; retrying with a larger "
                f"output limit ({OLLAMA_MAX_OUTPUT_TOKENS * (attempt + 2)} tokens).",
                flush=True,
            )

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


# ----------------------------------------------------------------------
# DOCLING: PARSE + HIERARCHICAL CHUNK (cached)
# ----------------------------------------------------------------------
def _file_key(path: str) -> str:
    """Cache key = file bytes + chunk settings. Change either -> re-parse."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    h.update(f"hierarchical|{MERGE_MAX_CHARS}|{USE_HEADINGS_IN_CONTEXT}".encode())
    return h.hexdigest()[:16]


def _merge_same_section(chunks: list) -> list:
    """Glue consecutive elements under the SAME heading path until MERGE_MAX_CHARS.
    A lone list item or one-line paragraph can't support a good question."""
    merged = []
    for c in chunks:
        if (merged
                and merged[-1]["headings"] == c["headings"]
                and len(merged[-1]["text"]) + len(c["text"]) < MERGE_MAX_CHARS):
            merged[-1]["text"] += " " + c["text"]
        else:
            merged.append(dict(c))
    return merged


def parse_and_chunk(path: str) -> list:
    """Returns [{"text", "headings", "doc"}]. Cached on disk."""
    name = Path(path).stem
    cache_file = Path(CACHE_DIR) / f"{name}.{_file_key(path)}.chunks.json"
    if cache_file.exists():
        print(f"  [{name}] chunks loaded from cache")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    from docling.document_converter import DocumentConverter
    from docling.chunking import HierarchicalChunker

    print(f"  [{name}] parsing with Docling (slow the first time)...")
    doc = DocumentConverter().convert(path).document

    print(f"  [{name}] chunking with HierarchicalChunker...")
    chunks = []
    for ch in HierarchicalChunker().chunk(dl_doc=doc):
        headings = list(getattr(ch.meta, "headings", None) or [])
        text = re.sub(r"\s+", " ", ch.text).strip()
        chunks.append({"text": text, "headings": headings, "doc": name})
    n_raw = len(chunks)

    if MERGE_MAX_CHARS:
        chunks = _merge_same_section(chunks)

    # Add headings AFTER merging so they appear once per chunk, not once per element
    if USE_HEADINGS_IN_CONTEXT:
        for c in chunks:
            if c["headings"]:
                c["text"] = " > ".join(c["headings"]) + ". " + c["text"]

    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]

    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [{name}] {n_raw} raw elements -> {len(chunks)} chunks cached")
    return chunks


# ----------------------------------------------------------------------
# CONTEXT BUILDERS
# A context = {"texts": [str, ...], "sources": [doc, ...]}
# The synthesizer writes questions answerable from ALL texts in a context.
# ----------------------------------------------------------------------
def _ctx(chunks):
    return {"texts": [c["text"] for c in chunks], "sources": sorted({c["doc"] for c in chunks})}


def build_single(chunks, n, rng):
    return [_ctx([c]) for c in rng.sample(chunks, min(n, len(chunks)))]


def _words(s: str) -> set:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def build_cross(chunks_a, chunks_b, n, rng):
    """Pair A/B chunks by word overlap so pairs are genuinely related.
    Random pairs give nonsense 'cross' questions."""
    wa = [_words(c["text"]) for c in chunks_a]
    wb = [_words(c["text"]) for c in chunks_b]
    scored = [(len(sa & sb) / len(sa | sb), i, j)
              for i, sa in enumerate(wa) for j, sb in enumerate(wb) if sa and sb]
    scored.sort(reverse=True)

    out, used_a, used_b = [], set(), set()
    for _, i, j in scored:
        if i in used_a or j in used_b:
            continue
        out.append(_ctx([chunks_a[i], chunks_b[j]]))
        used_a.add(i)
        used_b.add(j)
        if len(out) >= n:
            break
    rng.shuffle(out)
    return out


def _spread(chunks, k, rng, jitter=False):
    """Pick k chunks spread across the doc, preferring one per top-level section."""
    by_section = {}
    for c in chunks:
        by_section.setdefault(tuple(c["headings"][:1]), []).append(c)
    pool = [v[rng.randrange(len(v))] if jitter else v[0] for v in by_section.values()]
    if len(pool) < 2:                                  # no usable headings -> even spacing
        pool = chunks
    if len(pool) <= k:
        return list(pool)
    step = len(pool) / k
    return [pool[min(len(pool) - 1, int(i * step) + (rng.randint(0, 1) if jitter else 0))]
            for i in range(k)]


def build_summary(chunks, n, rng):
    return [_ctx(_spread(chunks, SUMMARY_CHUNKS_PER_CONTEXT, rng, jitter=(i > 0)))
            for i in range(n)]


def build_summary_both(chunks_a, chunks_b, n, rng):
    half = SUMMARY_CHUNKS_PER_CONTEXT // 2
    return [_ctx(_spread(chunks_a, half, rng, jitter=True)
                 + _spread(chunks_b, half, rng, jitter=True)) for _ in range(n)]


# ----------------------------------------------------------------------
# SYNTHESIZERS (different evolution mix + style per family)
# ----------------------------------------------------------------------
def make_synth(model, family: str) -> Synthesizer:
    if family == "single":
        evo = {Evolution.REASONING: 0.25, Evolution.CONCRETIZING: 0.25,
               Evolution.CONSTRAINED: 0.2, Evolution.HYPOTHETICAL: 0.15,
               Evolution.IN_BREADTH: 0.15}
        style = StylingConfig(
            scenario="A user asks detailed questions about a legal/technical document.",
            task="Answer questions strictly from the provided document context.",
            input_format="A clear, self-contained question in one or two sentences.",
            expected_output_format="A concise, factual answer grounded in the context.",
        )
    elif family == "cross":
        evo = {Evolution.COMPARATIVE: 0.4, Evolution.MULTICONTEXT: 0.4, Evolution.REASONING: 0.2}
        style = StylingConfig(
            scenario="A user asks questions that need information from TWO different documents.",
            task="Answer by combining, comparing or contrasting both documents.",
            input_format="A question that cannot be answered from only one of the documents.",
            expected_output_format="An answer that explicitly uses points from both documents.",
        )
    else:  # summary
        evo = {Evolution.IN_BREADTH: 0.5, Evolution.CONCRETIZING: 0.25, Evolution.COMPARATIVE: 0.25}
        style = StylingConfig(
            scenario="A user wants high-level overviews of documents.",
            task="Summarize, outline or give the main themes of the provided material.",
            input_format="An overview request such as 'Summarize...', 'What are the main "
                         "themes of...', 'Give a high-level outline of...'.",
            expected_output_format="A structured summary of 3-6 sentences covering the key points.",
        )
    return Synthesizer(
        model=model,
        evolution_config=EvolutionConfig(num_evolutions=1, evolutions=evo),
        styling_config=style,
    )


# ----------------------------------------------------------------------
# GENERATION WITH CHECKPOINTING
# ----------------------------------------------------------------------
def save(path: str, rows: list):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)                              # atomic write


def run_family(model, family, contexts, rows, done_batches, completed_contexts, path):
    synth = make_synth(model, family)
    src_lookup = {tuple(c["texts"]): c["sources"] for c in contexts}

    for b in range(0, len(contexts), BATCH_SIZE):
        batch_id = f"{family}-{b // BATCH_SIZE}"
        if batch_id in done_batches:
            print(f"  skip {batch_id} (already saved)")
            continue
        batch_end = min(b + BATCH_SIZE, len(contexts))
        print(f"  generating {batch_id} ({batch_end - b} contexts)...")
        for context_index in range(b, batch_end):
            context_key = (batch_id, context_index)
            if context_key in completed_contexts:
                print(f"  skip {batch_id} context {context_index} (already saved)")
                continue

            context = contexts[context_index]
            try:
                silvers = synth.generate_goldens_from_contexts(
                    contexts=[context["texts"]],
                    include_expected_output=True,
                    max_goldens_per_context=SILVERS_PER_CONTEXT,
                )
            except InvalidStructuredOutputError as error:
                print(
                    f"  skip {batch_id} context {context_index}: "
                    f"malformed Ollama response ({error})",
                    flush=True,
                )
                continue

            for s in silvers:
                ctx = getattr(s, "context", None)
                rows.append({
                    "input": s.input,
                    "expected_output": getattr(s, "expected_output", None),
                    "context": ctx,
                    "sources": src_lookup.get(tuple(ctx)) if ctx else None,
                    "family": family,
                    "batch_id": batch_id,
                    "context_index": context_index,
                    "additional_metadata": getattr(s, "additional_metadata", None),
                    "source_file": getattr(s, "source_file", None),
                })
            completed_contexts.add(context_key)
            save(path, rows)                           # checkpoint after each context


def load_silvers(path: str = OUT_PATH):
    """Use in your eval script. Zero generation tokens."""
    from deepeval.dataset import Golden
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Golden(input=r["input"], expected_output=r["expected_output"],
                   context=r["context"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--force", action="store_true", help="regenerate silvers (chunk cache is kept)")
    args = ap.parse_args()

    model = OllamaLLM()
    rng = random.Random(SEED)
    chunks_a = parse_and_chunk(str(DOC_A))
    chunks_b = parse_and_chunk(str(DOC_B))
    print(f"Doc A: {len(chunks_a)} chunks | Doc B: {len(chunks_b)} chunks")

    rows, done, completed_contexts = [], set(), set()
    if Path(args.out).exists() and not args.force:
        rows = json.loads(Path(args.out).read_text(encoding="utf-8"))
        done = {r["batch_id"] for r in rows if "context_index" not in r}
        completed_contexts = {
            (r["batch_id"], r["context_index"])
            for r in rows
            if "context_index" in r
        }
        print(f"Resuming: {len(rows)} silvers, {len(done)} batches already done")

    plan = {
        "single": build_single(chunks_a, N_SINGLE_CONTEXTS_PER_DOC, rng)
                  + build_single(chunks_b, N_SINGLE_CONTEXTS_PER_DOC, rng),
        "cross": build_cross(chunks_a, chunks_b, N_CROSS_CONTEXTS, rng),
        "summary": build_summary(chunks_a, N_SUMMARY_CONTEXTS, rng)
                   + build_summary(chunks_b, N_SUMMARY_CONTEXTS, rng)
                   + build_summary_both(chunks_a, chunks_b, N_SUMMARY_CONTEXTS, rng),
    }

    for family, contexts in plan.items():
        print(f"[{family}] {len(contexts)} contexts")
        run_family(model, family, contexts, rows, done, completed_contexts, args.out)

    print(f"\nDone. {len(rows)} silvers -> {args.out}")
    for fam in plan:
        print(f"  {fam}: {sum(r['family'] == fam for r in rows)}")


if __name__ == "__main__":
    main()