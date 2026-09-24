# Deterministic Mini-RAG

A small, offline RAG-style pipeline for local product knowledge. It parses
`Title:` and `Section:` metadata, creates exact source-preserving chunks,
ranks them with an in-repository BM25 implementation, produces extractive
answers with strict citations, evaluates top-three retrieval, and validates
the resulting artifacts.

## Architecture

- `main.py` orchestrates the state-enforced pipeline.
- `rag_pipeline.py` owns parsing, chunking, BM25 retrieval, answering,
  evaluation, grounding checks, and state transitions.
- `validation.py` contains reusable cross-artifact validation.
- `validate.py` is the standalone validation command.
- `tests/test_pipeline.py` exercises behavior at useful boundaries.

Lexical retrieval was chosen because this replaceable, small knowledge base
benefits from transparent numeric scoring, deterministic replay, and no
network dependency. It uses case folding, conservative plural normalization,
stop-word filtering, and a small explicit term boost for titles and sections.
Ranking ties use stable content-derived chunk IDs.

Chunking uses paragraph boundaries first, then sentence or whitespace
boundaries under an 800-character limit. Every stored chunk is an exact body
substring and its offsets index that body. This favors coherent evidence over
arbitrary slicing, though it is less semantically flexible than embedding-based
chunking.

Answers are deterministic and extractive: sourced sentences are selected only
from retrieved chunks and cited as `[doc_title §chunk_id]`. No LLM is used, so
`llm_calls.jsonl` is intentionally not produced.

## Commands

Requires Python 3.13 and `uv`; there are no third-party runtime dependencies.

```console
uv run python main.py
uv run python validate.py
uv run python -m unittest discover -s tests -v
```

The pipeline reads `kb/*.txt` and `queries.json`, then generates:

- `artifacts/chunks.json`
- `artifacts/retrieval.json`
- `artifacts/answers.json`
- `artifacts/eval.json`
- `artifacts/grounding_check.json`

`queries.json` is a top-level JSON array. Each query has `query_id`, `question`,
and a non-empty `expected_doc_titles` list.

## Limitations and tradeoffs

Lexical ranking cannot infer synonyms that share no terms with the query.
Extractive answers favor auditability over fluent synthesis. Contradiction
detection is deliberately conservative and recognizes only closely matching
claims with opposing negation or incompatible numeric values. The pipeline
fails rather than padding results when the corpus contains fewer than three
chunks.
