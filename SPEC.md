# Mini-RAG Pipeline Specification

## Objective

Build a deterministic, replayable local RAG-style pipeline that:

- ingests local product knowledge documents;
- parses `Title` and `Section` metadata;
- chunks documents deterministically;
- indexes and ranks chunks using code-owned retrieval logic;
- generates citation-strict grounded answers;
- evaluates retrieval deterministically;
- validates generated artifacts; and
- can regenerate everything from a clean checkout.

The core pipeline is offline and reproducible: the same inputs and code must produce the same ordered results and artifacts.

## Inputs

- `kb/*.txt`: local plain-text knowledge documents.
- `queries.json`: queries and expected document titles, following the supplied schema.

Document discovery and processing must be independent of fixture filenames, filesystem enumeration order, document order, and sample wording. Input errors such as missing metadata or malformed query records should fail clearly.

## Outputs

Required artifacts:

- `artifacts/chunks.json`
- `artifacts/retrieval.json`
- `artifacts/answers.json`
- `artifacts/eval.json`

Planned SHOULD-attempt artifact:

- `artifacts/grounding_check.json`

`llm_calls.jsonl` is not required because the planned core design does not use an LLM.

## Pipeline State Machine

The pipeline must enforce this exact order:

```text
INIT
→ DOCUMENTS_LOADED
→ DOCUMENTS_CHUNKED
→ INDEX_BUILT
→ RETRIEVAL_COMPLETE
→ ANSWERS_GENERATED
→ EVALUATION_COMPLETE
→ VALIDATION_COMPLETE
→ RESULTS_FINALISED
```

Each operation must require its preceding state and advance only to the next state. In particular, answers must never be generated before retrieval completes.

## Document Parsing and Chunking

The loader discovers all `.txt` files in `kb/`, parses line-based `Title:` and `Section:` headers, and treats the remaining content as the document body. Chunk text must be preserved exactly as sourced from that body: chunking must not normalize, summarize, or semantically rewrite it.

Chunk boundaries will be selected deterministically at paragraph or sentence boundaries under a conservative maximum size. Oversized units may be split at deterministic whitespace boundaries. Sentence/paragraph-aware chunks preserve coherent facts better than arbitrary byte or character slices while remaining deterministic.

Each chunk records:

- `chunk_id`
- `doc_title`
- `section`
- `text`
- `start_char`
- `end_char`

Offsets refer to the original document body and must satisfy `body[start_char:end_char] == text`. IDs will be derived reproducibly from stable metadata, body identity, and source offsets rather than discovery order or filename.

## Retrieval / Ranking

Retrieval uses local deterministic lexical ranking implemented in repository code, preferably BM25-style scoring. The implementation will tokenize and normalize query and chunk text, compute corpus statistics, assign numeric scores, and sort results using a stable final tie-breaker such as `chunk_id`.

At least the top three distinct chunks are returned for every query. An input corpus unable to supply three chunks should produce a clear failure rather than padded or duplicated results. Retrieval and ranking must not use an LLM, sample-specific keywords, or hardcoded expected answers.

Lexical retrieval fits a small, replaceable product knowledge base because it is transparent, deterministic, offline, easy to inspect, and straightforward to evaluate and replay.

## Answer Generation

The core submission uses deterministic extractive grounded answering. It inspects only retrieved chunks and selects supporting sourced sentences from that context. It must never introduce facts absent from retrieved text.

Every factual grounded answer must contain at least one strict citation in this form:

```text
[doc_title §chunk_id]
```

Citations may refer only to chunks retrieved for the same query. Answers use only these labels:

- `grounded_answer`
- `insufficient_context`
- `conflicting_context`

The controlled value is stored in the `answer_label` field. If sufficient support cannot be identified, the result is `insufficient_context`. If retrieved evidence contains materially contradictory supported claims, the result may be `conflicting_context`, with citations to the competing evidence.

Avoiding an LLM keeps the pipeline fully offline and replayable and removes API nondeterminism. Therefore, no `llm_calls.jsonl` is needed unless this design changes later.

## Deterministic Evaluation

For each query, evaluation compares the expected document titles from `queries.json` with the document titles represented by the retrieved top three chunks. It records matched titles, their ranks, and a deterministic explanation, then uses only:

- `hit`: all expected titles are represented in the top 3;
- `partial_hit`: at least 1, but not all expected titles are represented in top 3; or
- `miss`: none of the expected titles are represented.

Formally, if M = expected_titles ∩ retrieved_top3_titles:

hit if |M| == |expected_titles|
partial_hit if 0 < |M| < |expected_titles|
miss if |M| == 0

The aggregate contains:

- `total_queries`
- `hits`
- `partial_hits`
- `misses`
- `top3_hit_rate`

`top3_hit_rate = hits / total_queries`; for an empty query set it is defined as `0.0`. Counts and rate are calculated directly from the per-query records in stable query order.

## Grounding Validation

As a planned SHOULD-attempt, deterministic grounding validation will:

- verify every answer citation exists in retrieval results for the same query;
- verify every `used_chunk_id` was retrieved;
- measure reproducible token overlap/support between answer wording and cited chunk text; and
- write the per-query outcome and summary to `artifacts/grounding_check.json`.

This check supplements, but does not replace, strict citation and source-substring checks for extractive answers.

## Validation Command

The validation entry point will be:

```text
uv run python validate.py
```

It must check:

- required artifact existence and valid JSON;
- every input query was processed;
- at least three distinct retrieved chunks per query;
- all retrieval scores are numeric and finite;
- only controlled answer labels are used;
- grounded answers contain citations;
- citations and `used_chunk_id` values refer only to retrieved chunks;
- only controlled retrieval statuses are used; and
- the aggregate evaluation summary exists and agrees with per-query results.

Validation must return a non-zero exit status with useful diagnostics for malformed or inconsistent output.

## Testing Strategy

Behavioral tests take priority and cover:

- document parsing, exact chunk preservation, and body offsets;
- deterministic ranking and top-three retrieval;
- answer extraction and citation integrity;
- `hit`, `partial_hit`, and `miss` evaluation;
- pipeline state-order enforcement;
- validation failures for malformed or inconsistent artifacts; and
- end-to-end regeneration from temporary fixture inputs.

Tests must be deterministic, use replaceable fixtures, and require no network or API access. A repeat run over unchanged inputs should produce equivalent serialized artifacts.

## Assumptions / Non-Goals

- No database, vector service, or external proprietary data is needed.
- Answers and sample-specific behavior will not be hardcoded.
- Processing will not rely on exact fixture filenames, order, or wording.
- An API or stretch endpoint is out of scope until core requirements are complete.
- Generated artifacts may be deleted and must be reproducible.
- Code quality favors small, understandable modules over speculative abstractions.
