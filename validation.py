"""Reusable validation for generated mini-RAG artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag_pipeline import (
    ANSWER_LABELS,
    RETRIEVAL_STATUSES,
    PipelineError,
    load_documents,
    load_queries,
)


REQUIRED_ARTIFACTS = (
    "chunks.json",
    "retrieval.json",
    "answers.json",
    "eval.json",
)
CITATION_RE = re.compile(r"\[[^\[\]\r\n]+ §[^\[\]\s]+\]")


def _read_json(path: Path, errors: list[str]) -> Any | None:
    if not path.is_file():
        errors.append(f"Missing required artifact: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid JSON artifact {path}: {exc}")
        return None


def _records(payload: Any, key: str, artifact: str, errors: list[str]) -> list[Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        errors.append(f"{artifact} must contain a '{key}' list")
        return []
    return payload[key]


def _ids(records: list[Any], artifact: str, errors: list[str]) -> set[str]:
    identifiers: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
            errors.append(f"{artifact} record {index} has no valid query_id")
            continue
        identifiers.append(record["query_id"])
    duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        errors.append(f"{artifact} contains duplicate query IDs: {duplicates}")
    return set(identifiers)


def validate_artifacts(root: Path) -> list[str]:
    """Return all validation errors without mutating artifacts."""

    errors: list[str] = []
    artifacts_dir = root / "artifacts"
    payloads = {
        name: _read_json(artifacts_dir / name, errors) for name in REQUIRED_ARTIFACTS
    }
    if any(payload is None for payload in payloads.values()):
        return errors

    try:
        queries = load_queries(root / "queries.json")
    except (OSError, ValueError, PipelineError) as exc:
        errors.append(f"Cannot validate input queries: {exc}")
        return errors
    expected_ids = {query.query_id for query in queries}
    queries_by_id = {query.query_id: query for query in queries}
    try:
        documents = load_documents(root / "kb")
    except (OSError, ValueError, PipelineError) as exc:
        errors.append(f"Cannot validate source documents: {exc}")
        return errors
    documents_by_metadata: dict[tuple[str, str], list[str]] = {}
    for document in documents:
        documents_by_metadata.setdefault(
            (document.doc_title, document.section), []
        ).append(document.body)

    chunk_records = _records(payloads["chunks.json"], "chunks", "chunks.json", errors)
    chunk_ids: set[str] = set()
    chunks_by_id: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunk_records):
        if not isinstance(chunk, dict):
            errors.append(f"chunks.json record {index} must be an object")
            continue
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        doc_title = chunk.get("doc_title")
        section = chunk.get("section")
        start = chunk.get("start_char")
        end = chunk.get("end_char")
        required_chunk_fields = {
            "chunk_id",
            "doc_title",
            "section",
            "text",
            "start_char",
            "end_char",
        }
        missing = required_chunk_fields - chunk.keys()
        if missing:
            errors.append(f"chunks.json record {index} is missing {sorted(missing)}")
        if not isinstance(chunk_id, str) or not chunk_id:
            errors.append(f"chunks.json record {index} has no valid chunk_id")
        elif chunk_id in chunk_ids:
            errors.append(f"chunks.json contains duplicate chunk_id {chunk_id!r}")
        else:
            chunk_ids.add(chunk_id)
            chunks_by_id[chunk_id] = chunk
        if not isinstance(doc_title, str) or not isinstance(section, str):
            errors.append(f"chunks.json record {index} has invalid metadata")
        if (
            not isinstance(text, str)
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            errors.append(f"chunks.json record {index} has invalid text/offset fields")
        elif start < 0 or end <= start or len(text) != end - start:
            errors.append(f"chunks.json record {index} has inconsistent offsets")
        elif isinstance(doc_title, str) and isinstance(section, str):
            source_bodies = documents_by_metadata.get((doc_title, section), [])
            if not any(
                end <= len(body) and body[start:end] == text for body in source_bodies
            ):
                errors.append(
                    f"chunks.json record {index} is not an exact source-body substring"
                )

    retrieval_records = _records(
        payloads["retrieval.json"], "queries", "retrieval.json", errors
    )
    retrieval_ids = _ids(retrieval_records, "retrieval.json", errors)
    if retrieval_ids != expected_ids:
        errors.append(
            "retrieval.json query IDs do not exactly match queries.json "
            f"(missing={sorted(expected_ids - retrieval_ids)}, "
            f"extra={sorted(retrieval_ids - expected_ids)})"
        )

    retrieval_by_id: dict[str, dict[str, Any]] = {}
    for record in retrieval_records:
        if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
            continue
        query_id = record["query_id"]
        retrieval_by_id[query_id] = record
        if not isinstance(record.get("question"), str):
            errors.append(f"retrieval query {query_id!r} has no valid question")
        elif query_id in queries_by_id and record["question"] != queries_by_id[query_id].question:
            errors.append(f"retrieval query {query_id!r} question differs from input")
        hits = record.get("top_k")
        if not isinstance(hits, list):
            errors.append(f"retrieval query {query_id!r} has no top_k list")
            continue
        hit_ids: list[str] = []
        for position, hit in enumerate(hits):
            if not isinstance(hit, dict):
                errors.append(f"retrieval query {query_id!r} hit {position} is invalid")
                continue
            chunk_id = hit.get("chunk_id")
            score = hit.get("score")
            required_hit_fields = {
                "rank",
                "chunk_id",
                "doc_title",
                "score",
                "chunk_text",
            }
            missing = required_hit_fields - hit.keys()
            if missing:
                errors.append(
                    f"retrieval query {query_id!r} hit {position} is missing {sorted(missing)}"
                )
            rank = hit.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank != position + 1:
                errors.append(
                    f"retrieval query {query_id!r} hit {position} has invalid rank"
                )
            if not isinstance(hit.get("doc_title"), str) or not isinstance(
                hit.get("chunk_text"), str
            ):
                errors.append(
                    f"retrieval query {query_id!r} hit {position} has invalid text metadata"
                )
            if isinstance(chunk_id, str):
                hit_ids.append(chunk_id)
                if chunk_id not in chunk_ids:
                    errors.append(
                        f"retrieval query {query_id!r} references unknown chunk {chunk_id!r}"
                    )
                else:
                    source_chunk = chunks_by_id[chunk_id]
                    if (
                        hit.get("doc_title") != source_chunk.get("doc_title")
                        or hit.get("chunk_text") != source_chunk.get("text")
                    ):
                        errors.append(
                            f"retrieval query {query_id!r} hit {position} "
                            "does not match chunks.json"
                        )
            else:
                errors.append(
                    f"retrieval query {query_id!r} hit {position} has invalid chunk_id"
                )
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                errors.append(
                    f"retrieval query {query_id!r} hit {position} has non-finite/non-numeric score"
                )
        if len(hit_ids) < 3 or len(set(hit_ids)) < 3:
            errors.append(
                f"retrieval query {query_id!r} must have at least 3 distinct chunks"
            )

    answer_records = _records(
        payloads["answers.json"], "answers", "answers.json", errors
    )
    answer_ids = _ids(answer_records, "answers.json", errors)
    if answer_ids != expected_ids:
        errors.append("answers.json query IDs do not exactly match queries.json")
    for answer in answer_records:
        if not isinstance(answer, dict) or answer.get("query_id") not in retrieval_by_id:
            continue
        query_id = answer["query_id"]
        answer_label = answer.get("answer_label")
        answer_text = answer.get("answer")
        citations = answer.get("citations")
        used_ids = answer.get("used_chunk_ids")
        required_answer_fields = {
            "query_id",
            "answer_label",
            "answer",
            "citations",
            "used_chunk_ids",
        }
        missing = required_answer_fields - answer.keys()
        if missing:
            errors.append(f"answer query {query_id!r} is missing {sorted(missing)}")
        if "label" in answer:
            errors.append(
                f"answer query {query_id!r} uses legacy 'label'; use 'answer_label'"
            )
        if answer_label not in ANSWER_LABELS:
            errors.append(
                f"answer query {query_id!r} has invalid answer_label {answer_label!r}"
            )
        if not isinstance(answer_text, str) or not answer_text:
            errors.append(f"answer query {query_id!r} has invalid answer text")
            answer_text = ""
        if not isinstance(citations, list) or not all(
            isinstance(item, str) for item in citations
        ):
            errors.append(f"answer query {query_id!r} has invalid citations")
            citations = []
        if not isinstance(used_ids, list) or not all(
            isinstance(item, str) for item in used_ids
        ):
            errors.append(f"answer query {query_id!r} has invalid used_chunk_ids")
            used_ids = []
        hits = retrieval_by_id[query_id].get("top_k", [])
        retrieved_ids = {
            hit.get("chunk_id") for hit in hits if isinstance(hit, dict)
        }
        valid_citations = {
            f"[{hit.get('doc_title')} §{hit.get('chunk_id')}]"
            for hit in hits
            if isinstance(hit, dict)
        }
        embedded_citations = CITATION_RE.findall(answer_text)
        if answer_label in {"grounded_answer", "conflicting_context"} and not citations:
            errors.append(f"grounded answer query {query_id!r} has no citations")
        if (
            answer_label in {"grounded_answer", "conflicting_context"}
            and not embedded_citations
        ):
            errors.append(
                f"grounded answer query {query_id!r} has no citation in answer text"
            )
        if answer_label in {"grounded_answer", "conflicting_context"} and not used_ids:
            errors.append(f"grounded answer query {query_id!r} has no used_chunk_ids")
        for citation in citations:
            if citation not in valid_citations:
                errors.append(
                    f"answer query {query_id!r} has citation outside retrieval: {citation!r}"
                )
            if citation not in embedded_citations:
                errors.append(
                    f"answer query {query_id!r} does not include citation {citation!r}"
                )
        for citation in embedded_citations:
            if citation not in valid_citations:
                errors.append(
                    f"answer query {query_id!r} embeds unretrieved citation {citation!r}"
                )
            if citation not in citations:
                errors.append(
                    f"answer query {query_id!r} omits embedded citation from citations list"
                )
        for chunk_id in used_ids:
            if chunk_id not in retrieved_ids:
                errors.append(
                    f"answer query {query_id!r} uses unretrieved chunk {chunk_id!r}"
                )

    evaluation = payloads["eval.json"]
    evaluation_records = _records(evaluation, "queries", "eval.json", errors)
    evaluation_ids = _ids(evaluation_records, "eval.json", errors)
    if evaluation_ids != expected_ids:
        errors.append("eval.json query IDs do not exactly match queries.json")
    statuses: list[str] = []
    for record in evaluation_records:
        if not isinstance(record, dict):
            continue
        query_id = record.get("query_id")
        required_evaluation_fields = {
            "query_id",
            "expected_doc_titles",
            "retrieved_doc_titles_top3",
            "retrieval_status",
            "matched_expected_title",
            "explanation",
        }
        missing = required_evaluation_fields - record.keys()
        if missing:
            errors.append(f"eval query {query_id!r} is missing {sorted(missing)}")
        status = record.get("retrieval_status")
        if status not in RETRIEVAL_STATUSES:
            errors.append(
                f"eval query {record.get('query_id')!r} has invalid status {status!r}"
            )
        else:
            statuses.append(status)
        if query_id not in queries_by_id or query_id not in retrieval_by_id:
            continue
        query = queries_by_id[query_id]
        expected_titles = list(query.expected_doc_titles)
        retrieved_titles = [
            hit.get("doc_title")
            for hit in retrieval_by_id[query_id].get("top_k", [])[:3]
            if isinstance(hit, dict)
        ]
        if record.get("expected_doc_titles") != expected_titles:
            errors.append(f"eval query {query_id!r} expected titles differ from input")
        if record.get("retrieved_doc_titles_top3") != retrieved_titles:
            errors.append(f"eval query {query_id!r} top-3 titles differ from retrieval")

        unique_expected: list[str] = []
        seen_expected: set[str] = set()
        for title in expected_titles:
            normalized = title.casefold()
            if normalized not in seen_expected:
                seen_expected.add(normalized)
                unique_expected.append(title)
        unique_retrieved = {
            title.casefold() for title in retrieved_titles if isinstance(title, str)
        }
        matched = [
            title for title in unique_expected if title.casefold() in unique_retrieved
        ]
        if len(matched) == len(unique_expected):
            expected_status = "hit"
        elif matched:
            expected_status = "partial_hit"
        else:
            expected_status = "miss"
        if status != expected_status:
            errors.append(
                f"eval query {query_id!r} status is {status!r}; expected {expected_status!r}"
            )
        matched_expected_title = record.get("matched_expected_title")
        expected_match = bool(matched)
        if not isinstance(matched_expected_title, bool):
            errors.append(
                f"eval query {query_id!r} matched_expected_title must be boolean"
            )
        elif matched_expected_title != expected_match:
            errors.append(
                f"eval query {query_id!r} has inconsistent matched_expected_title"
            )
        if not isinstance(record.get("explanation"), str) or not record["explanation"]:
            errors.append(f"eval query {query_id!r} has no explanation")

    aggregate = evaluation.get("aggregate") if isinstance(evaluation, dict) else None
    if not isinstance(aggregate, dict):
        errors.append("eval.json has no aggregate summary")
    else:
        counts = Counter(statuses)
        expected_summary = {
            "total_queries": len(evaluation_records),
            "hits": counts["hit"],
            "partial_hits": counts["partial_hit"],
            "misses": counts["miss"],
        }
        for key, expected in expected_summary.items():
            if aggregate.get(key) != expected:
                errors.append(
                    f"eval aggregate {key!r} is {aggregate.get(key)!r}; expected {expected}"
                )
        expected_rate = (
            counts["hit"] / len(evaluation_records) if evaluation_records else 0.0
        )
        rate = aggregate.get("top3_hit_rate")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not math.isclose(rate, expected_rate, abs_tol=1e-6)
        ):
            errors.append(
                f"eval aggregate top3_hit_rate is {rate!r}; expected {expected_rate:.6f}"
            )

    grounding_path = artifacts_dir / "grounding_check.json"
    if grounding_path.exists():
        grounding = _read_json(grounding_path, errors)
        if grounding is not None:
            grounding_records = _records(
                grounding, "queries", "grounding_check.json", errors
            )
            grounding_ids = _ids(grounding_records, "grounding_check.json", errors)
            if grounding_ids != expected_ids:
                errors.append(
                    "grounding_check.json query IDs do not exactly match queries.json"
                )
            for record in grounding_records:
                if isinstance(record, dict) and record.get("passed") is not True:
                    errors.append(
                        f"grounding check failed for query {record.get('query_id')!r}"
                    )

    return errors
