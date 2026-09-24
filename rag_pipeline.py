"""Deterministic ingestion, retrieval, answering, and evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


ANSWER_LABELS = frozenset(
    {"grounded_answer", "insufficient_context", "conflicting_context"}
)
RETRIEVAL_STATUSES = frozenset({"hit", "partial_hit", "miss"})
TOP_K = 3
MAX_CHUNK_CHARS = 800

TOKEN_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)|\r?\n(?=\s*\r?\n)|\Z")
NEGATIONS = frozenset(
    {"no", "not", "never", "cannot", "can't", "isn't", "doesn't", "won't"}
)
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "after",
        "ago",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "than",
        "that",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
    }
)


class PipelineError(RuntimeError):
    """Raised when pipeline input or state is invalid."""


class PipelineState(str, Enum):
    INIT = "INIT"
    DOCUMENTS_LOADED = "DOCUMENTS_LOADED"
    DOCUMENTS_CHUNKED = "DOCUMENTS_CHUNKED"
    INDEX_BUILT = "INDEX_BUILT"
    RETRIEVAL_COMPLETE = "RETRIEVAL_COMPLETE"
    ANSWERS_GENERATED = "ANSWERS_GENERATED"
    EVALUATION_COMPLETE = "EVALUATION_COMPLETE"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"
    RESULTS_FINALISED = "RESULTS_FINALISED"


@dataclass(frozen=True)
class Document:
    doc_title: str
    section: str
    body: str
    source_digest: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_title: str
    section: str
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Query:
    query_id: str
    question: str
    expected_doc_titles: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Return deterministic case-folded tokens with conservative plurals folded."""

    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).casefold()
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif (
            len(token) > 4
            and token.endswith("s")
            and not token.endswith(("ss", "is", "us"))
        ):
            token = token[:-1]
        tokens.append(token)
    return tokens


def lexical_terms(text: str) -> list[str]:
    """Return content-bearing tokens used by ranking."""

    return [token for token in tokenize(text) if token not in STOP_WORDS]


def parse_document(source: str, source_name: str = "<document>") -> Document:
    """Parse leading Title/Section headers and retain the remaining body exactly."""

    title: str | None = None
    section: str | None = None
    body_start: int | None = None
    offset = 0

    for line in source.splitlines(keepends=True):
        header_candidate = line.rstrip("\r\n").strip().lstrip("\ufeff")
        if not header_candidate and (title is None or section is None):
            offset += len(line)
            continue
        if title is None and header_candidate.startswith("Title:"):
            title = header_candidate.removeprefix("Title:").strip()
        elif section is None and header_candidate.startswith("Section:"):
            section = header_candidate.removeprefix("Section:").strip()
        else:
            body_start = offset
            break
        offset += len(line)
    else:
        body_start = offset

    if not title:
        raise PipelineError(f"{source_name}: missing or empty Title header")
    if not section:
        raise PipelineError(f"{source_name}: missing or empty Section header")

    assert body_start is not None
    body = source[body_start:]
    if not body.strip():
        raise PipelineError(f"{source_name}: document body is empty")

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return Document(title, section, body, digest)


def load_documents(kb_dir: Path) -> list[Document]:
    """Load all top-level KB text files, then order by content-derived keys."""

    if not kb_dir.is_dir():
        raise PipelineError(f"Knowledge-base directory does not exist: {kb_dir}")

    paths = list(kb_dir.glob("*.txt"))
    if not paths:
        raise PipelineError(f"No .txt documents found in {kb_dir}")

    documents: list[Document] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            documents.append(parse_document(handle.read(), str(path)))

    documents.sort(
        key=lambda doc: (
            doc.doc_title.casefold(),
            doc.section.casefold(),
            doc.source_digest,
        )
    )
    identities = [
        (doc.doc_title.casefold(), doc.section.casefold(), doc.source_digest)
        for doc in documents
    ]
    if len(identities) != len(set(identities)):
        raise PipelineError("Duplicate documents have identical title, section, and body")
    return documents


def _choose_chunk_end(body: str, start: int, maximum: int) -> int:
    hard_end = min(start + maximum, len(body))
    if hard_end == len(body):
        return hard_end

    minimum = start + maximum // 2
    window = body[start:hard_end]

    paragraph_ends = [
        start + match.end()
        for match in re.finditer(r"\r?\n[ \t]*\r?\n", window)
        if start + match.end() >= minimum
    ]
    if paragraph_ends:
        return paragraph_ends[-1]

    sentence_ends = [
        start + match.end()
        for match in re.finditer(r"[.!?][\"')\]]*(?:[ \t]+|\r?\n+)", window)
        if start + match.end() >= minimum
    ]
    if sentence_ends:
        return sentence_ends[-1]

    whitespace = [
        start + match.end()
        for match in re.finditer(r"\s+", window)
        if start + match.end() >= minimum
    ]
    return whitespace[-1] if whitespace else hard_end


def chunk_documents(
    documents: Iterable[Document], max_chars: int = MAX_CHUNK_CHARS
) -> list[Chunk]:
    """Create exact source substrings using deterministic coherent boundaries."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    chunks: list[Chunk] = []
    for document in documents:
        start = 0
        while start < len(document.body):
            end = _choose_chunk_end(document.body, start, max_chars)
            if end <= start:
                raise PipelineError("Chunker failed to advance")
            text = document.body[start:end]
            if text.strip():
                identity = "\0".join(
                    (
                        document.doc_title,
                        document.section,
                        document.source_digest,
                        str(start),
                        str(end),
                    )
                )
                chunk_id = "chunk_" + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_title=document.doc_title,
                        section=document.section,
                        text=text,
                        start_char=start,
                        end_char=end,
                    )
                )
            start = end

    chunks.sort(
        key=lambda chunk: (
            chunk.doc_title.casefold(),
            chunk.section.casefold(),
            chunk.start_char,
            chunk.chunk_id,
        )
    )
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise PipelineError("Chunk ID collision detected")
    return chunks


def load_queries(path: Path) -> list[Query]:
    """Read the required top-level JSON array query schema."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineError(f"Query file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(payload, list):
        raise PipelineError("queries.json must be a top-level JSON array")
    records = payload

    queries: list[Query] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PipelineError(f"Query at index {index} must be an object")
        query_id = record.get("query_id", record.get("id"))
        question = record.get("question", record.get("query"))
        expected = record.get("expected_doc_titles")
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(query_id, str) or not query_id.strip():
            raise PipelineError(f"Query at index {index} has no valid query_id")
        if not isinstance(question, str) or not question.strip():
            raise PipelineError(f"Query {query_id!r} has no valid question")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(item, str) and item.strip() for item in expected)
        ):
            raise PipelineError(
                f"Query {query_id!r} must have non-empty expected_doc_titles"
            )
        queries.append(
            Query(
                query_id=query_id.strip(),
                question=question.strip(),
                expected_doc_titles=tuple(item.strip() for item in expected),
            )
        )

    queries.sort(key=lambda query: query.query_id)
    if len({query.query_id for query in queries}) != len(queries):
        raise PipelineError("Query IDs must be unique")
    return queries


class BM25Index:
    """Small deterministic BM25 index with explicit metadata boosting."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise PipelineError("Cannot build an index without chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.lengths: list[int] = []

        for chunk in chunks:
            weighted_tokens = (
                lexical_terms(chunk.text)
                + lexical_terms(chunk.doc_title) * 2
                + lexical_terms(chunk.section) * 2
            )
            frequencies = Counter(weighted_tokens)
            self.term_frequencies.append(frequencies)
            self.document_frequencies.update(frequencies.keys())
            self.lengths.append(len(weighted_tokens))
        self.average_length = sum(self.lengths) / len(self.lengths)

    def score(self, question: str, index: int) -> float:
        query_frequencies = Counter(lexical_terms(question))
        frequencies = self.term_frequencies[index]
        length = self.lengths[index]
        corpus_size = len(self.chunks)
        total = 0.0
        for term, query_count in query_frequencies.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self.document_frequencies[term]
            inverse_frequency = math.log(
                1.0
                + (corpus_size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * length / self.average_length
            )
            total += (
                inverse_frequency
                * frequency
                * (self.k1 + 1.0)
                / denominator
                * query_count
            )
        return total

    def search(self, question: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        if len(self.chunks) < top_k:
            raise PipelineError(
                f"Corpus contains {len(self.chunks)} chunks; at least {top_k} required"
            )
        scored = [
            (self.score(question, index), chunk)
            for index, chunk in enumerate(self.chunks)
        ]
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].doc_title.casefold(),
                item[1].section.casefold(),
                item[1].chunk_id,
            )
        )
        return [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "doc_title": chunk.doc_title,
                "section": chunk.section,
                "score": round(score, 12),
                "chunk_text": chunk.text,
            }
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def retrieve_queries(
    index: BM25Index, queries: Iterable[Query], top_k: int = TOP_K
) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query.query_id,
            "question": query.question,
            "top_k": index.search(query.question, top_k),
        }
        for query in queries
    ]


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()
        if candidate:
            sentences.append(candidate)
        start = end
    return sentences


def _content_terms(text: str) -> set[str]:
    terms = set(tokenize(text))
    content = terms - STOP_WORDS
    return content or terms


def _numeric_terms(text: str) -> set[str]:
    return {
        token for token in tokenize(text) if any(character.isdigit() for character in token)
    }


def _claims_conflict(first: str, second: str, query_terms: set[str]) -> bool:
    first_tokens = set(tokenize(first))
    second_tokens = set(tokenize(second))
    if not (query_terms & first_tokens & second_tokens):
        return False

    ignored = STOP_WORDS | NEGATIONS | _numeric_terms(first) | _numeric_terms(second)
    first_base = first_tokens - ignored
    second_base = second_tokens - ignored
    minimum_size = min(len(first_base), len(second_base))
    if not minimum_size:
        return False
    similarity = len(first_base & second_base) / minimum_size
    if similarity < 0.6:
        return False

    negation_differs = bool(first_tokens & NEGATIONS) != bool(second_tokens & NEGATIONS)
    first_numbers = _numeric_terms(first)
    second_numbers = _numeric_terms(second)
    values_differ = bool(
        first_numbers and second_numbers and first_numbers.isdisjoint(second_numbers)
    )
    return negation_differs or values_differ


def generate_answers(
    queries: Iterable[Query], retrieval: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build answers solely from verbatim sentences in retrieved chunks."""

    retrieval_by_id = {record["query_id"]: record for record in retrieval}
    answers: list[dict[str, Any]] = []

    for query in queries:
        result = retrieval_by_id[query.query_id]
        query_terms = _content_terms(query.question)
        candidates: list[dict[str, Any]] = []
        for hit in result["top_k"]:
            metadata_terms = _content_terms(hit["doc_title"] + " " + hit["section"])
            for sentence in _sentences(hit["chunk_text"]):
                evidence_terms = _content_terms(sentence) | metadata_terms
                overlap = query_terms & evidence_terms
                if not overlap:
                    continue
                candidates.append(
                    {
                        "sentence": sentence,
                        "chunk_id": hit["chunk_id"],
                        "doc_title": hit["doc_title"],
                        "overlap": len(overlap),
                        "coverage": len(overlap) / len(query_terms),
                        "retrieval_score": hit["score"],
                        "rank": hit["rank"],
                    }
                )

        candidates.sort(
            key=lambda item: (
                -item["overlap"],
                -item["coverage"],
                -item["retrieval_score"],
                item["rank"],
                item["chunk_id"],
                item["sentence"],
            )
        )

        if not candidates or candidates[0]["retrieval_score"] <= 0:
            answers.append(
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "answer_label": "insufficient_context",
                    "answer": "Insufficient context in the retrieved chunks.",
                    "citations": [],
                    "used_chunk_ids": [],
                    "extracts": [],
                }
            )
            continue

        selected = [candidates[0]]
        for candidate in candidates[1:6]:
            if candidate["chunk_id"] == candidates[0]["chunk_id"]:
                continue
            if _claims_conflict(
                candidates[0]["sentence"], candidate["sentence"], query_terms
            ):
                selected.append(candidate)
                break

        answer_label = (
            "conflicting_context" if len(selected) == 2 else "grounded_answer"
        )
        citations = [
            f"[{item['doc_title']} §{item['chunk_id']}]" for item in selected
        ]
        answer_text = " ".join(
            f"{item['sentence']} {citation}"
            for item, citation in zip(selected, citations, strict=True)
        )
        answers.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "answer_label": answer_label,
                "answer": answer_text,
                "citations": citations,
                "used_chunk_ids": [item["chunk_id"] for item in selected],
                "extracts": [
                    {"chunk_id": item["chunk_id"], "text": item["sentence"]}
                    for item in selected
                ],
            }
        )
    return answers


def evaluate_retrieval(
    queries: Iterable[Query], retrieval: list[dict[str, Any]]
) -> dict[str, Any]:
    retrieval_by_id = {record["query_id"]: record for record in retrieval}
    records: list[dict[str, Any]] = []

    for query in queries:
        top_three = retrieval_by_id[query.query_id]["top_k"][:3]
        retrieved_titles = [hit["doc_title"] for hit in top_three]
        title_ranks: dict[str, int] = {}
        for hit in top_three:
            title_ranks.setdefault(hit["doc_title"].casefold(), hit["rank"])
        unique_expected: list[str] = []
        seen_expected: set[str] = set()
        for title in query.expected_doc_titles:
            normalized = title.casefold()
            if normalized not in seen_expected:
                seen_expected.add(normalized)
                unique_expected.append(title)
        matched = [
            title
            for title in unique_expected
            if title.casefold() in title_ranks
        ]
        if len(matched) == len(unique_expected):
            status = "hit"
        elif matched:
            status = "partial_hit"
        else:
            status = "miss"

        if matched:
            rank_text = ", ".join(
                f"{title}={title_ranks[title.casefold()]}" for title in matched
            )
            explanation = (
                f"Matched {len(matched)} of {len(unique_expected)} "
                f"expected title(s) in the top 3 at rank(s): {rank_text}."
            )
        else:
            explanation = "No expected document title appeared in the top 3."

        records.append(
            {
                "query_id": query.query_id,
                "expected_doc_titles": list(query.expected_doc_titles),
                "retrieved_doc_titles_top3": retrieved_titles,
                "retrieval_status": status,
                "matched_expected_title": bool(matched),
                "matched_expected_titles": matched,
                "explanation": explanation,
            }
        )

    counts = Counter(record["retrieval_status"] for record in records)
    total = len(records)
    aggregate = {
        "top3_hit_rate": round(counts["hit"] / total, 6) if total else 0.0,
        "total_queries": total,
        "hits": counts["hit"],
        "partial_hits": counts["partial_hit"],
        "misses": counts["miss"],
    }
    return {"queries": records, "aggregate": aggregate}


def check_grounding(
    answers: list[dict[str, Any]], retrieval: list[dict[str, Any]]
) -> dict[str, Any]:
    retrieval_by_id = {record["query_id"]: record for record in retrieval}
    records: list[dict[str, Any]] = []

    for answer in answers:
        hits = retrieval_by_id[answer["query_id"]]["top_k"]
        chunks = {hit["chunk_id"]: hit for hit in hits}
        expected_citations = {
            f"[{hit['doc_title']} §{hit['chunk_id']}]" for hit in hits
        }
        citations_valid = all(
            citation in expected_citations for citation in answer["citations"]
        )
        used_ids_valid = all(
            chunk_id in chunks for chunk_id in answer["used_chunk_ids"]
        )

        support_checks: list[dict[str, Any]] = []
        for extract in answer.get("extracts", []):
            chunk = chunks.get(extract.get("chunk_id"))
            exact_match = bool(chunk and extract.get("text") in chunk["chunk_text"])
            extract_terms = set(tokenize(str(extract.get("text", ""))))
            chunk_terms = set(tokenize(chunk["chunk_text"])) if chunk else set()
            overlap = (
                len(extract_terms & chunk_terms) / len(extract_terms)
                if extract_terms
                else 0.0
            )
            support_checks.append(
                {
                    "chunk_id": extract.get("chunk_id"),
                    "exact_substring": exact_match,
                    "token_overlap": round(overlap, 6),
                    "supported": exact_match or overlap >= 0.9,
                }
            )

        evidence_required = answer["answer_label"] in {
            "grounded_answer",
            "conflicting_context",
        }
        evidence_supported = bool(support_checks) and all(
            check["supported"] for check in support_checks
        )
        passed = (
            citations_valid
            and used_ids_valid
            and (evidence_supported if evidence_required else True)
        )
        records.append(
            {
                "query_id": answer["query_id"],
                "status": "supported" if passed else "unsupported",
                "citations_retrieved": citations_valid,
                "used_chunk_ids_retrieved": used_ids_valid,
                "support_checks": support_checks,
                "passed": passed,
            }
        )

    passed_count = sum(record["passed"] for record in records)
    return {
        "queries": records,
        "aggregate": {
            "total_queries": len(records),
            "passed": passed_count,
            "failed": len(records) - passed_count,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(serialized, encoding="utf-8", newline="\n")


class Pipeline:
    """State-enforced orchestration for the complete artifact pipeline."""

    def __init__(self, root: Path):
        self.root = root
        self.artifacts_dir = root / "artifacts"
        self.state = PipelineState.INIT
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.queries: list[Query] = []
        self.index: BM25Index | None = None
        self.retrieval: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []
        self.evaluation: dict[str, Any] = {}

    def _advance(self, expected: PipelineState, next_state: PipelineState) -> None:
        if self.state is not expected:
            raise PipelineError(
                f"Invalid pipeline transition: {self.state.value} -> {next_state.value}; "
                f"expected current state {expected.value}"
            )
        self.state = next_state

    def load(self) -> None:
        if self.state is not PipelineState.INIT:
            self._advance(PipelineState.INIT, PipelineState.DOCUMENTS_LOADED)
            return
        self.documents = load_documents(self.root / "kb")
        self.queries = load_queries(self.root / "queries.json")
        self._advance(PipelineState.INIT, PipelineState.DOCUMENTS_LOADED)

    def chunk(self) -> None:
        if self.state is not PipelineState.DOCUMENTS_LOADED:
            self._advance(PipelineState.DOCUMENTS_LOADED, PipelineState.DOCUMENTS_CHUNKED)
            return
        self.chunks = chunk_documents(self.documents)
        write_json(
            self.artifacts_dir / "chunks.json",
            {"chunks": [asdict(chunk) for chunk in self.chunks]},
        )
        self._advance(PipelineState.DOCUMENTS_LOADED, PipelineState.DOCUMENTS_CHUNKED)

    def build_index(self) -> None:
        if self.state is not PipelineState.DOCUMENTS_CHUNKED:
            self._advance(PipelineState.DOCUMENTS_CHUNKED, PipelineState.INDEX_BUILT)
            return
        self.index = BM25Index(self.chunks)
        self._advance(PipelineState.DOCUMENTS_CHUNKED, PipelineState.INDEX_BUILT)

    def retrieve(self) -> None:
        if self.state is not PipelineState.INDEX_BUILT:
            self._advance(PipelineState.INDEX_BUILT, PipelineState.RETRIEVAL_COMPLETE)
            return
        assert self.index is not None
        self.retrieval = retrieve_queries(self.index, self.queries)
        write_json(self.artifacts_dir / "retrieval.json", {"queries": self.retrieval})
        self._advance(PipelineState.INDEX_BUILT, PipelineState.RETRIEVAL_COMPLETE)

    def answer(self) -> None:
        if self.state is not PipelineState.RETRIEVAL_COMPLETE:
            self._advance(
                PipelineState.RETRIEVAL_COMPLETE, PipelineState.ANSWERS_GENERATED
            )
            return
        self.answers = generate_answers(self.queries, self.retrieval)
        write_json(self.artifacts_dir / "answers.json", {"answers": self.answers})
        self._advance(
            PipelineState.RETRIEVAL_COMPLETE, PipelineState.ANSWERS_GENERATED
        )

    def evaluate(self) -> None:
        if self.state is not PipelineState.ANSWERS_GENERATED:
            self._advance(
                PipelineState.ANSWERS_GENERATED, PipelineState.EVALUATION_COMPLETE
            )
            return
        self.evaluation = evaluate_retrieval(self.queries, self.retrieval)
        write_json(self.artifacts_dir / "eval.json", self.evaluation)
        self._advance(
            PipelineState.ANSWERS_GENERATED, PipelineState.EVALUATION_COMPLETE
        )

    def validate(self) -> None:
        if self.state is not PipelineState.EVALUATION_COMPLETE:
            self._advance(
                PipelineState.EVALUATION_COMPLETE, PipelineState.VALIDATION_COMPLETE
            )
            return
        grounding = check_grounding(self.answers, self.retrieval)
        write_json(self.artifacts_dir / "grounding_check.json", grounding)

        from validation import validate_artifacts

        errors = validate_artifacts(self.root)
        if errors:
            raise PipelineError("Artifact validation failed:\n- " + "\n- ".join(errors))
        self._advance(
            PipelineState.EVALUATION_COMPLETE, PipelineState.VALIDATION_COMPLETE
        )

    def finalise(self) -> None:
        self._advance(
            PipelineState.VALIDATION_COMPLETE, PipelineState.RESULTS_FINALISED
        )

    def run(self) -> None:
        self.load()
        self.chunk()
        self.build_index()
        self.retrieve()
        self.answer()
        self.evaluate()
        self.validate()
        self.finalise()
