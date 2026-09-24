from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_pipeline import (
    BM25Index,
    Pipeline,
    PipelineError,
    PipelineState,
    Query,
    chunk_documents,
    evaluate_retrieval,
    generate_answers,
    parse_document,
    retrieve_queries,
)
from validation import validate_artifacts


def _write_fixture(root: Path) -> None:
    kb = root / "kb"
    kb.mkdir()
    documents = {
        "unrelated-name-z.txt": (
            "Title: Alpha Product\nSection: Limits\n\n"
            "Alpha transfers have a daily limit of 500 units."
        ),
        "unrelated-name-a.txt": (
            "Title: Beta Product\nSection: Availability\n\n"
            "Beta trading is available every weekend."
        ),
        "third.txt": (
            "Title: Gamma Product\nSection: Fees\n\n"
            "Gamma withdrawals have no service fee."
        ),
        "fourth.txt": (
            "Title: Delta Product\nSection: Security\n\n"
            "Delta accounts support two-factor authentication."
        ),
    }
    for name, contents in documents.items():
        (kb / name).write_text(contents, encoding="utf-8", newline="\n")
    (root / "queries.json").write_text(
        json.dumps(
            [
                {
                    "query_id": "q-alpha",
                    "question": "What is the daily Alpha transfer limit?",
                    "expected_doc_titles": ["Alpha Product"],
                },
                {
                    "query_id": "q-beta",
                    "question": "Is Beta trading available on weekends?",
                    "expected_doc_titles": ["Beta Product"],
                },
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


class ParsingAndRetrievalTests(unittest.TestCase):
    def test_chunks_are_exact_body_substrings_with_offsets(self) -> None:
        source = (
            "Title: Exact Product\r\nSection: Details\r\n\r\n"
            "First sourced sentence.  Second sourced sentence.\r\n\r\n"
            "A final paragraph with unchanged spacing."
        )
        document = parse_document(source)
        chunks = chunk_documents([document], max_chars=80)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(
                document.body[chunk.start_char : chunk.end_char], chunk.text
            )

    def test_chunking_and_ranking_are_deterministic(self) -> None:
        documents = [
            parse_document(
                "Title: Zebra\nSection: Usage\n\nZebra supports weekend trading."
            ),
            parse_document(
                "Title: Alpha\nSection: Usage\n\nAlpha supports weekday trading."
            ),
            parse_document(
                "Title: Gamma\nSection: Usage\n\nGamma supports evening trading."
            ),
        ]
        first_chunks = chunk_documents(documents)
        second_chunks = chunk_documents(reversed(documents))
        self.assertEqual(first_chunks, second_chunks)

        query = Query("q", "Can Zebra trade on weekends?", ("Zebra",))
        first = retrieve_queries(BM25Index(first_chunks), [query])
        second = retrieve_queries(BM25Index(second_chunks), [query])
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]["top_k"]), 3)
        self.assertEqual(first[0]["top_k"][0]["doc_title"], "Zebra")

    def test_answer_uses_retrieved_exact_text_and_citation(self) -> None:
        documents = [
            parse_document(
                "Title: Alpha\nSection: Limits\n\nAlpha has a limit of 500 units."
            ),
            parse_document("Title: Beta\nSection: Info\n\nBeta is available daily."),
            parse_document("Title: Gamma\nSection: Info\n\nGamma has no fee."),
        ]
        chunks = chunk_documents(documents)
        query = Query("q", "What is the Alpha limit?", ("Alpha",))
        retrieval = retrieve_queries(BM25Index(chunks), [query])
        answer = generate_answers([query], retrieval)[0]

        self.assertEqual(answer["answer_label"], "grounded_answer")
        self.assertNotIn("label", answer)
        self.assertTrue(answer["citations"])
        self.assertIn(answer["citations"][0], answer["answer"])
        used = answer["used_chunk_ids"][0]
        hit = next(item for item in retrieval[0]["top_k"] if item["chunk_id"] == used)
        self.assertIn(answer["extracts"][0]["text"], hit["chunk_text"])

    def test_retrieval_rejects_a_corpus_with_fewer_than_three_chunks(self) -> None:
        chunks = chunk_documents(
            [
                parse_document("Title: Alpha\nSection: Info\n\nAlpha content."),
                parse_document("Title: Beta\nSection: Info\n\nBeta content."),
            ]
        )
        with self.assertRaisesRegex(PipelineError, "at least 3 required"):
            BM25Index(chunks).search("Alpha")

    def test_answer_detects_close_explicitly_conflicting_claims(self) -> None:
        documents = [
            parse_document(
                "Title: Alpha Policy One\nSection: Transfers\n\n"
                "Alpha transfers are allowed for verified users."
            ),
            parse_document(
                "Title: Alpha Policy Two\nSection: Transfers\n\n"
                "Alpha transfers are not allowed for verified users."
            ),
            parse_document(
                "Title: Beta Policy\nSection: Transfers\n\n"
                "Beta transfers require a verified user."
            ),
        ]
        query = Query(
            "q", "Are Alpha transfers allowed for verified users?", ("Alpha Policy One",)
        )
        retrieval = retrieve_queries(BM25Index(chunk_documents(documents)), [query])
        answer = generate_answers([query], retrieval)[0]

        self.assertEqual(answer["answer_label"], "conflicting_context")
        self.assertNotIn("label", answer)
        self.assertEqual(len(answer["citations"]), 2)
        self.assertEqual(len(answer["used_chunk_ids"]), 2)


class EvaluationAndStateTests(unittest.TestCase):
    def test_hit_partial_hit_and_miss_semantics(self) -> None:
        queries = [
            Query("hit", "question", ("Alpha",)),
            Query("partial", "question", ("Alpha", "Missing")),
            Query("miss", "question", ("Missing",)),
        ]
        hits = [
            {"rank": 1, "doc_title": "Alpha"},
            {"rank": 2, "doc_title": "Beta"},
            {"rank": 3, "doc_title": "Gamma"},
        ]
        retrieval = [
            {"query_id": query.query_id, "question": query.question, "top_k": hits}
            for query in queries
        ]
        evaluation = evaluate_retrieval(queries, retrieval)

        self.assertEqual(
            [record["retrieval_status"] for record in evaluation["queries"]],
            ["hit", "partial_hit", "miss"],
        )
        matched_flags = [
            record["matched_expected_title"] for record in evaluation["queries"]
        ]
        self.assertEqual(matched_flags, [True, True, False])
        self.assertTrue(all(isinstance(flag, bool) for flag in matched_flags))
        self.assertEqual(evaluation["aggregate"]["hits"], 1)
        self.assertEqual(evaluation["aggregate"]["partial_hits"], 1)
        self.assertEqual(evaluation["aggregate"]["misses"], 1)

    def test_pipeline_rejects_skipped_stage(self) -> None:
        pipeline = Pipeline(Path("unused"))
        with self.assertRaises(PipelineError):
            pipeline.answer()
        self.assertEqual(pipeline.state, PipelineState.INIT)


class EndToEndTests(unittest.TestCase):
    def test_pipeline_regenerates_deterministic_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            first_pipeline = Pipeline(root)
            first_pipeline.run()
            self.assertEqual(first_pipeline.state, PipelineState.RESULTS_FINALISED)
            self.assertEqual(validate_artifacts(root), [])
            generated_answers = json.loads(
                (root / "artifacts" / "answers.json").read_text(encoding="utf-8")
            )["answers"]
            self.assertTrue(generated_answers)
            for answer in generated_answers:
                self.assertIn("answer_label", answer)
                self.assertNotIn("label", answer)

            artifact_names = (
                "chunks.json",
                "retrieval.json",
                "answers.json",
                "eval.json",
                "grounding_check.json",
            )
            first = {
                name: (root / "artifacts" / name).read_bytes()
                for name in artifact_names
            }
            Pipeline(root).run()
            second = {
                name: (root / "artifacts" / name).read_bytes()
                for name in artifact_names
            }
            self.assertEqual(first, second)

    def test_validator_rejects_non_numeric_retrieval_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root)
            Pipeline(root).run()
            path = root / "artifacts" / "retrieval.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["queries"][0]["top_k"][0]["score"] = "not-a-number"
            path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

            errors = validate_artifacts(root)
            self.assertTrue(any("non-finite/non-numeric score" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
