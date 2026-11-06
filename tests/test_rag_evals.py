import json
from pathlib import Path

import pytest

from bunge_watch.rag.evals import (
    GoldenQuestion,
    QuestionResult,
    compute_metrics,
    first_relevant_rank,
    load_fixture_corpus,
    load_questions,
    run_eval,
)
from bunge_watch.rag.embeddings import HashingEmbedder
from bunge_watch.rag.index import Retrieved

ROOT = Path(__file__).parent.parent
Q = GoldenQuestion(id="q", question="?", language="en", bill_id=5, expected_clauses=["Section 3"],
                   reference_answer="", answerable=True)


def chunk(bill_id, label):
    return Retrieved(0, bill_id, label, None, 0, 0, "", 0.0)


def test_first_relevant_rank_matches_bill_and_normalized_clause():
    assert first_relevant_rank(Q, [chunk(5, "Clause 1"), chunk(6, "Clause 3"), chunk(5, "Clause 3")]) == 3
    assert first_relevant_rank(Q, [chunk(5, "Clause 1")]) is None


def test_metric_calculations():
    results = [
        QuestionResult("a", "en", True, [], rank=1, status="answered",
                       citations=[{"clause": "Clause 3"}, {"clause": "Clause 9"}], citations_on_target=1,
                       judge={"faithfulness": "faithful", "correctness": "correct"}),
        QuestionResult("b", "sw", True, [], rank=4, status="answered", citations=[{"clause": "Clause 2"}],
                       citations_on_target=1, rejected_citations=1,
                       judge={"faithfulness": "partially_faithful", "correctness": "incorrect"}),
        QuestionResult("c", "en", True, [], rank=None, status="not_found"),
        QuestionResult("d", "en", False, [], status="not_found"),
        QuestionResult("e", "sw", False, [], status="answered", citations=[{"clause": "Clause 1"}]),
    ]
    m = compute_metrics(results, k=5)
    assert m["recall@5"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr"] == pytest.approx((1 + 0.25 + 0) / 3, abs=1e-4)
    assert m["recall@5_en"] == 0.5 and m["recall@5_sw"] == 1.0
    assert m["citation_accuracy"] == pytest.approx(2 / 3, abs=1e-4)         # answerable citations on target
    assert m["citation_quote_validity"] == pytest.approx(4 / 5, abs=1e-4)   # verified / proposed
    assert m["correct_refusal_rate"] == 0.5
    assert m["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["faithfulness"] == 0.75 and m["correctness"] == 0.5


def test_errors_are_not_scored_as_wrong_answers():
    results = [QuestionResult("a", "en", False, [], status="error", error="boom"),
               QuestionResult("b", "en", False, [], status="not_found")]
    m = compute_metrics(results, k=5)
    assert m["errors"] == 1 and m["correct_refusal_rate"] == 1.0


def test_fixture_golden_set_is_well_formed():
    questions = load_questions(ROOT / "evals" / "questions.jsonl")
    assert len(questions) >= 10
    assert {q.language for q in questions} == {"en", "sw"}
    assert any(not q.answerable for q in questions)
    assert all(q.expected_clauses for q in questions if q.answerable)


def test_retrieval_eval_on_fixtures_meets_ci_threshold():
    embedder = HashingEmbedder()
    conn = load_fixture_corpus(ROOT / "evals" / "fixtures" / "corpus.json", embedder)
    metrics, results = run_eval(conn, embedder, load_questions(ROOT / "evals" / "questions.jsonl"), k=5)
    assert metrics["recall@5"] >= 0.8
    assert all(r.status is None for r in results)  # retrieval-only: Claude never called


class ScriptedClient:
    """Answers QA calls with `answer_payload` and judge calls with a fixed verdict."""

    def __init__(self, answer_payload):
        from types import SimpleNamespace

        self.calls = []
        self._answer = answer_payload
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        is_judge = "You grade answers" in kwargs["system"][0]["text"]
        payload = ({"faithfulness": "faithful", "correctness": "correct", "rationale": "ok"} if is_judge
                   else self._answer(kwargs["messages"][0]["content"]))
        return SimpleNamespace(stop_reason="end_turn", stop_details=None, model=kwargs["model"],
                               content=[SimpleNamespace(type="text", text=json.dumps(payload))])


def test_full_eval_with_mocked_claude_and_judge():
    def answer(content):
        if "Clause 2. Water conservation levy" in content:
            return {"found": True, "answer": "Two per cent.",
                    "citations": [{"source": 1, "quote": "water conservation levy of two per cent"}]}
        return {"found": False, "answer": "Not covered.", "citations": []}

    embedder = HashingEmbedder()
    conn = load_fixture_corpus(ROOT / "evals" / "fixtures" / "corpus.json", embedder)
    questions = [q for q in load_questions(ROOT / "evals" / "questions.jsonl")
                 if q.id in ("water-levy-rate-en", "unanswerable-seeds-en", "boda-training-hours-en")]
    client = ScriptedClient(answer)
    metrics, results = run_eval(conn, embedder, questions, k=3, client=client, judge_client=client)

    by_id = {r.id: r for r in results}
    assert by_id["water-levy-rate-en"].status == "answered"
    assert by_id["water-levy-rate-en"].citations_on_target == 1
    assert by_id["unanswerable-seeds-en"].status == "not_found"
    assert by_id["boda-training-hours-en"].judge["correctness"] == "incorrect"  # refused: no judge call spent
    assert metrics["correct_refusal_rate"] == 1.0
    assert metrics["citation_accuracy"] == 1.0 and metrics["citation_quote_validity"] == 1.0
    assert metrics["correctness"] == 0.5
    judge_calls = [c for c in client.calls if "You grade answers" in c["system"][0]["text"]]
    assert len(judge_calls) == 1 and judge_calls[0]["model"] == "claude-sonnet-5-5"
