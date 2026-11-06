from fastapi import FastAPI
from fastapi.testclient import TestClient

from bunge_watch.api import create_app
from bunge_watch.rag.qa import Answer, Citation
from bunge_watch.rag.routes import create_router


def fake_answer(req):
    return Answer(
        question=req.question, status="answered", answer="Two per cent.",
        citations=[Citation(bill_id=9002, clause="Clause 2", quote="two per cent", chunk_id=1)],
        rejected_citations=[], sources=[], model="claude-opus-5-5",
    )


def app_with(answer):
    app = FastAPI()
    app.include_router(create_router(":memory:", answer=answer))
    return TestClient(app)


def test_ask_returns_structured_answer(monkeypatch):
    monkeypatch.setenv("BUNGE_WATCH_ENABLE_ASK", "1")
    seen = []
    api = app_with(lambda req: seen.append(req) or fake_answer(req))

    body = api.post("/ask", json={"question": "What is the levy?", "bill_id": 9002, "k": 3}).json()
    assert body["status"] == "answered"
    assert body["citations"] == [{"bill_id": 9002, "clause": "Clause 2", "quote": "two per cent", "chunk_id": 1}]
    assert body["reviewed"] is False
    assert (seen[0].bill_id, seen[0].k) == (9002, 3)


def test_ask_validates_input(monkeypatch):
    monkeypatch.setenv("BUNGE_WATCH_ENABLE_ASK", "1")
    api = app_with(fake_answer)
    assert api.post("/ask", json={"question": ""}).status_code == 422
    assert api.post("/ask", json={"question": "ok?", "k": 500}).status_code == 422


def test_ask_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("BUNGE_WATCH_ENABLE_ASK", raising=False)
    api = TestClient(create_app(str(tmp_path / "db.sqlite")))  # mounted on the real app
    assert api.post("/ask", json={"question": "What is the levy?"}).status_code == 503
    assert api.get("/health").json() == {"status": "ok"}
