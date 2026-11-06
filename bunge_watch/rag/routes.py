"""POST /ask: answer a question from the indexed bills, citing the exact clauses.

Answers are generated live and are not reviewed by a person, unlike published summaries. The endpoint is
therefore off unless BUNGE_WATCH_ENABLE_ASK=1, and every response carries `reviewed: false`.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .qa import DEFAULT_K, Answer, QAError, ask


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    bill_id: int | None = None
    k: int = Field(default=DEFAULT_K, ge=1, le=20)


def ask_enabled() -> bool:
    return os.environ.get("BUNGE_WATCH_ENABLE_ASK") == "1"


def create_router(
    db_path: str,
    *,
    answer: Callable[[AskRequest], Answer] | None = None,
) -> APIRouter:
    """`answer` is injectable for tests; by default it is built lazily on the first request, so the rest of
    the API starts without sqlite-vec, an embedding model or Claude credentials."""
    router = APIRouter(tags=["ask"])
    lock = threading.Lock()
    state: dict = {"answer": answer}

    def default_answer() -> Callable[[AskRequest], Answer]:
        import anthropic

        from .embeddings import get_embedder
        from .index import connect

        conn = connect(db_path)
        embedder = get_embedder(os.environ.get("BUNGE_WATCH_EMBEDDER", "local"))
        client = anthropic.Anthropic()

        def run(req: AskRequest) -> Answer:
            with lock:  # one shared SQLite connection
                return ask(client, conn, embedder, req.question, k=req.k, bill_id=req.bill_id)

        return run

    @router.post("/ask", response_model=Answer)
    def ask_endpoint(req: AskRequest) -> Answer:
        if not ask_enabled():
            raise HTTPException(status_code=503, detail="Q&A is disabled. Set BUNGE_WATCH_ENABLE_ASK=1 to enable.")
        if state["answer"] is None:
            state["answer"] = default_answer()
        try:
            return state["answer"](req)
        except QAError as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    return router
