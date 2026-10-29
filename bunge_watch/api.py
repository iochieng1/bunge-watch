"""Read-only public API. Only reviewed (published) summaries are ever returned."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import store


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="Bunge Watch", description="Kenyan bills in plain English and Kiswahili")
    conn = store.connect(db_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/bills")
    def bills(chamber: str | None = None, q: str | None = None) -> list[dict]:
        return store.list_bills(conn, chamber=chamber, query=q)

    @app.get("/bills/{bill_id}")
    def bill(bill_id: int) -> dict:
        found = store.get_bill(conn, bill_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Bill not found")
        return found

    return app
