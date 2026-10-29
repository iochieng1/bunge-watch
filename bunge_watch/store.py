"""SQLite storage for bills and their reviewed summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from .scraper import BillListing
from .summarizer import BillSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS bills (
    id INTEGER PRIMARY KEY,
    chamber TEXT NOT NULL,
    title TEXT NOT NULL,
    pdf_url TEXT NOT NULL UNIQUE,
    pdf_bytes INTEGER,
    published_month TEXT,
    comments_url TEXT,
    year INTEGER,
    first_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    bill_id INTEGER PRIMARY KEY REFERENCES bills(id),
    status TEXT NOT NULL CHECK (status IN ('draft', 'published')),
    model TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    published_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_bill(conn: sqlite3.Connection, bill: BillListing) -> tuple[int, bool]:
    """Insert a newly seen bill. Returns (bill id, whether it was new)."""
    row = conn.execute("SELECT id FROM bills WHERE pdf_url = ?", (bill.pdf_url,)).fetchone()
    if row:
        return row["id"], False
    cur = conn.execute(
        "INSERT INTO bills (chamber, title, pdf_url, pdf_bytes, published_month, comments_url, year, first_seen)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (bill.chamber, bill.title, bill.pdf_url, bill.pdf_bytes, bill.published_month,
         bill.comments_url, bill.year, _now()),
    )
    conn.commit()
    return cur.lastrowid, True


def bills_needing_summary(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT b.* FROM bills b LEFT JOIN summaries s ON s.bill_id = b.id"
        " WHERE s.bill_id IS NULL ORDER BY b.published_month DESC, b.id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def save_draft(conn: sqlite3.Connection, bill_id: int, summary: BillSummary, model: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO summaries (bill_id, status, model, summary_json, created_at)"
        " VALUES (?, 'draft', ?, ?, ?)",
        (bill_id, model, summary.model_dump_json(), _now()),
    )
    conn.commit()


def drafts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT b.id, b.title, s.summary_json FROM summaries s JOIN bills b ON b.id = s.bill_id"
        " WHERE s.status = 'draft' ORDER BY b.id"
    ).fetchall()


def publish(conn: sqlite3.Connection, bill_id: int) -> bool:
    cur = conn.execute(
        "UPDATE summaries SET status = 'published', published_at = ? WHERE bill_id = ? AND status = 'draft'",
        (_now(), bill_id),
    )
    conn.commit()
    return cur.rowcount == 1


def _public(row: sqlite3.Row) -> dict:
    bill = {k: row[k] for k in ("id", "chamber", "title", "pdf_url", "published_month", "comments_url", "year")}
    bill["summary"] = json.loads(row["summary_json"]) if row["status"] == "published" else None
    return bill


_PUBLIC_QUERY = (
    "SELECT b.*, s.status, s.summary_json FROM bills b LEFT JOIN summaries s ON s.bill_id = b.id"
)


def list_bills(conn: sqlite3.Connection, chamber: str | None = None, query: str | None = None) -> list[dict]:
    sql, params = _PUBLIC_QUERY + " WHERE 1=1", []
    if chamber:
        sql += " AND b.chamber = ?"
        params.append(chamber)
    if query:
        sql += " AND b.title LIKE ?"
        params.append(f"%{query}%")
    sql += " ORDER BY b.published_month DESC, b.id DESC"
    return [_public(r) for r in conn.execute(sql, params).fetchall()]


def get_bill(conn: sqlite3.Connection, bill_id: int) -> dict | None:
    row = conn.execute(_PUBLIC_QUERY + " WHERE b.id = ?", (bill_id,)).fetchone()
    return _public(row) if row else None
