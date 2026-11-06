"""Clause chunks and their vectors, stored with sqlite-vec in the same SQLite file as the bills.

Tables (all created here, so store.py is untouched):
- bill_texts: extracted text per bill (filled by `bunge-watch index --fetch`; scans are recorded but not indexed)
- rag_bills: which text (by hash) and which embedder each bill was indexed with
- rag_chunks: one row per chunk, with its clause label and char offsets into bill_texts.text
- rag_vectors: sqlite-vec vec0 table keyed by chunk id, with bill_id as a filterable metadata column
- rag_meta: the embedder name and dimension the vectors were built with
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import store
from .chunker import Chunk, chunk_bill
from .embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS bill_texts (
    bill_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    looks_scanned INTEGER NOT NULL,
    extracted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_bills (
    bill_id INTEGER PRIMARY KEY,
    text_sha256 TEXT NOT NULL,
    embedder TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY,
    bill_id INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    label TEXT NOT NULL,
    part TEXT,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE (bill_id, ordinal)
);
CREATE TABLE IF NOT EXISTS rag_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class IndexMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Retrieved:
    chunk_id: int
    bill_id: int
    label: str
    part: str | None
    start: int
    end: int
    text: str
    distance: float  # cosine distance: 0 is identical


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_vec(conn: sqlite3.Connection) -> sqlite3.Connection:
    try:
        import sqlite_vec
    except ImportError as err:
        raise RuntimeError("sqlite-vec is not installed: pip install -e .") from err
    if not hasattr(conn, "enable_load_extension"):
        raise RuntimeError("This Python's sqlite3 cannot load extensions; use a python.org/uv/pyenv build.")
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)
    conn.executescript(SCHEMA)
    return conn


def connect(path: str) -> sqlite3.Connection:
    """The bills database (via store.connect, so the bills tables exist) with sqlite-vec loaded."""
    return load_vec(store.connect(path))


def _vectors_ready(conn: sqlite3.Connection, embedder: Embedder) -> None:
    """Create the vec0 table on first use and refuse to mix vectors from different embedders."""
    meta = dict(conn.execute("SELECT key, value FROM rag_meta").fetchall())
    if not meta:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS rag_vectors USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, bill_id INTEGER, embedding FLOAT[{int(embedder.dim)}] distance_metric=cosine)"
        )
        conn.executemany("INSERT INTO rag_meta (key, value) VALUES (?, ?)",
                         [("embedder", embedder.name), ("dim", str(embedder.dim))])
        conn.commit()
    elif meta.get("embedder") != embedder.name or meta.get("dim") != str(embedder.dim):
        raise IndexMismatch(
            f"Index was built with {meta.get('embedder')} ({meta.get('dim')} dims), not {embedder.name}. "
            "Run `bunge-watch index --rebuild` to re-embed everything."
        )


def reset(conn: sqlite3.Connection) -> None:
    """Drop all chunks and vectors (keeps bill_texts, which is expensive to re-download)."""
    conn.executescript(
        "DROP TABLE IF EXISTS rag_vectors; DELETE FROM rag_chunks; DELETE FROM rag_bills; DELETE FROM rag_meta;"
    )
    conn.commit()


def save_text(conn: sqlite3.Connection, bill_id: int, text: str, looks_scanned: bool) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO bill_texts (bill_id, text, looks_scanned, extracted_at) VALUES (?, ?, ?, ?)",
        (bill_id, text, int(looks_scanned), _now()),
    )
    conn.commit()


def bills_missing_text(conn: sqlite3.Connection, limit: int | None = None) -> list[sqlite3.Row]:
    sql = ("SELECT b.id, b.title, b.pdf_url FROM bills b LEFT JOIN bill_texts t ON t.bill_id = b.id"
           " WHERE t.bill_id IS NULL ORDER BY b.id DESC")
    return conn.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()


def indexable_bills(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Bills with a usable text layer. Scans have no text to chunk, so they are skipped until OCR exists."""
    return conn.execute(
        "SELECT bill_id, text FROM bill_texts WHERE looks_scanned = 0 AND length(trim(text)) > 0 ORDER BY bill_id"
    ).fetchall()


def index_bill(conn: sqlite3.Connection, embedder: Embedder, bill_id: int, text: str) -> str:
    """Chunk, embed and store one bill. Returns "indexed", "unchanged" or "empty".

    Idempotent: the same text with the same embedder is a no-op, and changed text replaces the bill's old
    chunks and vectors in a single transaction.
    """
    _vectors_ready(conn, embedder)
    sha = hashlib.sha256(text.encode()).hexdigest()
    row = conn.execute("SELECT text_sha256, embedder FROM rag_bills WHERE bill_id = ?", (bill_id,)).fetchone()
    if row and row["text_sha256"] == sha and row["embedder"] == embedder.name:
        return "unchanged"

    chunks: list[Chunk] = chunk_bill(bill_id, text)
    vectors = embedder.embed_documents([c.text for c in chunks]) if chunks else []

    import sqlite_vec

    with conn:  # one transaction: a crash never leaves a half-indexed bill
        conn.execute("DELETE FROM rag_vectors WHERE bill_id = ?", (bill_id,))
        conn.execute("DELETE FROM rag_chunks WHERE bill_id = ?", (bill_id,))
        for chunk, vector in zip(chunks, vectors):
            cur = conn.execute(
                "INSERT INTO rag_chunks (bill_id, ordinal, label, part, start_char, end_char, text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bill_id, chunk.ordinal, chunk.label, chunk.part, chunk.start, chunk.end, chunk.text),
            )
            conn.execute(
                "INSERT INTO rag_vectors (chunk_id, bill_id, embedding) VALUES (?, ?, ?)",
                (cur.lastrowid, bill_id, sqlite_vec.serialize_float32(vector)),
            )
        conn.execute(
            "INSERT OR REPLACE INTO rag_bills (bill_id, text_sha256, embedder, chunk_count, indexed_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (bill_id, sha, embedder.name, len(chunks), _now()),
        )
    return "indexed" if chunks else "empty"


def search(
    conn: sqlite3.Connection, embedder: Embedder, question: str, *, k: int = 5, bill_id: int | None = None
) -> list[Retrieved]:
    """Top-k chunks by cosine distance, optionally limited to one bill."""
    if conn.execute("SELECT 1 FROM rag_meta LIMIT 1").fetchone() is None:
        return []
    _vectors_ready(conn, embedder)
    import sqlite_vec

    sql = "SELECT chunk_id, distance FROM rag_vectors WHERE embedding MATCH ? AND k = ?"
    params: list = [sqlite_vec.serialize_float32(embedder.embed_query(question)), k]
    if bill_id is not None:
        sql += " AND bill_id = ?"
        params.append(bill_id)
    hits = conn.execute(sql + " ORDER BY distance", params).fetchall()
    results = []
    for chunk_id, distance in hits:
        c = conn.execute("SELECT * FROM rag_chunks WHERE id = ?", (chunk_id,)).fetchone()
        results.append(Retrieved(c["id"], c["bill_id"], c["label"], c["part"], c["start_char"], c["end_char"],
                                 c["text"], float(distance)))
    return results
