import json
from pathlib import Path

import pytest

from bunge_watch.rag import index
from bunge_watch.rag.embeddings import HashingEmbedder

FIXTURES = Path(__file__).parent.parent / "evals" / "fixtures"


def load_fixture_corpus(manifest, embedder):
    conn = index.connect(":memory:")
    for bill in json.loads(Path(manifest).read_text())["bills"]:
        index.index_bill(conn, embedder, bill["bill_id"], (FIXTURES / bill["file"]).read_text())
    return conn


def count(conn, table, bill_id):
    return conn.execute(f"SELECT count(*) FROM {table} WHERE bill_id = ?", (bill_id,)).fetchone()[0]


def test_reindexing_is_idempotent(tmp_path):
    conn = index.connect(str(tmp_path / "db.sqlite"))
    embedder = HashingEmbedder()
    text = (FIXTURES / "boda_boda_safety_bill.txt").read_text()

    assert index.index_bill(conn, embedder, 1, text) == "indexed"
    first = (count(conn, "rag_chunks", 1), count(conn, "rag_vectors", 1))
    assert first[0] == first[1] > 0

    assert index.index_bill(conn, embedder, 1, text) == "unchanged"
    assert (count(conn, "rag_chunks", 1), count(conn, "rag_vectors", 1)) == first

    shorter = text.split("PART III")[0]
    assert index.index_bill(conn, embedder, 1, shorter) == "indexed"
    after = (count(conn, "rag_chunks", 1), count(conn, "rag_vectors", 1))
    assert after[0] == after[1] < first[0]  # old chunks and vectors replaced, not appended


def test_only_bills_with_a_text_layer_are_indexable(tmp_path):
    conn = index.connect(str(tmp_path / "db.sqlite"))
    index.save_text(conn, 1, "1. Short title\nThis Act...\n2. Next\nMore text here.", looks_scanned=False)
    index.save_text(conn, 2, "[page 1]\n", looks_scanned=True)
    assert [r["bill_id"] for r in index.indexable_bills(conn)] == [1]


def test_switching_embedder_requires_rebuild(tmp_path):
    conn = index.connect(str(tmp_path / "db.sqlite"))
    index.index_bill(conn, HashingEmbedder(dim=64), 1, "Some text without structure that is long enough.")
    with pytest.raises(index.IndexMismatch, match="rebuild"):
        index.index_bill(conn, HashingEmbedder(dim=128), 1, "Other text")
    index.reset(conn)
    assert index.index_bill(conn, HashingEmbedder(dim=128), 1, "Other text that is long enough too.") == "indexed"


@pytest.mark.parametrize(
    "question,bill_id,label",
    [
        ("How many hours of road safety training must a rider complete?", 9001, "Clause 4"),
        ("What is the rate of the water conservation levy?", 9002, "Clause 2"),
        ("What penalty applies to late remittance of the levy?", 9002, "Clause 5"),
        ("Maximum daily market fee a county may charge a trader", 9003, "Passage 1"),
    ],
)
def test_retrieval_returns_the_right_clause(question, bill_id, label):
    conn = load_fixture_corpus(FIXTURES / "corpus.json", HashingEmbedder())
    top = index.search(conn, HashingEmbedder(), question, k=3)[0]
    assert (top.bill_id, top.label) == (bill_id, label)


def test_search_can_filter_by_bill():
    embedder = HashingEmbedder()
    conn = load_fixture_corpus(FIXTURES / "corpus.json", embedder)
    hits = index.search(conn, embedder, "penalty fine", k=5, bill_id=9002)
    assert hits and {h.bill_id for h in hits} == {9002}


def test_search_on_empty_index_returns_nothing(tmp_path):
    conn = index.connect(str(tmp_path / "db.sqlite"))
    assert index.search(conn, HashingEmbedder(), "anything") == []
