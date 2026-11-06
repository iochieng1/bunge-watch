from pathlib import Path

from bunge_watch.rag.chunker import chunk_bill, chunk_fixed, normalize_label

FIXTURES = Path(__file__).parent.parent / "evals" / "fixtures"

NUMBERED_BILL = """\
THE EXAMPLE BILL, 2099

PART I - PRELIMINARY

1. Short title
This Act may be cited as the Example Act, 2099.

2. Interpretation
In this Act, unless the context otherwise requires-
(a) a list item that should stay inside clause 2;
1. a numbered list item that must not start a new clause.

PART II - SUBSTANCE

3. Amendment of section 12 of Cap. 470
Section 12 of the Income Tax Act is amended by deleting the words "ten" and substituting "twelve".

SECOND SCHEDULE
1. First paragraph of the schedule.
2. Second paragraph of the schedule.
"""


def test_numbered_clauses_parts_and_schedules():
    chunks = chunk_bill(7, NUMBERED_BILL)
    labels = [(c.label, c.part) for c in chunks]
    assert labels == [
        ("Preamble", None),
        ("Clause 1", "Part I"),
        ("Clause 2", "Part I"),
        ("Clause 3", "Part II"),
        ("Second Schedule", None),
    ]
    for c in chunks:
        assert NUMBERED_BILL[c.start:c.end] == c.text
        assert c.bill_id == 7
    clause2 = chunks[2].text
    assert "numbered list item" in clause2  # a "1." inside clause 2 is not a new clause
    assert "Section 12 of the Income Tax Act" in chunks[3].text  # body text "Section 12 of" is not a heading


def test_named_clause_headings_on_fixture():
    text = (FIXTURES / "county_water_levy_bill.txt").read_text()
    labels = [c.label for c in chunk_bill(9002, text)]
    assert labels == ["Preamble", "Clause 1", "Clause 2", "Clause 3", "Clause 4", "Clause 5", "Clause 6"]


def test_long_clause_is_split_with_same_label():
    body = " ".join(f"word{i}" for i in range(800))
    text = f"1. Short title\nShort.\n\n2. Long clause\n{body}\n\n3. Last\nThe end of the bill text here.\n"
    chunks = chunk_bill(1, text, max_chars=1000)
    long_parts = [c for c in chunks if c.label == "Clause 2"]
    assert len(long_parts) > 1
    assert long_parts[0].end > long_parts[1].start  # overlapping windows
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_unstructured_text_falls_back_to_overlapping_windows():
    text = (FIXTURES / "market_traders_notice.txt").read_text()
    assert [c.label for c in chunk_bill(9003, text)] == ["Passage 1"]

    chunks = chunk_fixed(9003, text, size=300, overlap=60)
    assert len(chunks) >= 3
    assert [c.label for c in chunks][:2] == ["Passage 1", "Passage 2"]
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap
        assert len(a.text) <= 300
    assert chunks[-1].end == len(text.rstrip())


def test_normalize_label():
    assert normalize_label("Section 5") == normalize_label("clause 5") == normalize_label("Kifungu 5") == "clause 5"
    assert normalize_label("5") == "clause 5"
    assert normalize_label("First  Schedule") == "first schedule"
