"""Split bill text into clause-aware chunks.

Kenyan bills are numbered clause by clause ("5. (1) The Authority shall ..."), grouped under PART headings,
and end with Schedules. Some drafts and amendment papers write "Clause 5" or "Section 5" headings instead.
One chunk per clause keeps each citation pointing at a single, checkable clause. Text with no recognisable
structure falls back to fixed-size windows with overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 2000      # clauses longer than this are split into overlapping windows
FALLBACK_CHARS = 1200
OVERLAP_CHARS = 200
MIN_HEADINGS = 2      # fewer clause headings than this means "no structure found"

_PART = re.compile(r"^[ \t]*PART[ \t]+([IVXLC]+|\d+)\b[^\n]{0,100}$", re.MULTILINE)
_SCHEDULE = re.compile(
    r"^[ \t]*((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)[ \t]+)?SCHEDULE\b[^\n]{0,80}$",
    re.MULTILINE,
)
# "Clause 5." / "Section 12A:" / "Clause 3 —". The punctuation (or end of line) after the number stops body
# text such as "Section 12 of the Income Tax Act is amended" from being read as a heading.
_NAMED = re.compile(r"^[ \t]*(?:Clause|Section)[ \t]+(\d{1,3}[A-Z]?)[ \t]*(?:[.:–—-]|$)", re.MULTILINE)
# "5. (1) The Authority ..." or "12A. Interpretation".
_NUMBERED = re.compile(r"^[ \t]*(\d{1,3})([A-Z]?)\.[ \t]+(?=\S)", re.MULTILINE)
_PAGE_MARKER = re.compile(r"\[page \d+\]")


@dataclass(frozen=True)
class Chunk:
    bill_id: int
    ordinal: int          # position within the bill, 0-based
    label: str            # what a citation shows: "Clause 5", "First Schedule", "Preamble", "Passage 3"
    part: str | None      # enclosing PART heading, if any
    start: int            # char offsets into the original text: text[start:end] == chunk.text
    end: int
    text: str


@dataclass(frozen=True)
class _Heading:
    start: int
    kind: str             # "part" | "schedule" | "clause"
    label: str


def _ordinal_word(match: re.Match) -> str:
    return (match.group(1) or "").strip().title()


def _headings(text: str) -> list[_Heading]:
    found: list[_Heading] = []
    for m in _PART.finditer(text):
        found.append(_Heading(m.start(), "part", f"Part {m.group(1)}"))
    for m in _SCHEDULE.finditer(text):
        found.append(_Heading(m.start(), "schedule", f"{_ordinal_word(m)} Schedule".strip()))
    for m in _NAMED.finditer(text):
        found.append(_Heading(m.start(), "clause", f"Clause {m.group(1)}"))
    numbered = [(m.start(), int(m.group(1)), m.group(1) + m.group(2)) for m in _NUMBERED.finditer(text)]

    # Walk in document order so numbered paragraphs can be checked against their context. Inside a Schedule
    # they stay part of the Schedule (cited as "First Schedule"), and a number that does not go up (a list
    # inside a clause, or the Memorandum's own numbering) is not treated as a new clause.
    events = sorted([(h.start, 0, h) for h in found] + [(s, 1, (n, label)) for s, n, label in numbered],
                    key=lambda e: (e[0], e[1]))
    result: list[_Heading] = []
    schedule: str | None = None
    last_number = 0
    taken = set()
    for start, kind, payload in events:
        if start in taken:
            continue
        if kind == 0:
            heading = payload
            if heading.kind == "schedule":
                schedule = heading.label
            elif heading.kind == "part":
                schedule = None
            result.append(heading)
            taken.add(start)
            continue
        number, label = payload
        if schedule or number <= last_number:
            continue
        last_number = number
        result.append(_Heading(start, "clause", f"Clause {label}"))
        taken.add(start)
    return result


def _windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Fixed-size windows over text[start:end], cut at whitespace where possible."""
    spans = []
    pos = start
    while pos < end:
        stop = min(pos + size, end)
        if stop < end:
            space = text.rfind(" ", pos + size // 2, stop)
            newline = text.rfind("\n", pos + size // 2, stop)
            cut = max(space, newline)
            if cut > pos:
                stop = cut
        spans.append((pos, stop))
        if stop >= end:
            break
        pos = max(stop - overlap, pos + 1)
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _substantive(fragment: str) -> bool:
    return len(_PAGE_MARKER.sub("", fragment).strip()) >= 20


def chunk_bill(bill_id: int, text: str, *, max_chars: int = MAX_CHARS) -> list[Chunk]:
    headings = _headings(text)
    clause_count = sum(h.kind == "clause" for h in headings)
    if clause_count < MIN_HEADINGS:
        return chunk_fixed(bill_id, text)

    segments: list[tuple[str, str | None, int, int]] = []  # (label, part, start, end)
    if headings[0].start > 0:
        segments.append(("Preamble", None, 0, headings[0].start))
    part: str | None = None
    for i, heading in enumerate(headings):
        end = headings[i + 1].start if i + 1 < len(headings) else len(text)
        if heading.kind == "part":
            part = heading.label
            # The PART line is a heading for what follows; keep it only if it carries body text of its own.
            first_newline = text.find("\n", heading.start, end)
            body = text[first_newline:end] if first_newline != -1 else ""
            if not _substantive(body):
                continue
        if heading.kind == "schedule":
            part = None
        segments.append((heading.label, part, heading.start, end))

    chunks: list[Chunk] = []
    for label, seg_part, start, end in segments:
        start, end = _trim(text, start, end)
        if not _substantive(text[start:end]):
            continue
        spans = [(start, end)] if end - start <= max_chars else _windows(text, start, end, max_chars, OVERLAP_CHARS)
        for s, e in spans:
            chunks.append(Chunk(bill_id, len(chunks), label, seg_part, s, e, text[s:e]))
    return chunks


def chunk_fixed(bill_id: int, text: str, *, size: int = FALLBACK_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for s, e in _windows(text, 0, len(text), size, overlap):
        s, e = _trim(text, s, e)
        if s < e and _substantive(text[s:e]):
            chunks.append(Chunk(bill_id, len(chunks), f"Passage {len(chunks) + 1}", None, s, e, text[s:e]))
    return chunks


_CLAUSE_REF = re.compile(r"^(?:clause|section|kifungu|cl\.|s\.)\s*(\d+[a-z]?)\b", re.IGNORECASE)


def normalize_label(label: str) -> str:
    """Canonical form for comparing clause references: "Section 5", "clause 5", "Kifungu 5" -> "clause 5"."""
    cleaned = " ".join(label.lower().replace("–", " ").replace("—", " ").split())
    m = _CLAUSE_REF.match(cleaned)
    if m:
        return f"clause {m.group(1)}"
    if cleaned.isdigit():
        return f"clause {cleaned}"
    return cleaned
