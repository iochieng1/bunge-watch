"""Download bill PDFs and pull their text out."""

from __future__ import annotations

import io
import logging

import httpx
from pypdf import PdfReader

# Many Parliament PDFs have minor structural errors that pypdf recovers from; don't spam the console.
logging.getLogger("pypdf").setLevel(logging.ERROR)

# Below this many characters per page we treat the PDF as a scan with no text layer.
MIN_CHARS_PER_PAGE = 200


def download_pdf(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def extract_text(pdf: bytes) -> tuple[str, bool]:
    """Return (text, looks_scanned)."""
    reader = PdfReader(io.BytesIO(pdf))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(f"[page {i}]\n{t.strip()}" for i, t in enumerate(pages, start=1))
    chars = sum(len(t.strip()) for t in pages)
    looks_scanned = not pages or chars / len(pages) < MIN_CHARS_PER_PAGE
    return text, looks_scanned
