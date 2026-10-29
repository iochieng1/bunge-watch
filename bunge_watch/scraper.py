"""Scrape bill listings from the Parliament of Kenya website (parliament.go.ke)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.parliament.go.ke"
LISTING_URLS = {
    "national_assembly": f"{BASE_URL}/the-national-assembly/house-business/bills",
    "senate": f"{BASE_URL}/the-senate/house-business/bills",
}
USER_AGENT = "BungeWatch/0.1 (civic tech; https://github.com/iochieng1/bunge-watch)"


@dataclass(frozen=True)
class BillListing:
    chamber: str
    title: str
    pdf_url: str
    pdf_bytes: int | None
    published_month: str | None  # "2026-10", taken from the upload folder in the PDF URL
    comments_url: str | None
    year: int | None


SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to"}


ACRONYMS = {"na": "NA"}  # "NA Bill No. 5" = National Assembly Bill


def _title_case(title: str) -> str:
    words = title.split(" ")
    out = []
    for i, word in enumerate(words):
        bare = word.strip("()[],.-").lower()
        starts_clause = i == 0 or words[i - 1] in {"-", "–"}
        if bare in ACRONYMS:
            out.append(word.replace(word.strip("()[],.-"), ACRONYMS[bare]))
        elif bare in SMALL_WORDS and not starts_clause:
            out.append(word.lower())
        else:
            out.append(re.sub(r"[A-Za-z]+", lambda m: m[0][0].upper() + m[0][1:].lower(), word))
    return " ".join(out)


def clean_title(raw: str) -> str:
    """Normalise the inconsistent titles on the site, e.g. 'THE TAX LAWS(AMENDMENT) BILL,2026'."""
    title = re.sub(r"\s+", " ", raw).strip()
    title = re.sub(r"\.(pdf|docx?)$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"(\d{4})\s*\(\d+\)$", r"\1", title)  # "..., 2024 (1)" duplicate-upload suffix
    title = title.rstrip(".").strip()
    title = re.sub(r"\(\s*", "(", title)
    title = re.sub(r"\s*\)", ")", title)
    title = re.sub(r"(\w)\(", r"\1 (", title)
    title = re.sub(r"\s*,\s*", ", ", title)
    return _title_case(title)


def _year(title: str) -> int | None:
    years = re.findall(r"\b(?:19|20)\d{2}\b", title)
    return int(years[-1]) if years else None


def _pdf_bytes(type_attr: str | None) -> int | None:
    match = re.search(r"length=(\d+)", type_attr or "")
    return int(match.group(1)) if match else None


def _published_month(pdf_url: str) -> str | None:
    match = re.search(r"/files/(\d{4}-\d{2})/", pdf_url)
    return match.group(1) if match else None


def parse_listing(html: str, chamber: str) -> list[BillListing]:
    soup = BeautifulSoup(html, "html.parser")
    bills = []
    for block in soup.select("div.post-block"):
        link = block.select_one(".post-title a[href$='.pdf']")
        if link is None:
            continue
        title = clean_title(link.get_text())
        if not re.search(r"\bbill\b", title, re.IGNORECASE):
            continue  # Hansard, order papers and other documents share this layout
        comments = block.select_one(".post-petition a")
        pdf_url = link["href"]
        bills.append(
            BillListing(
                chamber=chamber,
                title=title,
                pdf_url=pdf_url,
                pdf_bytes=_pdf_bytes(link.get("type")),
                published_month=_published_month(pdf_url),
                comments_url=comments["href"] if comments else None,
                year=_year(title),
            )
        )
    return bills


def last_page(html: str) -> int:
    # Pager links are HTML-escaped in the page source: "?title=%20&amp;page=14"
    pages = [int(p) for p in re.findall(r"[?&;]page=(\d+)", html)]
    return max(pages, default=0)


def make_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def scrape_chamber(client: httpx.Client, chamber: str, max_pages: int | None = None) -> list[BillListing]:
    url = LISTING_URLS[chamber]
    first = client.get(url, params={"page": 0})
    first.raise_for_status()
    pages = last_page(first.text) + 1
    if max_pages is not None:
        pages = min(pages, max_pages)

    seen: dict[str, BillListing] = {}
    for page in range(pages):
        html = first.text if page == 0 else client.get(url, params={"page": page}).raise_for_status().text
        for bill in parse_listing(html, chamber):
            seen.setdefault(bill.pdf_url, bill)
    return list(seen.values())
