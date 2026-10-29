from pathlib import Path

import httpx

from bunge_watch import scraper

FIXTURE = (Path(__file__).parent / "fixtures" / "national_assembly_bills.html").read_text(encoding="utf8")

HANSARD_BLOCK = """
<div class="post-block"><div class="post-content"><div class="post-title"><span>
<a href="https://www.parliament.go.ke/sites/default/files/2026-10/The%20Hansard.pdf" type="application/pdf; length=850623">
The Hansard - Thursday, 22 October 2026</a></span></div></div></div>
"""


def test_parse_listing_reads_real_page():
    bills = scraper.parse_listing(FIXTURE, "national_assembly")
    assert [b.title for b in bills] == [
        "The Livestock Bill, 2026",
        "The Anti-Doping (Amendment) Bill, 2026",
        "The Tax Laws (Amendment) Bill, 2026",
        "The National Coroners Service Bill, 2026",
    ]
    tax = bills[2]
    assert tax.year == 2026
    assert tax.published_month == "2026-10"
    assert tax.pdf_bytes == 2350237
    assert tax.pdf_url.endswith(".pdf")
    assert "national_assembly_petition" in tax.comments_url


def test_parse_listing_skips_documents_that_are_not_bills():
    assert scraper.parse_listing(HANSARD_BLOCK, "national_assembly") == []


def test_clean_title_handles_senate_style():
    raw = "The Statutory Instruments ( Amendment) Bill,No.11 of 2026."
    assert scraper.clean_title(raw) == "The Statutory Instruments (Amendment) Bill, No.11 of 2026"


def test_clean_title_fixes_case_and_upload_suffixes():
    raw = "Senate Bill No. 45 of 2024 - THE SPORTS (AMENDMENT) (NO.2) BILL, 2024 (1).pdf"
    assert scraper.clean_title(raw) == "Senate Bill No. 45 of 2024 - The Sports (Amendment) (No.2) Bill, 2024"
    assert scraper.clean_title("The state corporation (amendment) bill, 2026") == (
        "The State Corporation (Amendment) Bill, 2026"
    )


def test_last_page():
    assert scraper.last_page(FIXTURE) == 14


def test_scrape_chamber_walks_pages_and_dedupes():
    pages_requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        pages_requested.append(page)
        # pretend the listing has only two pages
        html = FIXTURE.replace("page=14", "page=1").replace("page=2", "page=1") if page == 0 else FIXTURE
        return httpx.Response(200, text=html)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        bills = scraper.scrape_chamber(client, "national_assembly")

    assert pages_requested == [0, 1]
    assert len(bills) == 4  # the same four bills on both pages are stored once
