from fastapi.testclient import TestClient

from bunge_watch import store
from bunge_watch.api import create_app
from bunge_watch.scraper import BillListing
from bunge_watch.summarizer import BillSummary

BILL = BillListing(
    chamber="national_assembly",
    title="The Tax Laws (Amendment) Bill, 2026",
    pdf_url="https://www.parliament.go.ke/sites/default/files/2026-10/tax.pdf",
    pdf_bytes=2350237,
    published_month="2026-10",
    comments_url="https://www.parliament.go.ke/contact/national_assembly_petition?bill=tax",
    year=2026,
)
SUMMARY = BillSummary(
    short_title="Tax Laws (Amendment) Bill, 2026",
    sponsor="Not stated",
    summary_en="Changes several tax laws.",
    summary_sw="Inabadilisha sheria kadhaa za kodi.",
    who_is_affected=["Taxpayers"],
    key_clauses=[],
    money_measures=[],
    counties_mentioned=[],
    confidence_notes="",
)


def test_upsert_is_idempotent(tmp_path):
    conn = store.connect(str(tmp_path / "db.sqlite"))
    first_id, new = store.upsert_bill(conn, BILL)
    again_id, again_new = store.upsert_bill(conn, BILL)
    assert new and not again_new and first_id == again_id


def test_drafts_stay_private_until_published(tmp_path):
    db = str(tmp_path / "db.sqlite")
    conn = store.connect(db)
    bill_id, _ = store.upsert_bill(conn, BILL)
    store.save_draft(conn, bill_id, SUMMARY, "claude-opus-5-5")
    api = TestClient(create_app(db))

    assert api.get(f"/bills/{bill_id}").json()["summary"] is None
    assert store.bills_needing_summary(conn, 10) == []

    assert store.publish(conn, bill_id)
    body = api.get(f"/bills/{bill_id}").json()
    assert body["summary"]["summary_sw"] == "Inabadilisha sheria kadhaa za kodi."
    assert api.get("/bills", params={"q": "Tax"}).json()[0]["id"] == bill_id
    assert api.get("/bills", params={"chamber": "senate"}).json() == []
    assert api.get("/bills/999").status_code == 404
