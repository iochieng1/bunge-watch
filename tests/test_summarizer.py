import json
from types import SimpleNamespace

import pytest

from bunge_watch.summarizer import MAX_PDF_BYTES, BillSummary, SummaryError, summarize_bill

SUMMARY = {
    "short_title": "Tax Laws (Amendment) Bill, 2026",
    "sponsor": "Not stated",
    "summary_en": "Changes several tax laws.",
    "summary_sw": "Inabadilisha sheria kadhaa za kodi.",
    "who_is_affected": ["Taxpayers"],
    "key_clauses": [{"reference": "Clause 3", "plain_english": "Raises a levy.", "plain_kiswahili": "Inaongeza ushuru."}],
    "money_measures": ["Levy raised"],
    "counties_mentioned": [],
    "confidence_notes": "",
}


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def response(stop_reason="end_turn", text=json.dumps(SUMMARY)):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


def test_summarize_from_text_returns_validated_summary():
    client = FakeClient(response())
    summary = summarize_bill(client, "The Tax Laws (Amendment) Bill, 2026", text="[page 1]\nA Bill for ...")

    assert isinstance(summary, BillSummary)
    assert summary.key_clauses[0].reference == "Clause 3"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "<bill>" in call["messages"][0]["content"][0]["text"]


def test_scanned_pdf_is_sent_as_document():
    client = FakeClient(response())
    summarize_bill(client, "Scanned Bill", pdf=b"%PDF-1.7 fake")
    assert client.calls[0]["messages"][0]["content"][0]["type"] == "document"


def test_oversized_scan_needs_ocr():
    with pytest.raises(SummaryError, match="OCR"):
        summarize_bill(FakeClient(response()), "Huge Bill", pdf=b"0" * (MAX_PDF_BYTES + 1))


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses_raise(stop_reason):
    with pytest.raises(SummaryError):
        summarize_bill(FakeClient(response(stop_reason=stop_reason)), "Bill", text="...")
