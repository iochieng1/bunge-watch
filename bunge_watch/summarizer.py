"""Plain-language English and Kiswahili summaries of bills, written by Claude.

Every summary is saved as a draft. A person reviews it before it is published (see `bunge-watch publish`).
"""

from __future__ import annotations

import base64

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-5-5"
# Claude API PDF limit; bigger scanned bills need OCR first.
MAX_PDF_BYTES = 32 * 1024 * 1024

SYSTEM_PROMPT = """\
You explain Kenyan bills to ordinary citizens for Bunge Watch, a non-partisan civic project.

Write for a reader with secondary-school education who has never read a bill. Use short sentences and
everyday words. Be strictly neutral: describe what the bill does, never whether it is good or bad, and
never mention politicians or parties except as the bill's sponsor.

Only state what the bill text says. Cite clause numbers for every key point. If something (sponsor,
commencement, cost) is not in the text, write "Not stated". Money measures means taxes, levies, fees,
fines and penalties, with amounts in KES as written.

Write the Kiswahili as a fluent Kenyan speaker would say it, not a word-for-word translation.
Use confidence_notes for anything you could not read or are unsure about (e.g. unreadable scanned pages).
"""


class Clause(BaseModel):
    reference: str
    plain_english: str
    plain_kiswahili: str


class BillSummary(BaseModel):
    short_title: str
    sponsor: str
    summary_en: str
    summary_sw: str
    who_is_affected: list[str]
    key_clauses: list[Clause]
    money_measures: list[str]
    counties_mentioned: list[str]
    confidence_notes: str


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "short_title": _STR,
        "sponsor": _STR,
        "summary_en": _STR,
        "summary_sw": _STR,
        "who_is_affected": _STR_LIST,
        "key_clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"reference": _STR, "plain_english": _STR, "plain_kiswahili": _STR},
                "required": ["reference", "plain_english", "plain_kiswahili"],
                "additionalProperties": False,
            },
        },
        "money_measures": _STR_LIST,
        "counties_mentioned": _STR_LIST,
        "confidence_notes": _STR,
    },
    "required": list(BillSummary.model_fields),
    "additionalProperties": False,
}


class SummaryError(RuntimeError):
    pass


def _bill_content(title: str, text: str | None, pdf: bytes | None) -> list[dict]:
    instruction = {"type": "text", "text": f"Summarise this bill: {title}"}
    if text is not None:
        return [{"type": "text", "text": f"<bill>\n{text}\n</bill>"}, instruction]
    if pdf is None:
        raise ValueError("Pass the bill's extracted text or its PDF bytes.")
    if len(pdf) > MAX_PDF_BYTES:
        raise SummaryError(f"Scanned PDF is {len(pdf) // 2**20} MB; run OCR first (limit is 32 MB).")
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf).decode()},
    }
    return [document, instruction]


def summarize_bill(
    client: anthropic.Anthropic,
    title: str,
    *,
    text: str | None = None,
    pdf: bytes | None = None,
    model: str = MODEL,
) -> BillSummary:
    """Summarise from extracted text when the PDF has a text layer, otherwise from the PDF itself."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _bill_content(title, text, pdf)}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
        # If a safety classifier declines, the API retries on Anthropic's recommended fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise SummaryError(f"Model declined to summarise {title!r}: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise SummaryError(f"Summary of {title!r} was cut off at max_tokens.")
    texts = [block.text for block in response.content if block.type == "text"]
    if not texts:
        raise SummaryError(f"No summary returned for {title!r}.")
    return BillSummary.model_validate_json(texts[-1])
