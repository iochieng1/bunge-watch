# Bunge Watch

Kenya's Constitution (Articles 10, 118 and 196) gives every citizen a right to take part in law-making. But bills
are published as long English PDFs, many of them scans, with short windows for public comment. Bunge Watch tracks
every bill in the National Assembly and the Senate and explains each one in **plain English and Kiswahili**: what
it does, who it affects, the clauses that matter, and any new taxes, levies or penalties. Each summary links
straight to Parliament's own "Submit Comments" page.

> **Every summary is reviewed by a person before it is published.** Claude writes a draft, a reviewer checks it
> against the bill, and only then does it appear in the public API.

## How it works

```
parliament.go.ke ──scrape──▶ SQLite ──PDF──▶ text layer? ──yes──▶ extracted text ─┐
 (NA + Senate bills)         (bills)            │                                 ├──▶ Claude ──▶ draft summary
                                                └──no (scan)──▶ PDF document ─────┘                      │
                                                                                       human review ◀────┘
                                                                                            │ publish
                                                                              public API ◀──┘
```

| Module | Does |
|---|---|
| `bunge_watch/scraper.py` | Reads the bills listings on parliament.go.ke (both chambers, all pages), cleans up inconsistent titles, records PDF size, upload month and the comments link |
| `bunge_watch/pdftext.py` | Downloads a bill and pulls its text out, flagging scans that have no text layer |
| `bunge_watch/summarizer.py` | Asks Claude (`claude-opus-5-5`) for a structured, neutral summary with clause references, in English and Kiswahili |
| `bunge_watch/store.py` | SQLite storage; summaries are `draft` until published |
| `bunge_watch/api.py` | Read-only FastAPI service that only ever returns published summaries |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

bunge-watch scrape --max-pages 1        # fetch the newest bills from both chambers
export ANTHROPIC_API_KEY=...            # needed only for summarize
bunge-watch summarize --limit 2         # draft summaries for 2 bills
bunge-watch review                      # read the drafts
bunge-watch publish 3                   # publish bill #3 after checking it
bunge-watch serve                       # http://127.0.0.1:8000/bills  (docs at /docs)
```

`summarize` calls the Claude API, which costs money per bill. A scanned 50-page bill sent as a PDF costs more than
one with a text layer. Start with `--limit 1`.

## API

| Endpoint | Returns |
|---|---|
| `GET /bills?chamber=senate&q=finance` | All tracked bills, newest first, with the published summary or `null` |
| `GET /bills/{id}` | One bill |
| `GET /health` | `{"status": "ok"}` |

## Tests

```bash
pytest
```

The scraper tests run against a saved excerpt of the real National Assembly bills page
(`tests/fixtures/national_assembly_bills.html`), so changes to Parliament's markup show up as failing tests. The
Claude client is faked in tests, so they cost nothing.

## Design choices

- **Neutral by construction.** The prompt forbids judging a bill or naming politicians other than the sponsor, and
  every key point must cite a clause.
- **Human in the loop.** Model output is never published automatically, because a wrong summary of a law does real
  harm.
- **Scans are handled.** Many bills (e.g. *The Tax Laws (Amendment) Bill, 2026*) are scanned images. These go to
  Claude as PDF documents; files over 32 MB are skipped with a message asking for OCR.
- **Safety fallback.** Requests opt into the API's server-side `fallbacks: "default"`, so a classifier decline is
  retried on Anthropic's recommended fallback model instead of failing.

## Related

[Public-Participation-Platform](https://github.com/iochieng1/Public-Participation-Platform) is the citizen-facing
side: web, SMS and USSD submissions with feedback. Bunge Watch is the watchdog side: it finds the bills and explains
them. See [docs/ROADMAP.md](docs/ROADMAP.md) for how they connect.

## License

MIT
