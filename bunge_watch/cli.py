"""bunge-watch command line: scrape, summarize, review, publish, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import pdftext, scraper, store

DEFAULT_DB = os.environ.get("BUNGE_WATCH_DB", "bunge_watch.db")


def cmd_scrape(args: argparse.Namespace) -> None:
    conn = store.connect(args.db)
    chambers = list(scraper.LISTING_URLS) if args.chamber == "all" else [args.chamber]
    with scraper.make_client() as client:
        for chamber in chambers:
            bills = scraper.scrape_chamber(client, chamber, max_pages=args.max_pages)
            new = [b for b in bills if store.upsert_bill(conn, b)[1]]
            print(f"{chamber}: {len(bills)} bills listed, {len(new)} new")
            for bill in new:
                print(f"  + {bill.title}")


def cmd_summarize(args: argparse.Namespace) -> None:
    import anthropic

    from .summarizer import MODEL, SummaryError, summarize_bill

    conn = store.connect(args.db)
    claude = anthropic.Anthropic()
    with scraper.make_client() as http:
        for row in store.bills_needing_summary(conn, args.limit):
            print(f"Summarising {row['title']} ...")
            pdf = pdftext.download_pdf(http, row["pdf_url"])
            text, scanned = pdftext.extract_text(pdf)
            try:
                summary = summarize_bill(claude, row["title"], text=None if scanned else text, pdf=pdf if scanned else None)
            except SummaryError as err:
                print(f"  skipped: {err}", file=sys.stderr)
                continue
            store.save_draft(conn, row["id"], summary, MODEL)
            print("  saved as draft; review with `bunge-watch review`")


def cmd_review(args: argparse.Namespace) -> None:
    conn = store.connect(args.db)
    rows = store.drafts(conn)
    if not rows:
        print("No drafts waiting for review.")
    for row in rows:
        print(f"\n#{row['id']}  {row['title']}")
        print(json.dumps(json.loads(row["summary_json"]), indent=2, ensure_ascii=False))


def cmd_publish(args: argparse.Namespace) -> None:
    conn = store.connect(args.db)
    if store.publish(conn, args.bill_id):
        print(f"Published summary for bill #{args.bill_id}.")
    else:
        sys.exit(f"No draft summary for bill #{args.bill_id}.")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bunge-watch", description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite file (default: %(default)s)")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("scrape", help="fetch new bills from parliament.go.ke")
    p.add_argument("--chamber", choices=["all", *scraper.LISTING_URLS], default="all")
    p.add_argument("--max-pages", type=int, help="stop after this many listing pages per chamber")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("summarize", help="draft summaries for bills that have none")
    p.add_argument("--limit", type=int, default=3)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("review", help="print draft summaries waiting for review")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("publish", help="publish a reviewed draft summary")
    p.add_argument("bill_id", type=int)
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("serve", help="run the public read-only API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
