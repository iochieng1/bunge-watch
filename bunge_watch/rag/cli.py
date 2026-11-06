"""`bunge-watch index`, `bunge-watch ask` and `bunge-watch eval`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .embeddings import PROVIDERS, get_embedder

DEFAULT_EMBEDDER = os.environ.get("BUNGE_WATCH_EMBEDDER", "local")


def _add_embedder_args(p: argparse.ArgumentParser, default: str = DEFAULT_EMBEDDER) -> None:
    p.add_argument("--embedder", choices=PROVIDERS, default=default, help="embedding provider (default: %(default)s)")
    p.add_argument("--embedding-model", help="override the provider's default model")


def _embedder(args: argparse.Namespace):
    try:
        return get_embedder(args.embedder, args.embedding_model)
    except RuntimeError as err:  # optional provider not installed
        sys.exit(str(err))


def cmd_index(args: argparse.Namespace) -> None:
    from .. import pdftext, scraper
    from . import index

    conn = index.connect(args.db)
    if args.fetch:
        missing = index.bills_missing_text(conn, args.fetch_limit)
        with scraper.make_client() as http:
            for row in missing:
                print(f"Extracting text: {row['title']} ...")
                text, scanned = pdftext.extract_text(pdftext.download_pdf(http, row["pdf_url"]))
                index.save_text(conn, row["id"], text, scanned)
                if scanned:
                    print("  looks like a scan with no text layer; it will not be indexed until OCR exists")
    if args.rebuild:
        index.reset(conn)
    embedder = _embedder(args)
    rows = index.indexable_bills(conn)
    if args.bill is not None:
        rows = [r for r in rows if r["bill_id"] == args.bill]
    if not rows:
        print("No bills with extracted text to index. Run `bunge-watch index --fetch` after `bunge-watch scrape`.")
        return
    counts: dict[str, int] = {}
    for row in rows:
        try:
            outcome = index.index_bill(conn, embedder, row["bill_id"], row["text"])
        except index.IndexMismatch as err:
            sys.exit(str(err))
        counts[outcome] = counts.get(outcome, 0) + 1
    print(f"Indexed with {embedder.name}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))


def cmd_ask(args: argparse.Namespace) -> None:
    import anthropic

    from . import index
    from .qa import QAError, ask

    conn = index.connect(args.db)
    embedder = _embedder(args)
    try:
        answer = ask(anthropic.Anthropic(), conn, embedder, args.question, k=args.k, bill_id=args.bill)
    except (QAError, index.IndexMismatch) as err:
        sys.exit(f"Could not answer: {err}")
    if args.json:
        print(answer.model_dump_json(indent=2))
        return
    print(f"[{answer.status}] {answer.answer}")
    for c in answer.citations:
        print(f'  - bill #{c.bill_id}, {c.clause}: "{c.quote}"')
    for r in answer.rejected_citations:
        print(f'  ! dropped citation ({r.reason}): "{r.quote}"', file=sys.stderr)
    print("Generated answer, not reviewed by a person. Check the quoted clauses in the bill itself.")


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evals, index

    embedder = _embedder(args)
    if args.corpus:
        conn = evals.load_fixture_corpus(args.corpus, embedder)
    else:
        conn = index.connect(args.db)
    questions = evals.load_questions(args.questions)

    client = judge_client = None
    mode = "retrieval-only"
    if not args.retrieval_only:
        if evals.has_claude_credentials():
            import anthropic

            client = anthropic.Anthropic()
            judge_client = None if args.no_judge else client
            mode = "full" if judge_client else "answers"
        else:
            print("No ANTHROPIC_API_KEY set: running retrieval metrics only.", file=sys.stderr)

    metrics, results = evals.run_eval(
        conn, embedder, questions, k=args.k, client=client, judge_client=judge_client,
        judge_model=args.judge_model, filter_by_bill=args.filter_by_bill,
    )
    config = evals.make_config(embedder, args.k, mode, args.questions, corpus=args.corpus or args.db,
                               judge_model=args.judge_model if judge_client else None)
    json_path, md_path = evals.write_report(args.out, metrics, results, config)
    print(json.dumps(metrics, indent=2))
    print(f"Report: {json_path} and {md_path}")

    recall = metrics.get(f"recall@{args.k}")
    if args.min_recall is not None and (recall is None or recall < args.min_recall):
        sys.exit(f"recall@{args.k} = {recall} is below the threshold {args.min_recall}")


def add_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("index", help="chunk and embed bills with extracted text for Q&A")
    p.add_argument("--fetch", action="store_true", help="download and extract text for bills that have none yet")
    p.add_argument("--fetch-limit", type=int, help="extract at most this many bills")
    p.add_argument("--bill", type=int, help="only (re)index this bill id")
    p.add_argument("--rebuild", action="store_true", help="drop all chunks and vectors first (needed to switch embedder)")
    _add_embedder_args(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer a question from the indexed bills, citing clauses")
    p.add_argument("question")
    p.add_argument("--bill", type=int, help="only search this bill id")
    p.add_argument("--k", type=int, default=5, help="chunks to retrieve (default: %(default)s)")
    p.add_argument("--json", action="store_true", help="print the full structured answer")
    _add_embedder_args(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="measure retrieval and answer accuracy on a golden question set")
    p.add_argument("--questions", default="evals/questions.jsonl")
    p.add_argument("--corpus", help="fixture manifest to index in memory (e.g. evals/fixtures/corpus.json);"
                                    " omit to evaluate against the indexed --db")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval-only", action="store_true", help="skip Claude: retrieval metrics only")
    p.add_argument("--no-judge", action="store_true", help="generate answers but skip the LLM judge")
    p.add_argument("--judge-model", default="claude-sonnet-5-5")
    p.add_argument("--filter-by-bill", action="store_true", help="restrict retrieval to each question's bill")
    p.add_argument("--min-recall", type=float, help="exit non-zero if recall@k is below this")
    p.add_argument("--out", default="evals/reports")
    _add_embedder_args(p)
    p.set_defaults(func=cmd_eval)
