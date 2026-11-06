# Ask the bills: clause-cited Q&A and its evaluation harness

**Status: day-1 scaffold.** The pipeline runs end to end and is tested offline, but retrieval quality on real
bills, the real embedding providers and the LLM judge have not been measured yet. The numbers that matter come
from the real golden set described below.

The goal is to answer questions such as *"Ni adhabu gani kwa kuchelewa kulipa ushuru wa maji?"* with an answer
that points at the exact clause and quotes it, and to **measure** how often that is right. This is not a chatbot:
there is no conversation, and the model only sees the clauses retrieval found.

## How it works

```
bills (SQLite) ──index --fetch──▶ bill_texts ──chunker──▶ clause chunks ──embedder──▶ rag_vectors (sqlite-vec)
                  (skips scans)                 (Clause N, PART,                       same SQLite file
                                                 Schedules, offsets)                           │
                                                                                               │ top-k
question (en / sw) ──embedder──────────────────────────────────────────────────────▶ cosine search
                                                                                               │
                                     Claude: answer ONLY from numbered sources ◀───────────────┘
                                     {found, answer, citations[{source, quote}]}
                                                │
                                     verify each quote is in its chunk ──▶ answered / not_found / unverified
                                                │                          + citations {bill_id, clause, quote}
                                     CLI `ask`  or  POST /ask (opt-in, reviewed: false)

evals/questions.jsonl ──▶ `bunge-watch eval` ──▶ recall@k, MRR, citation accuracy, quote validity,
                                                  refusal rates, judge scores ──▶ evals/reports/*.json + *.md
```

| Module | Does |
|---|---|
| `bunge_watch/rag/chunker.py` | One chunk per clause ("5. (1) ...", "Clause 5.", "Section 5:"), with the enclosing PART, Schedules kept whole, a Preamble chunk, char offsets into the text. Long clauses become overlapping windows with the same label. No structure: fixed 1,200-char windows with 200-char overlap ("Passage N"). |
| `bunge_watch/rag/embeddings.py` | `Embedder` protocol; `local` (multilingual-e5-small), `voyage`, and `fake` (hashed bag of words, tests only) |
| `bunge_watch/rag/index.py` | `bill_texts`, `rag_chunks`, `rag_bills`, `rag_meta` tables and a sqlite-vec `rag_vectors` table, all in the bills database. Idempotent per bill. |
| `bunge_watch/rag/qa.py` | Retrieval + Claude answer with structured output, then quote verification |
| `bunge_watch/rag/evals.py` | Golden-set loader, metrics, LLM judge, JSON + markdown reports |
| `bunge_watch/rag/routes.py`, `rag/cli.py` | `POST /ask`; `bunge-watch index`, `ask`, `eval` |

## Use it

```bash
pip install -e '.[dev,rag-local]'          # sentence-transformers; first run downloads the model (~470 MB)

bunge-watch scrape --max-pages 1
bunge-watch index --fetch --fetch-limit 5  # download PDFs, store their text, chunk and embed
bunge-watch index                          # safe to re-run: unchanged bills are skipped

export ANTHROPIC_API_KEY=...
bunge-watch ask "What is the penalty for late remittance of the levy?"
bunge-watch ask "Je, shule zinalipishwa ushuru wa maji?" --bill 12 --json

BUNGE_WATCH_ENABLE_ASK=1 bunge-watch serve # then POST /ask {"question": "...", "bill_id": 12, "k": 5}
```

`--embedder voyage` (with `pip install -e '.[rag-voyage]'` and `VOYAGE_API_KEY`) or `BUNGE_WATCH_EMBEDDER=voyage`
switches provider. Vectors from different embedders cannot be mixed: switching needs `bunge-watch index --rebuild`
(extracted text is kept, so nothing is downloaded again).

Scanned bills (no text layer) are recorded in `bill_texts` but not indexed; they come back once OCR exists
(roadmap item 6).

### What an answer looks like

```json
{
  "question": "What is the penalty for late remittance of the levy?",
  "status": "answered",
  "answer": "Five per cent of the unpaid amount for each month or part of a month it remains unpaid.",
  "citations": [{"bill_id": 9002, "clause": "Clause 5", "chunk_id": 14,
                 "quote": "a penalty of five per cent of the unpaid amount for each month"}],
  "rejected_citations": [],
  "sources": [{"bill_id": 9002, "clause": "Clause 5", "chunk_id": 14, "distance": 0.21}, "..."],
  "model": "claude-opus-5-5",
  "reviewed": false
}
```

`status` is `answered` (at least one verified citation), `not_found` (the retrieved clauses don't cover it), or
`unverified` (the model answered but none of its quotes appear in the sources: treat it as wrong).

## Evaluation

```bash
# Offline, as in CI: synthetic fixtures, fake embedder, retrieval metrics only
bunge-watch eval --corpus evals/fixtures/corpus.json --embedder fake --retrieval-only --min-recall 0.8

# Against your indexed database with the real embedder; answers and judge run if ANTHROPIC_API_KEY is set
bunge-watch eval --questions evals/real_questions.jsonl --k 5
bunge-watch eval --questions evals/real_questions.jsonl --no-judge   # answers without judge calls
```

Each run writes `evals/reports/eval-<timestamp>-<mode>.json` (config, metrics, every question's retrieved
clauses, answer, citations and judge rationale) and a `.md` summary table. Reports are gitignored.

### Golden-set format

One JSON object per line in `evals/questions.jsonl`:

```json
{"id": "water-late-penalty-en", "question": "What is the penalty for late remittance of the levy?",
 "language": "en", "bill_id": 9002, "expected_clauses": ["Clause 5"],
 "reference_answer": "Five per cent of the unpaid amount for each month ...", "answerable": true}
```

`language` is `en` or `sw`. `expected_clauses` lists every clause a correct answer may rely on; labels are
compared after normalising, so "Section 5", "clause 5" and "Kifungu 5" all match "Clause 5". Unanswerable
questions have `answerable: false`, `expected_clauses: []` and usually `bill_id: null`.

### Metrics

| Metric | Definition | Needs Claude |
|---|---|---|
| `recall@k` | Share of answerable questions where at least one top-k chunk is from the expected bill and an expected clause. Also reported per language (`recall@k_en`, `recall@k_sw`). | no |
| `mrr` | Mean of 1 / rank of the first such chunk (0 if none in top k). | no |
| `citation_accuracy` | Of the verified citations on answerable questions, the share whose bill and clause are expected. | yes |
| `citation_quote_validity` | Verified citations / all citations the model proposed. Low values mean the model paraphrases or invents quotes. | yes |
| `correct_refusal_rate` | Share of unanswerable questions answered `not_found`. | yes |
| `false_refusal_rate` | Share of answerable questions not `answered` (refused or unverified). | yes |
| `faithfulness` | Judge: are the answer's claims all in the retrieved sources? faithful 1, partially 0.5, unfaithful 0. | yes + judge |
| `correctness` | Judge: does the answer match the human reference? correct 1, partially 0.5, incorrect 0. An answerable question that was refused scores 0 without a judge call. | yes + judge |

API errors are recorded as `status: "error"`, counted in `errors`, and left out of the rates rather than scored as
wrong answers.

### The fixture set

`evals/fixtures/` holds three short **synthetic** bill excerpts (invented, labelled as such in every file, bill
ids 9001-9003) and `evals/questions.jsonl` has 12 questions over them: 7 English, 3 Kiswahili, 2 unanswerable.
One excerpt has no clause numbering, to exercise the fallback chunker. Its job is to catch regressions in the
plumbing (chunk labels, index, retrieval, metric code). It says nothing about real-world accuracy, and the fake
embedder carries a small Kiswahili-to-English glossary so the Kiswahili questions can retrieve at all.

### Building the real set (~50 questions)

1. **Pick 8-10 real bills** with a text layer, mixing chambers and types: a Finance/Tax Laws bill, an amendment
   bill ("Section 12 of Cap. 470 is amended..."), a long bill with PARTs and Schedules, a Senate county bill.
2. **Index them** with the embedder you plan to use, then run `bunge-watch index` and inspect `rag_chunks` for
   each bill (`SELECT label, substr(text,1,80) FROM rag_chunks WHERE bill_id = ?`). Fix the chunker before writing
   questions if labels are wrong; the golden set depends on them.
3. **Write questions from the bill, not from the system.** Read a clause, write the question a citizen would ask,
   note the clause label(s) and a one-sentence reference answer. Aim for about 50: ~30 English, ~15 Kiswahili
   (written by a fluent speaker, not machine-translated; include everyday words like *ushuru*, *faini*, *ada*),
   and ~8 unanswerable but plausible ones (about a bill that is not indexed, or a detail the bill omits).
4. **Include the hard cases:** answers spanning two clauses, questions about Schedules, amounts and dates,
   questions whose wording differs from the bill's ("boda boda" vs "motorcycle for hire").
5. **Have a second person check** every reference answer and expected clause against the PDF. Record who wrote
   and who checked each item (e.g. extra `"author"` / `"checked_by"` keys; the loader ignores keys it doesn't
   know).
6. **Keep it out of prompts.** Never tune the system prompt on specific golden questions; hold back ~10 questions
   that you only run before releases.
7. **Run it** (`bunge-watch eval --questions evals/real_questions.jsonl`). Each question costs one Claude Opus 5.5
   answer call plus one Claude Sonnet 5.5 judge call; run with `--no-judge` or `--retrieval-only` while iterating
   on retrieval. Re-run after any change to the chunker, embedder, `k`, or prompts, and compare reports.

Before trusting judge scores, calibrate the judge: hand-grade 20-30 answers, compare with its verdicts, and feed it
known negatives (an empty answer, "I don't know", a confident answer to a different question) to confirm it fails
them.

## Trade-offs

**sqlite-vec vs pgvector / Supabase.** sqlite-vec keeps vectors in the same file as the bills: no server, no
second datastore to back up, transactions span chunks and vectors so re-indexing is atomic, and CI needs nothing
but a pip install. Search is exact (brute-force) KNN, which is fast at our scale: Kenya publishes a few hundred
bills a year, a few thousand chunks per year, well under what brute force handles in milliseconds. pgvector (or
Supabase) becomes the better choice if the API runs as several processes that write concurrently, if we need ANN
indexes (HNSW) for millions of chunks (county assemblies, Hansard), or if we need hybrid search with Postgres
full-text. The `index.py` surface (`index_bill`, `search`) is small enough to port. One cost of sqlite-vec: it is a
loadable extension, and some Python builds (notably macOS system Python) cannot load extensions.

**Embedding model for Kiswahili.** Anthropic does not offer embeddings; its documentation points to Voyage AI,
which is wired in as `--embedder voyage`. The default is `intfloat/multilingual-e5-small` instead, because:
- Bills are English but questions will often be Kiswahili, so retrieval must be cross-lingual. The multilingual
  E5 models were trained on about 100 languages, and the evaluations reported for them include MIRACL, a retrieval
  benchmark that covers Swahili. That is evidence, not proof: our own Kiswahili golden questions are the test.
- It runs on a CPU, costs nothing per query, and bill text and citizens' questions never leave the server.
- It is small (118M parameters, 384 dimensions). `multilingual-e5-base`/`-large` or `BAAI/bge-m3` are drop-in
  upgrades via `--embedding-model` if the eval shows Kiswahili recall is weak; `bge-m3` also takes 8k-token inputs.
Limits: e5-small reads at most 512 tokens, so very long chunks are truncated (clause chunks are capped at 2,000
characters, roughly 400-500 tokens of English, to stay under it). Legal Kiswahili vocabulary is thin in any
model's training data. Compare `recall@k_sw` with `recall@k_en` on the real set and switch models if the gap is
large.

**Clause-aware vs fixed-size chunking.** Clause chunks make citations exact (a chunk *is* a clause, so the label
is the citation) and keep a provision's conditions and exceptions together. The costs: the heading regexes are
heuristics tuned to Kenyan drafting (numbered clauses, "Clause N", PART, Schedules) and can misfire on bad PDF
text; short clauses carry little context (a clause that says "contravenes section 5" doesn't say what section 5
requires, which is why the helmet fine question needs both clauses); and long clauses still need windowing.
Numbers that don't increase are not treated as new clauses, so lists inside a clause and the Memorandum's own
numbering stay put. Fixed-size windows are the fallback for unstructured text and are labelled "Passage N", which
is a weaker citation.

**Citation verification.** The model cites a numbered source and a verbatim quote; the bill and clause come from
the index, so the model cannot invent a clause number. Each quote is matched against the retrieved chunk text
after normalising whitespace, case, curly quotes, dashes and `[page N]` markers. This catches invented and
paraphrased quotes. It does **not** prove the quote supports the answer (a real quote can be attached to a wrong
claim), which is what the faithfulness judge and human spot checks are for. Exact substring matching also rejects
quotes with harmless differences (an OCR ligature, a dropped word), so watch `citation_quote_validity` before
loosening it.

**LLM-as-judge limitations.** The judge is a different model (Claude Sonnet 5.5) from the one answering (Claude
Opus 5.5) to reduce self-preference, but it is the same family, so some bias remains. Judges reward fluent length
(the rubric tells it not to), can be swayed by text inside the answer, and are non-deterministic, so small score
changes between runs are noise. Its verdicts are only as good as the human reference answers. Treat judge scores
as a trend line; the programmatic metrics (recall, MRR, citation accuracy, quote validity, refusal rates) are the
ones to gate on.

**Live answers vs human review.** Published summaries are always reviewed by a person; `/ask` answers cannot be.
That is why `/ask` is off unless `BUNGE_WATCH_ENABLE_ASK=1`, every answer is marked `reviewed: false`, answers
without a verified quote are not reported as answered, and the CLI tells readers to check the quoted clause.

**Prompt caching.** The QA and judge system prompts are fixed and marked for caching, with the per-question
sources after them. The QA prompt is currently below the model's minimum cacheable length, so it only starts
caching if the instructions grow; retrieved sources differ per question and are not worth caching.

## Not done yet

- Real-bill golden set and a baseline report with the real embedder.
- Hybrid retrieval (BM25 over `rag_chunks` via SQLite FTS5) for exact terms such as "Cap. 470" or amounts.
- Following cross-references ("contravenes section 5") to pull in the referenced clause.
- Page numbers in citations (offsets are stored; mapping them to `[page N]` markers is straightforward).
- Rate limiting and cost caps on `/ask` before it is enabled publicly.
