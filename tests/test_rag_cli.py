import json
from pathlib import Path

import pytest

from bunge_watch import cli

ROOT = Path(__file__).parent.parent


def test_eval_cli_writes_reports_and_enforces_threshold(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    args = ["eval", "--corpus", str(ROOT / "evals/fixtures/corpus.json"),
            "--questions", str(ROOT / "evals/questions.jsonl"), "--embedder", "fake", "--out", str(tmp_path)]
    cli.main(args + ["--retrieval-only", "--min-recall", "0.8"])
    reports = sorted(p.suffix for p in tmp_path.iterdir())
    assert reports == [".json", ".md"]
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["config"]["mode"] == "retrieval-only"
    assert "| recall@5 |" in next(tmp_path.glob("*.md")).read_text()

    with pytest.raises(SystemExit, match="below the threshold"):
        cli.main(args + ["--retrieval-only", "--k", "1", "--min-recall", "0.99"])
