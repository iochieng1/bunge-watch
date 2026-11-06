import json
from types import SimpleNamespace

import pytest

from bunge_watch.rag.index import Retrieved
from bunge_watch.rag.qa import QAError, generate_answer, normalize_text, verify_citations

CLAUSE_2 = Retrieved(11, 9002, "Clause 2", None, 0, 100,
                     "Clause 2. Water conservation levy\n(1) There shall be charged a water conservation levy of two "
                     "per cent of the amount billed", 0.1)
CLAUSE_3 = Retrieved(12, 9002, "Clause 3", None, 100, 200,
                     "Clause 3. Exemptions\nThe levy shall not be charged on water supplied to households, public "
                     "schools, public hospitals or places of worship.", 0.2)


class FakeClient:
    def __init__(self, payload, stop_reason="end_turn"):
        self.calls = []
        self._payload, self._stop = payload, stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self._stop, stop_details=None, model=kwargs["model"],
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=json.dumps(self._payload))],
        )


def test_quotes_are_checked_against_retrieved_text():
    raw = [
        {"source": 1, "quote": "a water conservation levy of two\nper cent"},           # whitespace differs: ok
        {"source": 1, "quote": "public hospitals or places of worship"},                # wrong source: moved
        {"source": 2, "quote": "The levy is three per cent"},                           # invented: rejected
        {"source": 9, "quote": "x"},                                                    # too short: rejected
    ]
    verified, rejected = verify_citations(raw, [CLAUSE_2, CLAUSE_3])
    assert [(c.bill_id, c.clause) for c in verified] == [(9002, "Clause 2"), (9002, "Clause 3")]
    assert [r.reason for r in rejected] == ["quote not found in sources", "quote too short to check"]


def test_normalize_ignores_page_markers_and_curly_quotes():
    assert normalize_text("the “levy” [page 3]\n shall") == 'the "levy" shall'


def test_answer_with_verified_citations():
    client = FakeClient({"found": True, "answer": "Two per cent.",
                         "citations": [{"source": 1, "quote": "levy of two per cent of the amount billed"}]})
    answer = generate_answer(client, "What is the levy rate?", [CLAUSE_2, CLAUSE_3])
    assert answer.status == "answered"
    assert answer.citations[0].clause == "Clause 2"
    assert answer.reviewed is False

    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert '<source id="1" bill_id="9002" clause="Clause 2">' in call["messages"][0]["content"]


def test_refusal_path_when_sources_do_not_answer():
    client = FakeClient({"found": False, "answer": "The bills provided do not cover fuel prices.", "citations": []})
    answer = generate_answer(client, "Does this raise fuel prices?", [CLAUSE_2])
    assert answer.status == "not_found"
    assert answer.citations == []


def test_answer_with_only_invented_quotes_is_unverified():
    client = FakeClient({"found": True, "answer": "Three per cent.",
                         "citations": [{"source": 1, "quote": "a levy of three per cent"}]})
    answer = generate_answer(client, "What is the levy rate?", [CLAUSE_2])
    assert answer.status == "unverified"
    assert answer.citations == [] and len(answer.rejected_citations) == 1


def test_no_retrieved_chunks_means_not_found_without_calling_claude():
    client = FakeClient({})
    assert generate_answer(client, "Anything?", []).status == "not_found"
    assert client.calls == []


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_responses_raise(stop_reason):
    with pytest.raises(QAError):
        generate_answer(FakeClient({}, stop_reason=stop_reason), "Q?", [CLAUSE_2])
