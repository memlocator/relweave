"""Typed relations on vLLM without vLLM: the continuation request and the unfinished-request check."""

import sys
import types
from types import SimpleNamespace as NS

import pytest

from relweave.vllm_generator import BLOCK_START, VllmExtractor, _check_finished


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def _out(text, token_ids, finish="stop", prompt_ids=(1, 2), stop_reason=None):
    return NS(prompt_token_ids=list(prompt_ids),
              outputs=[NS(text=text, token_ids=list(token_ids), finish_reason=finish, stop_reason=stop_reason)])


@pytest.fixture
def fake_vllm(monkeypatch):
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def _extractor(answers):
    ex = object.__new__(VllmExtractor)
    ex.tokenizer, ex.max_new_tokens, ex.lora = FakeTokenizer(), 100, None
    ex._params = lambda text, entity_types=None, max_tokens=None: entity_types
    sent = []

    def generate(reqs, params, lora_request=None, use_tqdm=False):
        sent.extend(reqs)
        return answers

    ex.llm = NS(generate=generate)
    return ex, sent


def test_the_relation_request_continues_from_the_entity_lines(fake_vllm):
    ex, sent = _extractor([_out("S1\nR X E1 E2 asserted\n", [7])])
    # the first request stopped at the first block; vLLM leaves the stop string out of the text
    first = [_out("E1 Person: Ann\nE2 Org: Acme", [5, 6, 7], stop_reason=BLOCK_START)]
    texts = ex._typed_relations([("c1", "text", None)], first, frozenset())
    assert sent == [{"ids": [1, 2, *(ord(c) for c in "E1 Person: Ann\nE2 Org: Acme\n")]}]
    assert texts == ["E1 Person: Ann\nE2 Org: Acme\nS1\nR X E1 E2 asserted\n"]


def test_a_chunk_that_ended_without_a_block_keeps_its_answer(fake_vllm):
    ex, sent = _extractor([])
    first = [_out("E1 Person: Ann\n", [5])]
    assert ex._typed_relations([("c1", "text", None)], first, frozenset()) == ["E1 Person: Ann\n"]
    assert sent == []


def test_an_unfinished_request_raises():
    _check_finished([_out("E1 Person: Ann\n", [5], finish="length")])
    with pytest.raises(RuntimeError, match="did not finish 1 of 2"):
        _check_finished([_out("E1 Person: Ann\n", [5]), _out("", [], finish="abort")])
