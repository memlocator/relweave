"""The generator behind an HTTP server: a vLLM OpenAI-compatible endpoint serving the merged generator (the model
repository's root: `vllm serve chrullis/relweave-4b-base`). Prompt, grammar and parsing are the same as the local
backends; the server batches the concurrent requests. Only the standard library is used.

Stop on repeat is applied after decoding (relweave.vllm_generator.drop_repeats), as with in-process vLLM.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from relweave.formats import FORMATS, STUDENT_FORMAT, ExtractionResult, chat_messages, finish, generation_prompt
from relweave.schema import BUSINESS, Schema
from relweave.vllm_generator import drop_repeats


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310  (the caller's own server URL)
        return json.loads(r.read())


def served_model(base_url: str, timeout: float = 30) -> str:
    """The id of the first model the server lists (a vLLM server serves one)."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read())["data"][0]["id"]


class RemoteExtractor:
    """extract_batch(items) like QwenExtractor, over HTTP; `concurrency` requests in flight (the server batches)."""

    def __init__(self, base_url: str, tokenizer_dir: Path, model: str | None = None, max_new_tokens: int = 2500,
                 fmt: str = STUDENT_FORMAT, compact: bool = False, schema: Schema | None = None,
                 conditioned: bool = False, stop_on_repeat: bool = True, concurrency: int = 32,
                 timeout: float = 600):
        from transformers import AutoTokenizer
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model or served_model(base_url)
        self.fmt, self.schema, self.compact, self.conditioned = fmt, schema or BUSINESS, compact, conditioned
        self.format = FORMATS[fmt].with_schema(self.schema)
        self.max_new_tokens, self.stop_on_repeat = max_new_tokens, stop_on_repeat
        self.concurrency, self.timeout = concurrency, timeout
        self.name = f"qwen_finetuned_{fmt}_remote"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))  # the chat template the generator was trained with

    def _one(self, item: tuple[str, str]) -> ExtractionResult:
        cid, text = item
        prompt = generation_prompt(self.tokenizer, chat_messages(text, self.fmt, compact=self.compact,
                                                                 schema=self.schema, conditioned=self.conditioned))
        t0 = time.perf_counter()
        body = {"model": self.model, "prompt": prompt, "max_tokens": self.max_new_tokens, "temperature": 0.0,
                "structured_outputs": {"grammar": self.format.ebnf_for(self.format.grammar_key(text))}}
        raw = _post(self.url, body, self.timeout)["choices"][0]["text"]
        return finish(cid, text, drop_repeats(raw) if self.stop_on_repeat else raw, time.perf_counter() - t0,
                      fmt=self.fmt, schema=self.schema)

    def extract_batch(self, items: list[tuple[str, str]], batch_size: int = 0) -> list[ExtractionResult]:
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(self._one, items))
