"""Output formats for the extractor: how a chunk is shown to the model, what the model
emits, how that is decoded, and the decoding grammar.

Gold labels and results are always stored as ExtractionOutput (JSON on disk). A format
only changes the surface the model reads and writes. Every decoded output goes through
the same lenient validator and span resolver, so formats are directly comparable.

`lines`:

    input   S1: Maria Lind, chief executive of Acme Robotics AB, said ...
            S2: She added that Acme may open an office in Luleå.
    output  E1 Person: Maria Lind | She
            E2 Org: Acme Robotics AB | the company | Acme
            R EXECUTIVE_OF E1 E2 asserted S1 title=chief executive
            R OPERATES_IN E2 E4 hedged S2

Evidence is a sentence reference, so the model copies nothing but mention strings.
An entity's name is its longest surface.

`sentences` (the generator's format): like `lines`, but relations are grouped under sentence headers that must
appear in increasing order, each at most once:

            E1 Person: Maria Lind | She
            E2 Org: Acme Robotics AB | the company | Acme
            S1
            R EXECUTIVE_OF E1 E2 asserted title=chief executive
            S2
            R OPERATES_IN E2 E4 hedged

The grammar is built per chunk from its sentence count, so the model walks the text
once and must stop after the last sentence. In `lines`, the free sentence number on
every relation invited the same counter loop that broke per-mention occurrence indices.

`json`: the original format, one JSON object with verbatim evidence sentences (the teacher prompts use it).

The second half of the module turns a model answer into a validated ExtractionOutput (finish: decode, validate
under the schema, verify every mention and evidence string against the chunk) and stores results as JSONL.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from relweave.schema import BUSINESS, SCHEMAS, Schema
from relweave.schema.output import (
    MAX_ENTITIES_PER_CHUNK,
    MAX_MENTIONS_PER_ENTITY,
    MAX_RELATIONS_PER_CHUNK,
    ExtractionOutput,
    Modality,
    json_schema,
    parse_lenient,
    schema_description,
)

# --------------------------------------------------------------------------
# Sentences of a chunk (the S1, S2, ... the line formats number)
# --------------------------------------------------------------------------

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
# a full stop after one of these is not a sentence end: a single-letter initial ("Johan W. Arnberg")
# or a title or abbreviation ("St. Andrew's", "Mr. Eberth", "lit. 'The Swedish'")
NO_SPLIT_AFTER = re.compile(r"(?:(?<![\w.])[A-Z]|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Gen|Col|Lt|Mt|Rev|lit|approx|vs|No|no))\.$")


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for m in SENTENCE_RE.finditer(text):
        if NO_SPLIT_AFTER.search(text[max(0, m.start() - 8):m.start()]):
            continue
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    return [(s, e) for s, e in bounds if text[s:e].strip()]


def sentence_windows(text: str, max_chars: int, overlap_sentences: int = 1) -> list[tuple[int, int]]:
    """(start, end) of sentence-aligned windows of at most max_chars (a longer single sentence is its own window),
    consecutive windows sharing overlap_sentences sentences."""
    sents = sentence_bounds(text)
    out, i = [], 0
    while i < len(sents):
        j = i
        while j < len(sents) and sents[j][1] - sents[i][0] <= max_chars:
            j += 1
        j = max(j, i + 1)
        out.append((sents[i][0], sents[j - 1][1]))
        if j >= len(sents):
            break
        i = max(j - overlap_sentences, i + 1)
    return out


PREAMBLE = (
    "You extract a typed knowledge graph from one text chunk. Entities group every mention "
    "of the same real-world thing, including pronouns and descriptions such as 'the company'. "
    "List each distinct surface string of an entity once, copied verbatim from the text; every "
    "occurrence of that string counts as a mention. Every relation must be legal for the entity "
    "types and must carry the modality the text expresses. Extract what the text states, even "
    "if it seems implausible."
)

EXAMPLE_CHUNK = (
    "Maria Lind, chief executive of Acme Robotics AB, said the company would keep its "
    "Gothenburg headquarters. She added that Acme may open an office in Luleå."
)
EXAMPLE_OUTPUT = {
    "entities": [
        {"id": "e1", "type": "Person", "name": "Maria Lind", "attributes": {},
         "mentions": ["Maria Lind", "She"]},
        {"id": "e2", "type": "Org", "name": "Acme Robotics AB", "attributes": {},
         "mentions": ["Acme Robotics AB", "the company", "Acme"]},
        {"id": "e3", "type": "Place", "name": "Gothenburg", "attributes": {},
         "mentions": ["Gothenburg"]},
        {"id": "e4", "type": "Place", "name": "Luleå", "attributes": {},
         "mentions": ["Luleå"]},
    ],
    "relations": [
        {"type": "EXECUTIVE_OF", "source": "e1", "target": "e2", "attributes": {"title": "chief executive"},
         "modality": "asserted",
         "evidence": "Maria Lind, chief executive of Acme Robotics AB, said the company would keep its Gothenburg headquarters."},
        {"type": "HEADQUARTERED_IN", "source": "e2", "target": "e3", "attributes": {}, "modality": "asserted",
         "evidence": "Maria Lind, chief executive of Acme Robotics AB, said the company would keep its Gothenburg headquarters."},
        {"type": "OPERATES_IN", "source": "e2", "target": "e4", "attributes": {}, "modality": "hedged",
         "evidence": "She added that Acme may open an office in Luleå."},
    ],
}

# A reference to a sentence that does not exist decodes to this evidence string, which
# never occurs in a chunk, so the resolver counts it as unverifiable evidence.
MISSING_EVIDENCE = "\x00missing sentence\x00"


class Format(Protocol):
    name: str
    schema: Schema

    def with_schema(self, schema: Schema) -> Format: ...

    def system_prompt(self) -> str: ...
    def render_input(self, text: str) -> str: ...
    def encode(self, out: ExtractionOutput, text: str) -> str: ...
    def decode(self, raw: str, text: str) -> tuple[dict | None, str | None]: ...
    def grammar_key(self, text: str): ...
    def compile(self, compiler, text: str = ""): ...


# System prompt for a fine-tuned student trained with compact=True. The full prompt (schema
# text plus worked example) is about 1,000 tokens, 70% of an average training example; a
# fine-tuned model has learned the schema and format, and the decoding grammar enforces the
# format, so the compact prompt cuts training and extraction time by about 3x.
COMPACT_SYSTEM = "Extract the typed knowledge graph from the text."

# System prompt of a schema-conditioned model: the schema is part of the prompt (rendered from the
# Schema), so one model serves any schema instead of having one baked into its weights.
CONDITIONED_SYSTEM = "Extract the typed knowledge graph from the text, using only these types."


def messages(fmt: Format, chunk_text: str, with_example: bool = True,
             compact: bool = False, conditioned: bool = False) -> list[dict[str, str]]:
    """Chat messages for one chunk: system prompt, one worked example, the chunk.
    compact=True: a one-line system prompt and no worked example (fine-tuned students only).
    conditioned=True: a one-line system prompt plus the rendered schema, no example.
    The worked example is a business-schema example, so it is left out under any other schema."""
    if conditioned:
        return [{"role": "system", "content": CONDITIONED_SYSTEM + "\n" + fmt.schema.render()},
                {"role": "user", "content": fmt.render_input(chunk_text)}]
    if compact:
        return [{"role": "system", "content": COMPACT_SYSTEM},
                {"role": "user", "content": fmt.render_input(chunk_text)}]
    msgs = [{"role": "system", "content": fmt.system_prompt()}]
    if with_example and fmt.schema.name == BUSINESS.name:
        example = ExtractionOutput.model_validate(EXAMPLE_OUTPUT)
        msgs.append({"role": "user", "content": fmt.render_input(EXAMPLE_CHUNK)})
        msgs.append({"role": "assistant", "content": fmt.encode(example, EXAMPLE_CHUNK)})
    msgs.append({"role": "user", "content": fmt.render_input(chunk_text)})
    return msgs


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


class SchemaBound:
    """A format works against one Schema (default BUSINESS); with_schema returns a copy bound to another."""
    schema: Schema = BUSINESS

    def with_schema(self, schema: Schema):
        other = copy.copy(self)
        other.schema = schema
        return other

    def _attribute_keys(self) -> tuple[str, ...]:
        return tuple(sorted({a for e in self.schema.edge_names() for a in self.schema.attributes(e)}))


class JsonFormat(SchemaBound):
    name = "json"

    def system_prompt(self) -> str:
        return (PREAMBLE + " Output only JSON that matches the schema; 'mentions' is the list of "
                "surface strings and 'evidence' quotes the supporting sentence verbatim.\n\n"
                + schema_description(self.schema))

    def render_input(self, text: str) -> str:
        return f"Text:\n{text}\n\nJSON:"

    def encode(self, out: ExtractionOutput, text: str) -> str:
        return json.dumps(out.model_dump(mode="json"), ensure_ascii=False)

    def decode(self, raw: str, text: str) -> tuple[dict | None, str | None]:
        body = raw.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.startswith("json"):
                body = body[4:]
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end == -1:
            return None, "no JSON object"
        try:
            data = json.loads(body[start:end + 1])
        except ValueError as e:
            return None, f"json: {e}"
        if not isinstance(data, dict):
            return None, "json: not an object"
        return data, None

    def grammar_key(self, text: str):
        return None  # one grammar for every chunk

    def compile(self, compiler, text: str = ""):
        return compiler.compile_json_schema(json.dumps(json_schema(self.schema)), any_whitespace=False)


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------

ENTITY_RE = re.compile(r"^E(\d+) (\w+): (.*)$")
RELATION_RE = re.compile(r"^R (\w+) E(\d+) E(\d+) (\w+) S(\d+)(?: (.*))?$")


DESCRIPTIVE_STARTS = ("the ", "a ", "an ", "his ", "her ", "its ", "their ", "this ", "that ", "these ", "those ")


def display_name(surfaces: list[str]) -> str:
    """Name for an entity: the longest surface that is a name rather than a description.

    "the company name" or "the service" must not beat "Tetra Pak" or "Spotify". Falls
    back to the longest surface when every surface is descriptive or a pronoun.
    """
    names = [x for x in surfaces if not x.lower().startswith(DESCRIPTIVE_STARTS) and any(ch.isupper() for ch in x)]
    return max(names or surfaces, key=len)


def _alternatives(values) -> str:
    return " | ".join(json.dumps(v) for v in values)


class LineFormat(SchemaBound):
    name = "lines"

    def system_prompt(self) -> str:
        return (
            PREAMBLE + "\n\nThe text is given one sentence per line, numbered S1, S2, ... "
            "Answer with one line per entity, then one line per relation, and nothing else:\n"
            "E<n> <NodeType>: <surface> | <surface> | ...\n"
            "R <EDGE_TYPE> E<source> E<target> <modality> S<sentence> [key=value; key=value]\n"
            "The sentence number points at the sentence that states the relation.\n\n"
            + schema_description(self.schema)
        )

    def sentences(self, text: str) -> list[tuple[int, int]]:
        return sentence_bounds(text)

    def render_input(self, text: str) -> str:
        return "\n".join(f"S{i + 1}: {text[a:b]}" for i, (a, b) in enumerate(self.sentences(text)))

    def _sentence_of(self, text: str, evidence: str) -> int:
        pos = text.find(evidence)
        bounds = self.sentences(text)
        if pos == -1:
            return 1
        for i, (a, b) in enumerate(bounds):
            if a <= pos < b:
                return i + 1
        return len(bounds)

    def encode(self, out: ExtractionOutput, text: str) -> str:
        keys = self._attribute_keys()
        number = {e.id: i + 1 for i, e in enumerate(out.entities)}
        lines = [f"E{number[e.id]} {str(e.type)}: " + " | ".join(e.mentions) for e in out.entities]
        for r in out.relations:
            line = (f"R {r.type} E{number[r.source]} E{number[r.target]} {r.modality.value} "
                    f"S{self._sentence_of(text, r.evidence)}")
            attrs = {k: v for k, v in r.attributes.items() if k in keys and v}
            if attrs:
                line += " " + "; ".join(f"{k}={v}" for k, v in attrs.items())
            lines.append(line)
        return "\n".join(lines)

    def decode(self, raw: str, text: str) -> tuple[dict | None, str | None]:
        bounds = self.sentences(text)
        entities, relations, bad = [], [], 0
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if m := ENTITY_RE.match(line):
                surfaces = list(dict.fromkeys(s.strip() for s in m.group(3).split("|") if s.strip()))
                if not surfaces:
                    bad += 1
                    continue
                entities.append({"id": f"e{m.group(1)}", "type": m.group(2),
                                 "name": display_name(surfaces), "attributes": {},
                                 "mentions": surfaces[:MAX_MENTIONS_PER_ENTITY]})
            elif m := RELATION_RE.match(line):
                idx = int(m.group(5))
                evidence = text[bounds[idx - 1][0]:bounds[idx - 1][1]] if 1 <= idx <= len(bounds) else MISSING_EVIDENCE
                attrs = {}
                for part in (m.group(6) or "").split(";"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if k.strip() and v.strip():
                            attrs[k.strip()] = v.strip()
                relations.append({"type": m.group(1), "source": f"e{m.group(2)}", "target": f"e{m.group(3)}",
                                  "modality": m.group(4), "evidence": evidence, "attributes": attrs})
            else:
                bad += 1
        if not entities and not relations and bad:
            return None, f"lines: no parseable line ({bad} malformed)"
        # entity lines that repeat an id keep the first occurrence
        seen, unique = set(), []
        for e in entities[:MAX_ENTITIES_PER_CHUNK]:
            if e["id"] not in seen:
                seen.add(e["id"])
                unique.append(e)
        return {"entities": unique, "relations": relations[:MAX_RELATIONS_PER_CHUNK]}, None

    def _vocabulary_rules(self) -> list[str]:
        """Grammar rules shared by every line grammar: attributes, surfaces and the schema's names."""
        keys = self._attribute_keys()
        return [
            *(['attrs ::= "" | " " attr ("; " attr){0,3}', 'attr ::= akey "=" [^;\\n<]+'] if keys else
              ['attrs ::= ""']),
            'surface ::= [^|\\n ] [^|\\n]*',
            'num ::= [1-9] [0-9]?',
            f"ntype ::= {_alternatives(self.schema.node_names())}",
            f"etype ::= {_alternatives(self.schema.edge_names())}",
            f"modality ::= {_alternatives(m.value for m in Modality)}",
            *([f"akey ::= {_alternatives(keys)}"] if keys else []),
        ]

    def ebnf(self) -> str:
        """Decoding grammar, derived from the schema and the list caps."""
        return "\n".join([
            f"root ::= entity{{0,{MAX_ENTITIES_PER_CHUNK}}} relation{{0,{MAX_RELATIONS_PER_CHUNK}}}",
            f'entity ::= "E" num " " ntype ": " surface (" | " surface){{0,{MAX_MENTIONS_PER_ENTITY - 1}}} "\\n"',
            'relation ::= "R " etype " E" num " E" num " " modality " S" num attrs "\\n"',
            *self._vocabulary_rules(),
        ])

    def grammar_key(self, text: str):
        return None

    def compile(self, compiler, text: str = ""):
        return compiler.compile_grammar(self.ebnf())


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------

SENTENCE_HEADER_RE = re.compile(r"^S(\d+)$")
SENTENCE_RELATION_RE = re.compile(r"^R (\w+) E(\d+) E(\d+) (\w+)(?: (.*))?$")
MAX_RELATIONS_PER_SENTENCE = 24  # was 6: that cut 7-8% of labels (list sentences) and the model learned to fill blocks to 6


class SentenceFormat(LineFormat):
    name = "sentences"
    header_re = SENTENCE_HEADER_RE
    relation_rule = 'relation ::= "R " etype " E" num " E" num " " modality attrs "\\n"'

    def system_prompt(self) -> str:
        return (
            PREAMBLE + "\n\nThe text is given one sentence per line, numbered S1, S2, ... "
            "Answer with one line per entity, then walk the sentences in order. For each sentence "
            "that states relations, write its number on its own line followed by one line per "
            "relation, and nothing else:\n"
            "E<n> <NodeType>: <surface> | <surface> | ...\n"
            "S<sentence>\n"
            "R <EDGE_TYPE> E<source> E<target> <modality> [key=value; key=value]\n\n"
            + schema_description(self.schema)
        )

    def encode(self, out: ExtractionOutput, text: str) -> str:
        keys = self._attribute_keys()
        number = {e.id: i + 1 for i, e in enumerate(out.entities)}
        lines = [f"E{number[e.id]} {str(e.type)}: " + " | ".join(e.mentions) for e in out.entities]
        by_sentence: dict[int, list[tuple]] = {}
        for r in out.relations:
            line = f"R {r.type} E{number[r.source]} E{number[r.target]} {r.modality.value}"
            attrs = {k: v for k, v in r.attributes.items() if k in keys and v}
            if attrs:
                line += " " + "; ".join(f"{k}={v}" for k, v in attrs.items())
            key = (number[r.source], number[r.target], r.type)
            by_sentence.setdefault(self._sentence_of(text, r.evidence), []).append((key, line))
        # canonical order within a sentence (source, target, type): the labeller's order is
        # arbitrary, and token cross-entropy would otherwise penalise a correct set in another order
        by_sentence = {k: [line for _, line in sorted(v)] for k, v in by_sentence.items()}
        lines += self._blocks(by_sentence, len(self.sentences(text)))
        return "\n".join(lines)

    def _blocks(self, by_sentence: dict[int, list[str]], n_sentences: int) -> list[str]:
        """Answer lines after the entities: a header and the relation lines of each sentence
        that states relations."""
        out = []
        for idx in sorted(by_sentence):
            out.append(f"S{idx}")
            out.extend(by_sentence[idx][:MAX_RELATIONS_PER_SENTENCE])
        return out

    def decode(self, raw: str, text: str) -> tuple[dict | None, str | None]:
        bounds = self.sentences(text)
        entities, relations, bad, current = [], [], 0, None
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if m := ENTITY_RE.match(line):
                surfaces = list(dict.fromkeys(x.strip() for x in m.group(3).split("|") if x.strip()))
                if surfaces:
                    entities.append({"id": f"e{m.group(1)}", "type": m.group(2), "name": display_name(surfaces),
                                     "attributes": {}, "mentions": surfaces[:MAX_MENTIONS_PER_ENTITY]})
                else:
                    bad += 1
            elif m := self.header_re.match(line):
                current = int(m.group(1))
            elif m := SENTENCE_RELATION_RE.match(line):
                ok = current is not None and 1 <= current <= len(bounds)
                evidence = text[bounds[current - 1][0]:bounds[current - 1][1]] if ok else MISSING_EVIDENCE
                attrs = {}
                for part in (m.group(5) or "").split(";"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if k.strip() and v.strip():
                            attrs[k.strip()] = v.strip()
                relations.append({"type": m.group(1), "source": f"e{m.group(2)}", "target": f"e{m.group(3)}",
                                  "modality": m.group(4), "evidence": evidence, "attributes": attrs})
            else:
                bad += 1
        if not entities and not relations and bad:
            return None, f"sentences: no parseable line ({bad} malformed)"
        seen, unique = set(), []
        for e in entities[:MAX_ENTITIES_PER_CHUNK]:
            if e["id"] not in seen:
                seen.add(e["id"])
                unique.append(e)
        return {"entities": unique, "relations": relations[:MAX_RELATIONS_PER_CHUNK]}, None

    def ebnf_for(self, n_sentences: int, entity_types: dict[int, str] | None = None,
                 grounded: dict[int, set[int]] | None = None) -> str:
        """Grammar for a chunk with n sentences: optional sentence blocks in strict order.

        With entity_types (entity number -> node type, known once the entity lines are written),
        relation lines may only use edge types whose endpoint types fit, and no self-loops.
        With grounded (sentence number -> entity numbers mentioned in that sentence), a relation in
        block S<k> may only connect entities mentioned in sentence k; blocks without such a pair vanish."""
        def alternatives(allowed=None):
            alts = []
            for name in self.schema.edge_names():
                srcs = [n for n, t in entity_types.items() if t in self.schema.sources(name) and (allowed is None or n in allowed)]
                tgts = [n for n, t in entity_types.items() if t in self.schema.targets(name) and (allowed is None or n in allowed)]
                for src in srcs:
                    if others := [t for t in tgts if t != src]:
                        alts.append(f'"{name} E{src} E" ({_alternatives(str(t) for t in others)})')
            return alts

        if grounded is not None and entity_types is not None:
            rules, blocks = [], []
            for k in range(1, n_sentences + 1):
                if alts := alternatives(grounded.get(k, set())):
                    blocks.append(k)
                    rules += [f'block{k} ::= "S{k}\\n" relation{k}{{1,{MAX_RELATIONS_PER_SENTENCE}}}',
                              f'relation{k} ::= "R " ({" | ".join(alts)}) " " modality attrs "\\n"']
            root = f"root ::= entity{{0,{MAX_ENTITIES_PER_CHUNK}}} " + " ".join(f"block{k}?" for k in blocks)
            rules = [root.rstrip()] + rules
            relation = None
        else:
            relation = self.relation_rule
            alts = None
            if entity_types is not None:
                alts = alternatives()
                relation = f'relation ::= "R " ({" | ".join(alts) or chr(34) * 2}) " " modality attrs "\\n"'
            rules = self._root_rules(n_sentences, has_relations=entity_types is None or bool(alts))
        rules += [
            f'entity ::= "E" num " " ntype ": " surface (" | " surface){{0,{MAX_MENTIONS_PER_ENTITY - 1}}} "\\n"',
            *([relation] if relation else []),
            *self._vocabulary_rules(),
        ]
        return "\n".join(rules)

    def _root_rules(self, n_sentences: int, has_relations: bool) -> list[str]:
        """Root and block rules: optional sentence blocks in strict order."""
        n = n_sentences if has_relations else 0  # no legal relation between these entities: entity lines only
        blocks = " ".join(f"block{k}?" for k in range(1, n + 1))
        rules = [f"root ::= entity{{0,{MAX_ENTITIES_PER_CHUNK}}} {blocks}".rstrip()]
        return rules + [f'block{k} ::= "S{k}\\n" relation{{1,{MAX_RELATIONS_PER_SENTENCE}}}' for k in range(1, n + 1)]

    def grammar_key(self, text: str):
        return len(self.sentences(text))

    def compile(self, compiler, text: str = ""):
        return compiler.compile_grammar(self.ebnf_for(self.grammar_key(text)))


# Formats by name. Other formats can be added to this dict (register_format) and are then found by name
# everywhere a format name is accepted (finish, load_results, chat_messages).
FORMATS: dict[str, Format] = {"json": JsonFormat(), "lines": LineFormat(), "sentences": SentenceFormat()}


def register_format(fmt: Format) -> Format:
    if FORMATS.get(fmt.name, fmt) is not fmt and type(FORMATS[fmt.name]) is not type(fmt):
        raise ValueError(f"a different format is already registered as {fmt.name}")
    FORMATS[fmt.name] = fmt
    return fmt
STUDENT_FORMAT = "sentences"

# Business attribute keys, for importers that predate schemas; a schema's own keys come from
# fmt.schema (SchemaBound._attribute_keys).
ATTRIBUTE_KEYS: tuple[str, ...] = FORMATS["sentences"]._attribute_keys()


# --------------------------------------------------------------------------
# Prompts, parsing and span resolution of model answers
# --------------------------------------------------------------------------


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """A full conversation as training text; the one place the chat template is applied."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)


def generation_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    """The prompt for generating the next assistant turn, cut from render_chat so it ends exactly
    where a training answer starts. Templates differ in what they put before the answer (Unsloth's
    Qwen3-4B-Instruct template adds an empty <think></think> block in training text but not in its
    generation prompt), so deriving the prompt from the training rendering keeps the two aligned."""
    marker = "\x00ANSWER\x00"
    full = render_chat(tokenizer, [*messages, {"role": "assistant", "content": marker}])
    return full[:full.index(marker)]


def chat_messages(chunk_text: str, fmt: str, with_example: bool = True, compact: bool = False,
                  schema: Schema | None = None, conditioned: bool = False) -> list[dict[str, str]]:
    """Prompt for one chunk in the given output format; compact=True for compact-prompt students.
    schema: the schema the prompt and grammar describe (default BUSINESS); conditioned=True puts the
    rendered schema in the system prompt."""
    f = FORMATS[fmt] if schema is None else FORMATS[fmt].with_schema(schema)
    return messages(f, chunk_text, with_example=with_example, compact=compact, conditioned=conditioned)


@dataclass
class ExtractionResult:
    chunk_id: str
    raw: str
    output: ExtractionOutput | None
    error: str | None = None
    seconds: float = 0.0
    unresolved_mentions: int = 0
    unresolved_evidence: int = 0
    dropped_relations: int = 0
    illegal_relations: int = 0
    dropped_entities: int = 0  # entities whose type is not in the schema
    repaired_relations: int = 0
    resolved_spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    fmt: str = "json"
    schema: str | None = None  # name of the schema the answer was parsed under (None: business)


def parse_output(raw: str, repair: bool = False, fmt: str = "json",
                 chunk_text: str = "", schema: Schema | None = None,
                 stats: dict | None = None) -> tuple[ExtractionOutput | None, str | None, int, int]:
    """Parse model text in the given format. Returns (output, error, illegal relations dropped,
    relations repaired). The line format needs the chunk text to resolve sentence references."""
    data, err = FORMATS[fmt].decode(raw, chunk_text)
    if data is None:
        return None, err, 0, 0
    try:
        out, dropped, repaired = parse_lenient(data, repair=repair, schema=schema, stats=stats)
    except (ValidationError, ValueError) as e:
        msg = e.errors()[0]["msg"] if isinstance(e, ValidationError) else str(e)
        return None, f"schema: {msg}", 0, 0
    return out, None, dropped, repaired


def find_all(text: str, needle: str) -> list[tuple[int, int]]:
    """All non-overlapping whole-word occurrences of needle in text.

    An occurrence counts only if it is not glued to letters or digits on either side,
    so "he" does not match inside "the" and "her" not inside "father". Edges of the
    needle that are themselves punctuation ("H&M", "(publ)") need no boundary.
    """
    out = []
    if not needle:
        return out
    pos = text.find(needle)
    while pos != -1:
        end = pos + len(needle)
        left_ok = not needle[0].isalnum() or pos == 0 or not text[pos - 1].isalnum()
        right_ok = not needle[-1].isalnum() or end == len(text) or not text[end].isalnum()
        if left_ok and right_ok:
            out.append((pos, end))
            pos = text.find(needle, end)
        else:
            pos = text.find(needle, pos + 1)
    return out


def resolve_spans(chunk_text: str, output: ExtractionOutput) -> tuple[dict[str, list[tuple[int, int]]], int]:
    """Expand each entity's surface strings to character spans.

    Longer surfaces are claimed first and a span already claimed (by any entity)
    is skipped, so 'Granit' inside 'Granit Bygg Group AB' is not double counted.
    Returns spans per entity id and the number of surfaces that never matched.
    """
    claims: list[tuple[int, str, str]] = []  # (-len, entity id, surface)
    for ent in output.entities:
        for surface in dict.fromkeys(ent.mentions):
            claims.append((-len(surface), ent.id, surface))
    claims.sort()
    taken: list[tuple[int, int]] = []
    spans: dict[str, list[tuple[int, int]]] = {e.id: [] for e in output.entities}
    unmatched = 0
    for _, eid, surface in claims:
        hits = [(a, b) for a, b in find_all(chunk_text, surface) if not any(a < tb and b > ta for ta, tb in taken)]
        if not hits and not find_all(chunk_text, surface):
            unmatched += 1
        spans[eid].extend(hits)
        taken.extend(hits)
    for eid in spans:
        spans[eid].sort()
    return spans, unmatched


def resolve(chunk_text: str, output: ExtractionOutput) -> tuple[ExtractionOutput, dict]:
    """Verify every mention surface and evidence by exact string match.

    Returns a cleaned output (surfaces absent from the text removed, entities left
    with no span removed together with their relations, relations with
    unverifiable evidence removed) and counters.
    """
    stats = {"unresolved_mentions": 0, "unresolved_evidence": 0, "dropped_relations": 0, "spans": {}}
    kept_entities = []
    for ent in output.entities:
        good = [m for m in dict.fromkeys(ent.mentions) if m and find_all(chunk_text, m)]
        stats["unresolved_mentions"] += len(set(ent.mentions)) - len(good)
        if good:
            kept_entities.append(ent.model_copy(update={"mentions": good}))
    # model_copy, not a constructor: a constructor validates under BUSINESS, whatever schema produced output
    cleaned = output.model_copy(update={"entities": kept_entities, "relations": []})
    spans, _ = resolve_spans(chunk_text, cleaned)
    kept_entities = [e for e in kept_entities if spans[e.id]]
    kept_ids = {e.id for e in kept_entities}
    stats["spans"] = {eid: spans[eid] for eid in kept_ids}
    kept_relations = []
    for r in output.relations:
        if r.source not in kept_ids or r.target not in kept_ids:
            stats["dropped_relations"] += 1
            continue
        if r.evidence not in chunk_text:
            stats["unresolved_evidence"] += 1
            continue
        kept_relations.append(r)
    return output.model_copy(update={"entities": kept_entities, "relations": kept_relations}), stats


class Extractor(Protocol):
    name: str

    def extract(self, chunk_id: str, chunk_text: str) -> ExtractionResult: ...


def finish(chunk_id: str, chunk_text: str, raw: str, seconds: float, repair: bool = False,
           fmt: str = "json", schema: Schema | None = None) -> ExtractionResult:
    parse_stats: dict = {}
    output, err, illegal, repaired = parse_output(raw, repair=repair, fmt=fmt, chunk_text=chunk_text,
                                                  schema=schema, stats=parse_stats)
    name = (schema or BUSINESS).name
    if output is None:
        return ExtractionResult(chunk_id=chunk_id, raw=raw, output=None, error=err, seconds=seconds, fmt=fmt,
                                schema=name)
    cleaned, stats = resolve(chunk_text, output)
    return ExtractionResult(chunk_id=chunk_id, raw=raw, output=cleaned, seconds=seconds,
                            unresolved_mentions=stats["unresolved_mentions"],
                            unresolved_evidence=stats["unresolved_evidence"],
                            dropped_relations=stats["dropped_relations"],
                            illegal_relations=illegal,
                            dropped_entities=parse_stats.get("dropped_entities", 0),
                            repaired_relations=repaired,
                            resolved_spans=stats["spans"], fmt=fmt, schema=name)


def save_results(results: list[ExtractionResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "chunk_id": r.chunk_id, "format": r.fmt, "raw": r.raw, "error": r.error, "seconds": r.seconds,
                "output": r.output.model_dump(mode="json") if r.output else None,
                "unresolved_mentions": r.unresolved_mentions,
                "unresolved_evidence": r.unresolved_evidence,
                "dropped_relations": r.dropped_relations,
                "illegal_relations": r.illegal_relations,
                "resolved_spans": r.resolved_spans,
                **({"schema": r.schema} if r.schema else {}),
            }, ensure_ascii=False) + "\n")


def _schema_for(record: dict, schema: Schema | dict | None) -> Schema:
    """The schema a stored result is read under: the argument (one Schema, or chunk id -> Schema), else the
    record's own "schema" name (it must be registered: relweave.schema.load_schema registers a user schema),
    else business."""
    if isinstance(schema, dict):
        schema = schema.get(record["chunk_id"])
    if schema is not None:
        return schema
    name = record.get("schema")
    if name is None:
        return BUSINESS
    if name not in SCHEMAS:  # a name from a file resolves only against registered schemas, never as code
        raise ValueError(f"result {record['chunk_id']} was written under schema {name!r}, which is not registered; "
                         "pass schema= or load it first (relweave.schema.load_schema)")
    return SCHEMAS[name]


def load_results(path: Path, chunk_texts: dict[str, str] | None = None, repair: bool = False,
                 schema: Schema | dict | None = None) -> list[ExtractionResult]:
    """Load results. With chunk_texts, re-parse and re-resolve from the stored raw text,
    so parser and resolver changes apply without rerunning the model. Every record is read under its schema
    (see _schema_for): a relation type the business schema lacks is not dropped."""
    out = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        sch = _schema_for(d, schema)
        if chunk_texts is not None and d["chunk_id"] in chunk_texts:
            out.append(finish(d["chunk_id"], chunk_texts[d["chunk_id"]], d["raw"], d["seconds"], repair=repair,
                              fmt=d.get("format", "json"), schema=sch))
            continue
        out.append(ExtractionResult(
            chunk_id=d["chunk_id"], raw=d["raw"], error=d["error"], seconds=d["seconds"],
            output=ExtractionOutput.model_validate(d["output"], context={"schema": sch}) if d["output"] else None,
            unresolved_mentions=d["unresolved_mentions"], unresolved_evidence=d["unresolved_evidence"],
            dropped_relations=d["dropped_relations"], illegal_relations=d.get("illegal_relations", 0),
            resolved_spans={k: [tuple(x) for x in v] for k, v in d["resolved_spans"].items()},
            fmt=d.get("format", "json"), schema=sch.name,
        ))
    return out
