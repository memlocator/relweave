"""Rule-based relations the generator tends to miss, added after the merge from the entities and mention positions
it found: roles carried by titles next to a name ("Trial Attorney Asha Natarajan of the Fraud Section", "Fraud
Section Trial Attorney Asha Natarajan", "Kappahl's sustainability director, Sandra Roos") and party membership carried
by an adjective in a description ("Alan Lowenthal, a Democratic state senator"). A rule adds a relation only between
a Person and an Org the graph already has, only when the words between them are a title (or nothing), and only when
the graph has no relation between that pair yet, so it fills gaps and never overrides the model.

The title lexicon sets the type: executive titles EXECUTIVE_OF, board seats BOARD_MEMBER_OF, every other job title
EMPLOYED_BY (the schema's role rules). RULES switches the step off for evaluation.
"""
from __future__ import annotations

import re

from relweave.graph import Entity, Evidence, Graph, Occurrence, Relation

RULES = True
EXECUTIVE = ("chief executive officer", "chief executive", "chief financial officer", "chief operating officer",
             "chief technology officer", "ceo", "cfo", "coo", "cto", "president", "chairman", "chairwoman",
             "chairperson", "chair", "managing director", "executive chairman", "founder and chief executive")
BOARD = ("board member", "non-executive director", "independent director", "director on the board")
JOBS = ("trial attorney", "assistant united states attorney", "assistant u.s. attorney", "attorney", "prosecutor",
        "special agent in charge", "special agent", "agent", "inspector", "investigator", "officer", "spokesperson",
        "spokesman", "spokeswoman", "press officer", "director", "manager", "head of", "engineer", "analyst",
        "economist", "scientist", "researcher", "professor", "lecturer", "coach", "manager", "striker", "midfielder",
        "defender", "goalkeeper", "player", "captain", "editor", "reporter", "correspondent", "judge", "lawyer",
        "counsel", "partner", "associate", "employee", "worker", "consultant", "adviser", "advisor", "secretary",
        "minister", "ambassador", "senator", "representative", "member")
PARTIES = {"democratic": "Democratic Party", "republican": "Republican Party", "labour": "Labour Party",
           "conservative": "Conservative Party", "liberal democrat": "Liberal Democrats", "green": "Green Party"}
TITLE = "|".join(re.escape(t) for t in sorted(EXECUTIVE + BOARD + JOBS, key=len, reverse=True))
TITLE_RE = re.compile(rf"^(?:(?:the|a|an|its|their|senior|deputy|assistant|acting|former|interim|lead|chief|head)\s+)*"
                      rf"(?:[a-z][\w-]*\s+){{0,2}}(?:{TITLE})s?\b", re.I)
PARTY_RE = re.compile(r",?\s*(?:a|an|the)\s+(" + "|".join(PARTIES) + r")\s+(?:[a-z]+\s+){0,2}"
                      r"(?:senator|representative|congressman|congresswoman|mp|candidate|member|politician|governor|"
                      r"lawmaker|legislator|councillor|councilor|mayor)\b", re.I)

PARTY_BEFORE_RE = re.compile(r"\b(" + "|".join(PARTIES) + r")\s+(?:[a-z]+\s+){0,2}(?:senator|representative|congressman|"
                             r"congresswoman|mp|candidate|member|politician|governor|lawmaker|legislator|councillor|"
                             r"councilor|mayor)\s+$", re.I)


def _type_for(title: str) -> str:
    t = title.lower()
    if any(re.search(rf"\b{re.escape(x)}\b", t) for x in BOARD):
        return "BOARD_MEMBER_OF"
    if any(re.search(rf"\b{re.escape(x)}\b", t) for x in EXECUTIVE) and "assistant" not in t and "vice" not in t:
        return "EXECUTIVE_OF"
    return "EMPLOYED_BY"


def _sentence_ok(text: str, a: int, b: int) -> bool:
    """No sentence end between two offsets."""
    return not re.search(r"[.!?]\s", text[min(a, b):max(a, b)])


def apply_rules(g: Graph) -> Graph:
    """The graph with rule relations added (see the module docstring)."""
    if not RULES:
        return g
    text = g.text
    linked = {frozenset((r.source, r.target)) for r in g.relations}
    persons = [(e, m) for e in g.entities if e.type == "Person" for m in e.mentions if m.text[:1].isupper()]
    orgs = [(e, m) for e in g.entities if e.type == "Org" for m in e.mentions if m.text[:1].isupper()]
    added: list[Relation] = []

    def add(typ, p, o, title, start, end):
        key = frozenset((p.id, o.id))
        if key in linked:
            return
        linked.add(key)
        added.append(Relation(typ, p.id, o.id, attributes={"role": title.strip()}, score=None, origin="rule",
                              evidence=[Evidence(-1, (start, end))]))

    for p, pm in persons:
        for o, om in orgs:
            if not _sentence_ok(text, pm.start, om.start):
                continue
            # ORG ['s] TITLE PERSON  ("Fraud Section Trial Attorney Asha Natarajan", "Kappahl's director, Sandra Roos")
            if om.end <= pm.start and pm.start - om.end <= 60:
                between = text[om.end:pm.start]
                m = re.fullmatch(r"(?:'s|’s)?\s*(.*?)[,\s]*", between, re.S)
                if m and (m.group(1) == "" and between.strip() in ("'s", "’s") or TITLE_RE.fullmatch(m.group(1))):
                    add(_type_for(m.group(1)) if m.group(1) else "EMPLOYED_BY", p, o, m.group(1), om.start, pm.end)
                    continue
            # TITLE PERSON of/at/for ORG  ("Trial Attorney Asha Natarajan of the Fraud Section")
            if pm.end <= om.start and om.start - pm.end <= 25:
                after = text[pm.end:om.start]
                before = text[max(0, pm.start - 60):pm.start]
                tm = re.search(rf"(?:{TITLE})s?\s*$", before, re.I)
                if tm and re.fullmatch(r",?\s+(?:of|at|for|with|from)\s+(?:the\s+)?", after, re.I):
                    title = before[tm.start():].strip()
                    add(_type_for(title), p, o, title, pm.start - len(before) + tm.start(), om.end)
            # PERSON from/of/at a short description ORG  ("Richard Lloyd from product sampling charity Which?")
            if pm.end <= om.start and om.start - pm.end <= 60:
                d = re.fullmatch(r",?\s+(?:of|at|for|with|from)\s+(?:the\s+)?((?:[a-z][\w-]*\s+){1,4})", text[pm.end:om.start])
                if d and not re.search(r"\b(?:and|or|by|to|in|on|who|which|that)\b", d.group(1)):
                    add("EMPLOYED_BY", p, o, "", pm.start, om.end)
    # PERSON, a Democratic state senator / Republican state senator PERSON -> MEMBER_OF the party
    entities = list(g.entities)
    for p, pm in persons:
        m = PARTY_RE.match(text, pm.end) or PARTY_BEFORE_RE.search(text, max(0, pm.start - 60), pm.start)
        if not m:
            continue
        party = PARTIES[m.group(1).lower()]
        org = next((e for e in entities if e.type == "Org" and e.name.lower() == party.lower()), None)
        if org is None:
            org = Entity(f"r{len(entities) + 1}", "Org", party, [], aliases=[m.group(1)])
            entities.append(org)
        if frozenset((p.id, org.id)) not in linked:
            linked.add(frozenset((p.id, org.id)))
            added.append(Relation("MEMBER_OF", p.id, org.id, score=None, origin="rule",
                                  evidence=[Evidence(-1, (pm.start, m.end()))]))
    if not added:
        return g
    return Graph(entities=entities, relations=list(g.relations) + added, text=g.text, document=g.document,
                 schema=g.schema, model=g.model, version=g.version, warnings=g.warnings)
