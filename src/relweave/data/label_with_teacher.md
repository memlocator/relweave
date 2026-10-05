# Labelling chunks with a frontier model

`relweave label prompts` writes one JSON file per chunk with `chunk_id` and `messages`
(system prompt with the schema, one worked example, the chunk). Whoever answers them, an
agent or an API script, writes ONLY the JSON answer to `<raw dir>/<chunk file stem>.txt`.

Prompt for an agent:

---
There are N prompt files in <prompt dir>/*.json. Each has "chunk_id" and "messages". For each
file, follow the system prompt and the worked example exactly and produce the JSON extraction
for the text in the final user message. Write ONLY that JSON, no prose or code fences, to
<raw dir>/<name>.txt where <name> is the prompt file's basename without .json.

Each entity's "mentions" is a list of DISTINCT surface strings copied verbatim from the chunk;
every occurrence of a listed string counts as a mention, so list a pronoun under an entity only
if every occurrence of it in the chunk refers to that entity. Longer surfaces are matched
first, so a bare surname is safe to list even when it also appears inside someone else's
full name. Evidence is copied verbatim. Only
edge types and endpoint types the schema allows; when several fit, use the most specific
definition (a birthplace is BORN_IN, a market is OPERATES_IN, not LOCATED_IN). Modality as the
text expresses it. Extract what the text states even if it seems implausible.

Conventions from the first labelling round: a family acting as owner or founder ("the
Wallenberg family") is an Org. If a string refers to different entities in different places
in the chunk (a bare surname shared by father and son, "it" for two companies), leave it
out of every entity. Copy evidence character for character, including unusual spaces such
as the thin space after "lit.". Do not infer relations from names alone or from names merely appearing next to each other.
A list does state relations when a lead sentence or heading states the relation for every item ("Investor
AB owns significant holdings of the following companies:" then the companies; "Member of" lines under a
person's honours): each item gets the relation. A Person is LOCATED_IN a Place only where they live or are
based; where they studied, died, visited or where an event happened gives no relation.
Do not list a mention that contains another entity's name ("the trial of Lundin and
Schneiter"): longer surfaces are matched first and would swallow that name. Do not read any
existing labels; label independently. A place where an Org was founded is not its
headquarters: "founded in Basel" alone gives no HEADQUARTERED_IN.
---

Answers are parsed and span-verified like any extractor output. Unverifiable mentions,
evidence and illegal relations are dropped on import and counted in the command output.
