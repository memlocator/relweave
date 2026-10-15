There are N prompt files in <prompt dir>/*.json. Each has "chunk_id" and "messages". For each
file, follow the system prompt and the worked example exactly and produce the JSON extraction
for the text in the final user message. Write ONLY that JSON, no prose or code fences, to
<raw dir>/<name>.txt where <name> is the prompt file's basename without .json.

Each entity's "mentions" is a list of DISTINCT surface strings copied verbatim from the chunk;
every occurrence of a listed string counts as a mention, so list a pronoun under an entity only
if every occurrence of it in the chunk refers to that entity. Longer surfaces are matched
first, so a bare surname is safe to list even when it also appears inside someone else's
full name. Evidence is copied verbatim. Only edge types and endpoint types the schema allows; when several fit, use
the most specific definition. Modality as the text expresses it. Extract what the text states even if it seems
implausible.

If a string refers to different entities in different places in the chunk (a bare surname shared by father and son,
"it" for two companies), leave it out of every entity. Copy evidence character for character, including unusual
spaces. Do not infer relations from names alone or from names merely appearing next to each other. A list does state
relations when a lead sentence or heading states the relation for every item: each item gets the relation. Do not list
a mention that contains another entity's name ("the trial of Lundin and Schneiter"): longer surfaces are matched first
and would swallow that name. Do not read any existing labels; label independently.
