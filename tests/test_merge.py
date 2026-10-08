from relweave.merge import merge, normalise, is_generic, is_key
from relweave.records import ChunkEntity, ChunkGraph, ChunkRelation, Mention


def ent(i, typ, name, *mentions):
    ms = [Mention(*m) if isinstance(m, tuple) else Mention(m) for m in (mentions or (name,))]
    return ChunkEntity(i, typ, name, ms)


def cg(k, entities, relations=()):
    return ChunkGraph(k, list(entities), list(relations))


def names(g):
    return sorted(e.name for e in g.entities)


def test_normalise_and_generic():
    assert normalise("The Nordic Bank ASA") == "nordic bank"
    assert normalise("Acme, Inc.") == "acme"
    assert normalise("Volvo AB") == "volvo"
    assert is_generic("it") and is_generic("He") and is_generic("The company")
    assert is_generic("the parent company") and is_generic("its subsidiary")
    assert not is_generic("The Hague") and not is_generic("Adidas")


def test_anchor_merges_same_offset_even_with_different_names():
    a = cg(0, [ent("e1", "Org", "Acme", ("Acme", 10), ("the firm", 40))])
    b = cg(1, [ent("e1", "Org", "Acme Holdings", ("the firm", 40))])
    g = merge([a, b], "x" * 100)
    assert len(g.entities) == 1


def test_anchor_needs_the_same_string_at_the_offset():
    a = cg(0, [ent("e1", "Org", "Acme", ("Acme", 5))])
    b = cg(1, [ent("e1", "Org", "Bolt", ("Bolt", 5))])
    assert len(merge([a, b], "x" * 20).entities) == 2


def test_name_merge_with_legal_suffix_and_the_prefix():
    a = cg(0, [ent("e1", "Org", "Nordic Bank ASA")])
    b = cg(1, [ent("e1", "Org", "The Nordic Bank")])
    g = merge([a, b], "")
    assert len(g.entities) == 1
    assert g.entities[0].name == "Nordic Bank ASA"
    assert g.entities[0].chunks == [0, 1]


def test_different_type_does_not_merge():
    a = cg(0, [ent("e1", "Org", "Paris")])
    b = cg(1, [ent("e1", "Place", "Paris")])
    assert len(merge([a, b], "").entities) == 2


def test_shared_proper_mention_merges_but_pronoun_does_not():
    a = cg(0, [ent("e1", "Org", "Acme", "Acme", "it")])
    b = cg(1, [ent("e1", "Org", "Bolt", "Bolt", "it")])
    assert len(merge([a, b], "").entities) == 2
    c = cg(1, [ent("e1", "Org", "Acme Corp Group", "Acme Corp Group", "Acme")])
    assert len(merge([a, c], "").entities) == 1


def test_same_chunk_entities_are_never_merged_by_name():
    a = cg(0, [ent("e1", "Org", "Acme"), ent("e2", "Org", "Acme Inc")])
    assert len(merge([a], "").entities) == 2
    b = cg(1, [ent("e1", "Org", "Acme")])
    assert len(merge([a, b], "").entities) == 2  # joining both would fuse two entities of chunk 0


def test_generic_description_joins_the_single_candidate():
    a = cg(0, [ent("e1", "Org", "Acme", "Acme", "The company")])
    b = cg(1, [ent("e1", "Org", "The company", "The company")])
    g = merge([a, b], "")
    assert len(g.entities) == 1 and g.entities[0].name == "Acme"


def test_generic_description_with_two_candidates_stays_separate():
    a = cg(0, [ent("e1", "Org", "Acme", "Acme", "the company"), ent("e2", "Org", "Bolt", "Bolt", "the company")])
    b = cg(1, [ent("e1", "Org", "The company", "the company")])
    g = merge([a, b], "")
    assert len(g.entities) == 3


def test_generic_description_needs_same_type():
    a = cg(0, [ent("e1", "Org", "Acme", "Acme", "the group")])
    b = cg(1, [ent("e1", "Person", "The group", "the group")])
    assert len(merge([a, b], "").entities) == 2


def test_relations_rewritten_deduped_with_max_score_and_evidence():
    a = cg(0, [ent("e1", "Org", "Acme"), ent("e2", "Org", "Bolt")],
           [ChunkRelation("OWNS_STAKE_IN", "e1", "e2", score=0.7, origin="generator")])
    b = cg(1, [ent("e1", "Org", "Bolt"), ent("e2", "Org", "Acme")],
           [ChunkRelation("OWNS_STAKE_IN", "e2", "e1", score=2.1, origin="head"),
            ChunkRelation("OWNS_STAKE_IN", "e1", "e1", score=9.0)])  # Bolt -> Bolt: a self-loop, dropped
    g = merge([a, b], "")
    assert len(g.entities) == 2
    assert len(g.relations) == 1
    r = g.relations[0]
    assert r.score == 2.1 and [v.chunk for v in r.evidence] == [0, 1]
    assert r.origin == "generator"
    assert (r.source, r.target) == ("e1", "e2")


def test_symmetric_relations_dedupe_order_insensitively():
    from relweave.schema import BUSINESS
    sym = next(n for n in BUSINESS.edge_names() if BUSINESS.symmetric(n))
    s, t = sorted(BUSINESS.sources(sym) & BUSINESS.targets(sym))[0], None
    a = cg(0, [ent("e1", s, "Ann"), ent("e2", s, "Bob")], [ChunkRelation(sym, "e1", "e2", score=1.0)])
    b = cg(1, [ent("e1", s, "Bob"), ent("e2", s, "Ann")], [ChunkRelation(sym, "e1", "e2", score=3.0)])
    g = merge([a, b], "")
    assert len(g.relations) == 1 and g.relations[0].score == 3.0


def test_mentioned_in_is_derived_from_offsets_and_deduped():
    text = "Acme owns Bolt. Acme is big."
    a = cg(0, [ent("e1", "Org", "Acme", ("Acme", 0), ("Acme", 16))])
    b = cg(1, [ent("e1", "Org", "Acme", ("Acme", 16), ("Acme", 99))])  # 99: not in this text, unknown offset ignored
    g = merge([a, b], text)
    recs = [(m["entity"], m["mention"], m["start"], m["end"]) for m in g.mentioned_in]
    assert recs == [("e1", "Acme", 0, 4), ("e1", "Acme", 16, 20)]
    assert len(g.text) == len(text) and len(g.document["chunks"]) == 2


def test_chunk_warnings_flow_into_the_graph():
    a = cg(0, [])
    a.warnings.append("chunk 0: broke")
    assert merge([a], "").warnings == ["chunk 0: broke"]


def test_entity_groups_agree_with_merge_ids():
    from relweave.merge import entity_groups
    a = cg(0, [ent("e1", "Org", "Acme"), ent("e2", "Org", "Bolt")])
    b = cg(1, [ent("e1", "Org", "Bolt"), ent("e2", "Org", "Cargo")])
    groups = entity_groups([a, b])
    assert groups[(0, "e2")] == groups[(1, "e1")]
    assert len(set(groups.values())) == 3
    assert {e.id for e in merge([a, b], "").entities} == set(groups.values())


# ---- review fixes: false merges ------------------------------------------------------------------------------

def test_pronouns_never_anchor_entities():
    a = cg(0, [ent("e1", "Org", "Acme Corp", ("Acme Corp", 0), ("It", 30))])
    b = cg(1, [ent("e1", "Org", "Beta Ltd", ("Beta Ltd", 21), ("It", 30))])
    assert len(merge([a, b], "x" * 60).entities) == 2


def test_anchor_requires_the_same_type():
    a = cg(0, [ent("e1", "Person", "Paris Hilton", ("Paris", 5))])
    b = cg(1, [ent("e1", "Place", "Elsewhere", ("Paris", 5))])
    assert len(merge([a, b], "x" * 20).entities) == 2


def test_is_generic_all_caps_and_one_are_not_pronouns():
    for s in ("US", "WHO", "IT", "One"):
        assert not is_generic(s), s
    assert is_generic("it") and is_generic("It") and is_generic("They")


def test_shared_surname_of_two_different_people_does_not_merge():
    a = cg(0, [ent("e1", "Person", "Petter Lunde", "Petter Lunde", "Lunde"), ent("e2", "Person", "Hilde Lunde", "Hilde Lunde")])
    b = cg(1, [ent("e1", "Person", "Lunde", "Lunde")])
    c = cg(2, [ent("e1", "Person", "Lunde", "Lunde")])
    assert len(merge([a, b, c], "").entities) == 4


def test_surname_merges_into_the_unique_full_name_even_in_one_chunk():
    a = cg(0, [ent("e1", "Person", "Tor Amundsen", "Tor Amundsen"), ent("e2", "Person", "Amundsen", "Amundsen")])
    g = merge([a, cg(1, [ent("e1", "Person", "Tor Amundsen")])], "")
    assert len(g.entities) == 1 and g.entities[0].name == "Tor Amundsen"


def test_role_and_description_mentions_do_not_merge_people_or_things():
    for typ, m in (("Person", "the CEO"), ("Person", "the chairman"), ("Object", "the plant"),
                   ("Org", "the Norwegian company")):
        a = cg(0, [ent("e1", typ, "Alpha One", "Alpha One", m)])
        b = cg(1, [ent("e1", typ, "Beta Two", "Beta Two", m)])
        assert len(merge([a, b], "").entities) == 2, m


def test_pure_pronoun_entities_are_not_merged_by_rule_c():
    a = cg(0, [ent("e1", "Org", "Acme", "Acme", "it")])
    b = cg(1, [ent("e1", "Org", "It", "it")])
    assert len(merge([a, b], "").entities) == 2


def test_description_joins_only_a_nearby_unique_candidate():
    grid = cg(0, [ent("e1", "Org", "Grid Solutions", "Grid Solutions", "the area")])
    water = cg(1, [ent("e1", "Org", "Water Technology", "Water Technology", "the area")])
    area = cg(3, [ent("e1", "Org", "The area", "The area")])
    g = merge([grid, water, cg(2, []), area], "")
    assert len(g.entities) == 3  # chunk 3 is next to chunk 2 only; neither candidate is adjacent
    near = cg(2, [ent("e1", "Org", "The area", "The area")])
    g = merge([grid, water, near], "")
    assert sorted(e.name for e in g.entities) == ["Grid Solutions", "Water Technology"]
    assert next(e for e in g.entities if e.name == "Water Technology").chunks == [1, 2]


def test_rule_c_does_not_depend_on_entity_order():
    mk = lambda order: [cg(0, [ent("e1", "Org", "Acme", "Acme", "the company")]),
                        cg(1, [ent(i, "Org", "The company", "the company") for i in order])]
    # two all-generic entities in one chunk, one candidate: both are offered the same snapshot
    g1, g2 = merge(mk(["e1", "e2"]), ""), merge(mk(["e2", "e1"]), "")
    assert len(g1.entities) == len(g2.entities)


def test_union_log_names_the_rule_of_each_join():
    from relweave.merge import union_log
    a = cg(0, [ent("e1", "Org", "Acme", ("Acme", 5), "the company")])
    b = cg(1, [ent("e1", "Org", "Acme", ("Acme", 5))])
    c = cg(2, [ent("e1", "Org", "Acme Inc")])
    d = cg(3, [ent("e2", "Org", "The company", "The company")])
    rules = [r for r, _, _ in union_log([a, b, c, d])]
    assert rules[0] == "a" and "b" in rules and "c" in rules


def test_related_organisations_in_one_chunk_stay_separate_with_their_relation():
    a = cg(0, [ent("e1", "Org", "Telenor", "Telenor"), ent("e2", "Org", "Telenor Norge", "Telenor Norge")],
           [ChunkRelation("SUBSIDIARY_OF", "e2", "e1", score=2.0)])
    b = cg(1, [ent("e1", "Org", "Telenor Norge", "Telenor Norge")])
    g = merge([a, b], "")
    assert sorted(e.name for e in g.entities) == ["Telenor", "Telenor Norge"]
    assert [r.type for r in g.relations] == ["SUBSIDIARY_OF"]


def test_place_and_org_prefixes_do_not_force_join():
    for typ, short, long in (("Place", "York", "New York"), ("Org", "Volkswagen", "Volkswagen Group")):
        a = cg(0, [ent("e1", typ, short, short), ent("e2", typ, long, long)])
        assert len(merge([a], "").entities) == 2


def test_is_key_rejects_determiner_led_generic_and_function_words():
    assert not is_key("The Company") and not is_key("AND") and not is_key("The")
    assert is_key("Amundsen") and is_key("Tor Amundsen") and is_key("IBM")


def test_carry_forward_description_joins_the_earlier_name_with_its_head_noun():
    a = cg(0, [ent("e1", "Org", "Keswick Mountain Rescue Team"), ent("e2", "Org", "Lowther Foundation")])
    b = cg(1, [ent("e1", "Org", "The team", "The team"), ent("e2", "Org", "Ridgeway Outdoor")],
           [ChunkRelation("DONATED_TO", "e2", "e1")])
    g = merge([a, b], "")
    team = next(e for e in g.entities if e.name == "Keswick Mountain Rescue Team")
    assert len(g.entities) == 3 and [(r.source, r.target) for r in g.relations] == [
        (next(e.id for e in g.entities if e.name == "Ridgeway Outdoor"), team.id)]


def test_carry_forward_stays_apart_when_two_names_share_the_head_noun():
    a = cg(0, [ent("e1", "Org", "Keswick Rescue Team"), ent("e2", "Org", "Borrowdale Rescue Team")])
    b = cg(1, [ent("e1", "Org", "The team", "the team")])
    assert len(merge([a, b], "").entities) == 3


def test_carry_forward_never_joins_within_one_chunk():
    a = cg(0, [ent("e1", "Org", "Keswick Rescue Team"), ent("e2", "Org", "The team", "the team")])
    assert len(merge([a], "").entities) == 2
