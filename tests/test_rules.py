from relweave.graph import Entity, Graph, Occurrence, Relation
from relweave.rules import apply_rules


def graph(text, *ents, relations=()):
    es = []
    for i, (typ, name) in enumerate(ents, 1):
        k = text.find(name)
        es.append(Entity(f"e{i}", typ, name, [Occurrence(name, k, k + len(name), 0)]))
    return Graph(entities=es, relations=list(relations), text=text)


def rels(g):
    n = {e.id: e.name for e in g.entities}
    return {(n[r.source], r.type, n[r.target]) for r in g.relations}


def test_org_title_person():
    g = graph("Fraud Section Trial Attorney Asha Natarajan is handling the case.", ("Org", "Fraud Section"),
              ("Person", "Asha Natarajan"))
    assert rels(apply_rules(g)) == {("Asha Natarajan", "EMPLOYED_BY", "Fraud Section")}


def test_possessive_executive_title():
    g = graph("Kappahl's chief executive, Sandra Roos, said so.", ("Org", "Kappahl"), ("Person", "Sandra Roos"))
    assert rels(apply_rules(g)) == {("Sandra Roos", "EXECUTIVE_OF", "Kappahl")}


def test_title_person_of_org():
    g = graph("Special Agent Ryan Whalen of the FBI led it.", ("Person", "Ryan Whalen"), ("Org", "FBI"))
    assert rels(apply_rules(g)) == {("Ryan Whalen", "EMPLOYED_BY", "FBI")}


def test_party_adjective_before_and_after():
    a = apply_rules(graph("said Republican state senator Abel Maldonado.", ("Person", "Abel Maldonado")))
    b = apply_rules(graph("Alan Lowenthal, a Democratic state senator, agreed.", ("Person", "Alan Lowenthal")))
    assert rels(a) == {("Abel Maldonado", "MEMBER_OF", "Republican Party")}
    assert rels(b) == {("Alan Lowenthal", "MEMBER_OF", "Democratic Party")}


def test_existing_relation_and_sentence_break_block_rules():
    g = graph("Acme Corp. Chief executive Ana Varga spoke.", ("Org", "Acme Corp"), ("Person", "Ana Varga"))
    assert rels(apply_rules(g)) == set()  # a sentence ends between them
    g = graph("Acme's chief executive, Ana Varga, spoke.", ("Org", "Acme"), ("Person", "Ana Varga"),
              relations=[Relation("OWNS_STAKE_IN", "e2", "e1")])
    assert rels(apply_rules(g)) == {("Ana Varga", "OWNS_STAKE_IN", "Acme")}  # the model's relation stands
