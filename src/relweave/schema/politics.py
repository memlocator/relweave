"""The politics ontology: parties, offices, elections and legislation."""

from relweave.schema import register
from relweave.schema.base import Entity, Relation, Schema


class Person(Entity):
    """A human being: politician, official, voter, commentator."""


class Org(Entity):
    """A political party, ministry, government, parliament, council or other public body."""


class Place(Entity):
    """A country, region, constituency, city or other territory."""


class Event(Entity):
    """An election, referendum or vote."""
    kind: str


class Law(Entity):
    """A bill, act, treaty, regulation or constitutional amendment."""


class MemberOfParty(Relation[Person, Org]):
    """Person is or was a member of the target Org, which must be a political party or parliamentary group; party membership only, since an office in a ministry, government or parliament is HOLDS_OFFICE; being elected for a party counts only when membership is stated"""
    aliases = ("PARTY_MEMBER", "BELONGS_TO_PARTY", "AFFILIATED_WITH_PARTY")
    paraphrases = ("Person is a member of the political party Org", "Person belongs to the party Org")
    start: str | None = None
    end: str | None = None


class HoldsOffice(Relation[Person, Org]):
    """Person holds or held a political or public office (minister, mayor, speaker, member of parliament, president); the target is the Org whose office it is (government, ministry, council, parliament), never a company board; a mayor or president of a Place links to that government Org if the text names it, else no edge; title is the office; "MP for Leeds" gives HOLDS_OFFICE to the parliament and REPRESENTS to Leeds, both; mere party membership is MEMBER_OF_PARTY"""
    aliases = ("OFFICEHOLDER_OF", "SERVES_IN", "HOLDS_POST_IN")
    paraphrases = ("Person holds or held a public post such as minister, mayor, speaker or member of parliament in the government body Org", "Person serves or served in a political office of the target Org")
    title: str | None = None
    start: str | None = None
    end: str | None = None


class CandidateIn(Relation[Person, Event]):
    """Person stood as a candidate in the election or referendum Event, whether or not they won; the result is WON_ELECTION"""
    aliases = ("RAN_IN", "STOOD_IN", "STOOD_FOR_ELECTION_IN")
    paraphrases = ("Person ran as a candidate in the election Event", "Person stood for election in the Event, win or lose")


class WonElection(Relation[Person | Org, Event]):
    """The source won its own contest in the election Event: a candidate who was elected or a party that took the most votes or seats, by plurality or majority as the text states the win; an MP winning a seat in a general election counts; losing candidates are only CANDIDATE_IN"""
    aliases = ("WON_VOTE", "ELECTED_IN", "VICTOR_IN")
    paraphrases = ("Person or Org won the election Event", "Person or Org came first in the election Event")


class Represents(Relation[Person, Place]):
    """Person represents the Place as an elected or appointed representative (member of parliament for a constituency, senator for a state, ambassador to a country), not merely lives or was born there"""
    aliases = ("REPRESENTATIVE_OF", "SITS_FOR", "REPRESENTATIVE_FOR")
    paraphrases = ("Person is the elected or appointed representative of the Place, such as its member of parliament, senator or ambassador", "Person speaks for the Place in office, which differs from merely living there")
    start: str | None = None
    end: str | None = None


class Proposed(Relation[Person | Org, Law]):
    """Person or Org introduced or sponsored the Law (bill, motion, treaty) in the legislature; passing it is PASSED, and a mere opinion on it is not a relation"""
    aliases = ("SPONSORED", "INTRODUCED", "INTRODUCED_BILL")
    paraphrases = ("Person or Org put the Law forward as its sponsor or introducer", "The source introduced the bill, motion or treaty Law in the legislature")
    date: str | None = None


class Passed(Relation[Org, Law]):
    """The legislative body Org adopted, enacted or ratified the Law; the proposer of the Law is PROPOSED, and a bill that was rejected is not passed"""
    aliases = ("ENACTED", "ADOPTED", "RATIFIED")
    paraphrases = ("The legislative body Org approved the Law so that it was enacted", "Org adopted, enacted or ratified the Law, which excludes a rejected bill")
    date: str | None = None


class GovernsIn(Relation[Org, Place]):
    """Org (government, ministry, council, party in power) holds governing authority over the Place; an Org that merely has offices or members there does not"""
    aliases = ("RULES", "ADMINISTERS", "HAS_AUTHORITY_OVER")
    paraphrases = ("Org is the governing authority of the Place", "Org rules or administers the Place, which is more than having members or offices there")
    start: str | None = None
    end: str | None = None


class AlliedWith(Relation[Org, Org]):
    """Two Orgs (parties, governments, blocs) are allies or coalition partners, which the text must state explicitly; shared membership of an organisation alone is not an alliance, and a rivalry or a merger is not one either"""
    aliases = ("COALITION_PARTNER_OF", "ALLY_OF", "PARTNERED_WITH")
    paraphrases = ("Two Orgs are explicitly stated to be allies or coalition partners", "One Org is the coalition partner or ally of the other Org, and the link works both ways")
    symmetric = True
    start: str | None = None
    end: str | None = None


class SucceededInOffice(Relation[Person, Person]):
    """Source Person directly followed the target Person in the same office; the office is stated in the text, and a person who merely came later or is a relative is not a successor"""
    aliases = ("SUCCEEDED", "SUCCESSOR_OF", "TOOK_OVER_FROM")
    paraphrases = ("Source took over the office the target Person previously held", "Source Person replaced the target Person in office")
    title: str | None = None
    date: str | None = None


POLITICS = register(Schema(name="politics",
    entities=[Person, Org, Place, Event, Law],
    relations=[MemberOfParty, HoldsOffice, CandidateIn, WonElection, Represents, Proposed, Passed,
               GovernsIn, AlliedWith, SucceededInOffice]))
