"""The business ontology (Nordic/European business Wikipedia), as Pydantic classes. Single source of
the business schema."""

from relweave.schema import register
from relweave.schema.base import Entity, Relation, Schema


class Person(Entity):
    """A human being."""


class Org(Entity):
    """A company, institution, family acting as owner, or other organised body; publications, imprints and brands are Orgs."""


class Object(Entity):
    """A physical thing: vehicle, vessel, building, artwork."""
    kind: str


class Place(Entity):
    """A country, region, city or other location."""


class Coordinate(Entity):
    """A numeric map coordinate."""
    lat: str
    lon: str


class Event(Entity):
    """Something that happened at a time: a sale, meeting, election, ceremony."""
    kind: str


class EmployedBy(Relation[Person, Org]):
    """Person works for Org in a non-executive role"""
    aliases = ("WORKS_FOR", "EMPLOYEE_OF", "STAFF_OF")
    paraphrases = ("Person has a job at Org below executive level", "Person is employed at Org, but not as an executive")
    role: str | None = None
    start: str | None = None
    end: str | None = None


class ExecutiveOf(Relation[Person, Org]):
    """Person holds an executive title at Org (CEO, CFO, managing director) or chairs its board or supervisory board (title chairman, vice chairman, honorary chairman)"""
    aliases = ("EXECUTIVE_AT", "LEADS", "OFFICER_OF")
    paraphrases = ("Person is a top executive of Org, such as chief executive, finance chief or managing director, or the chair of its board", "Person runs Org or chairs its board, holding a title such as CEO, managing director or chairman")
    title: str | None = None
    start: str | None = None
    end: str | None = None


class BoardMemberOf(Relation[Person, Org]):
    """Person sits on the board of Org as a member; a board chair is EXECUTIVE_OF instead"""
    aliases = ("DIRECTOR_OF", "SITS_ON_BOARD_OF", "BOARD_SEAT_AT")
    paraphrases = ("Person sits on the board of directors of Org as an ordinary member", "Person sits on the board of Org with a seat but without a leading role")
    start: str | None = None
    end: str | None = None


class MemberOf(Relation[Person, Org]):
    """Person is a member of Org (club, party, association)"""
    aliases = ("BELONGS_TO", "AFFILIATED_WITH", "MEMBERSHIP_IN")
    paraphrases = ("Person belongs to the club, party or association Org", "Org counts Person among its members")
    start: str | None = None
    end: str | None = None


class FamilyOf(Relation[Person, Person]):
    """Family tie; kind is what the target is to the source: spouse, parent, child or sibling"""
    aliases = ("RELATIVE_OF", "KIN_OF", "RELATED_TO")
    paraphrases = ("The two Persons are close relatives, such as spouses, parents, children or siblings", "Source and target Person are family, and kind says what the target is to the source")
    kind: str | None = None
    symmetric = True


class AssociateOf(Relation[Person, Person]):
    """Persons described as associates, partners or close contacts"""
    aliases = ("CLOSE_TO", "CONFIDANT_OF", "ACQUAINTED_WITH")
    paraphrases = ("The two Persons are named as associates, partners or close contacts of each other", "One Person is described as an associate or close contact of the other Person, and the link works both ways")
    symmetric = True


class MetWith(Relation[Person, Person]):
    """Two persons met in person"""
    aliases = ("MET", "MET_IN_PERSON", "HAD_MEETING_WITH")
    paraphrases = ("Two Persons met face to face", "One Person and another Person were together in a meeting, and the link works both ways")
    date: str | None = None
    symmetric = True


class CommunicatedWith(Relation[Person, Person]):
    """Two persons communicated (call, email, message, letter)"""
    aliases = ("CONTACTED", "CORRESPONDED_WITH", "SPOKE_WITH")
    paraphrases = ("Two Persons were in contact by phone, email, message or letter", "One Person called, wrote to or messaged the other Person, and the link works both ways")
    date: str | None = None
    channel: str | None = None
    symmetric = True


class LocatedIn(Relation[Person | Org | Object | Place, Place]):
    """Source is or was resident or situated in the Place (dated if the text says when), or a Place lies within another; not a birthplace, not a market; an Org that is based in a Place is HEADQUARTERED_IN"""
    aliases = ("SITUATED_IN", "RESIDES_IN", "LIES_IN")
    paraphrases = ("Person, Org or Object is or was resident or situated in the Place, or a Place lies within another Place", "The source lives or sits in the target Place, or is a smaller Place inside it, and a birthplace or a market does not count")
    start: str | None = None
    end: str | None = None


class BornIn(Relation[Person, Place]):
    """Person was born in the Place"""
    aliases = ("BIRTHPLACE", "BORN_AT", "NATIVE_OF")
    paraphrases = ("The birthplace of Person is the Place", "Person was born in the town, city or country Place")
    date: str | None = None


class OperatesIn(Relation[Org, Place]):
    """Org has operations, offices, plants, stores or sales in the Place"""
    aliases = ("ACTIVE_IN", "OPERATES_AT", "DOES_BUSINESS_IN")
    paraphrases = ("Org does business in the Place through operations, offices, plants, stores or sales", "Org does business in the Place (offices, plants, stores or sales)")
    start: str | None = None
    end: str | None = None


class HeadquarteredIn(Relation[Org, Place]):
    """Org has its headquarters in, or is based in, the Place"""
    aliases = ("BASED_IN", "HQ_IN", "SEATED_IN")
    paraphrases = ("The head office of Org is in the Place", "The Place is the seat or home base of Org")


class Founded(Relation[Person | Org, Org]):
    """Person or Org founded or co-founded the Org; when Orgs merge to form a new Org, each merging Org FOUNDED it"""
    aliases = ("ESTABLISHED", "CREATED", "SET_UP")
    paraphrases = ("Person or Org started the target Org, alone or with others", "The source founded or co-founded the target Org, and Orgs that merge into a new Org each founded it")
    date: str | None = None


class OwnsStakeIn(Relation[Person | Org, Org]):
    """Person or Org owns shares in, a stake in, or controls through ownership the Org; share is the stated fraction"""
    aliases = ("SHAREHOLDER_OF", "HOLDS_SHARES_IN", "STAKEHOLDER_IN")
    paraphrases = ("Person or Org holds shares or an ownership stake in the target Org", "The source controls the target Org through ownership, and share is the stated fraction")
    share: str | None = None
    start: str | None = None
    end: str | None = None


class SubsidiaryOf(Relation[Org, Org]):
    """Source Org is a subsidiary, division or brand of the target Org; publications, magazines, newspapers, imprints and product brands are Orgs linked this way, never Objects; in a chain of owners, link only to the nearest parent the text states"""
    aliases = ("DIVISION_OF", "UNIT_OF", "BRAND_OF")
    paraphrases = ("Org is a subsidiary, division or brand belonging to the parent Org", "The source Org sits under the target Org as a unit, imprint or product brand, and the target is the nearest parent the text names")
    start: str | None = None
    end: str | None = None


class Acquired(Relation[Person | Org, Org]):
    """Person or Org bought or took over the target Org; only when the text names the buyer"""
    aliases = ("BOUGHT", "TOOK_OVER", "PURCHASED")
    paraphrases = ("Person or Org purchased or took over the target Org, and the text names the buyer", "The source took control of the target Org by buying it, and the text names the source")
    date: str | None = None


class HasCoordinate(Relation[Place, Coordinate]):
    """Place has the stated numeric coordinate"""
    aliases = ("AT_COORDINATE", "GEOLOCATED_AT", "LOCATED_AT_COORDINATE")
    paraphrases = ("The Place sits at the stated map coordinate", "The numeric Coordinate gives the position of the source Place")


class OwnsObject(Relation[Person | Org, Object]):
    """Person or Org owns the Object (a physical thing: vehicle, vessel, building, artwork); brands and publications are Orgs"""
    aliases = ("OWNS", "POSSESSES", "HOLDS_TITLE_TO")
    paraphrases = ("Person or Org is the owner of the Object, a vehicle, vessel, building or artwork", "The source holds ownership of the target physical thing, whereas brands and publications are Orgs")
    start: str | None = None
    end: str | None = None


class TransferredObject(Relation[Event, Object]):
    """The Event transferred the Object (sale, delivery, seizure)"""
    aliases = ("HANDED_OVER", "DELIVERED", "CONVEYED")
    paraphrases = ("The Event, such as a sale, delivery or seizure, moved the Object to a new holder", "In the Event the Object was transferred from one party to another")


class ParticipatedIn(Relation[Person | Org, Event]):
    """Person or Org took part in the Event; role is buyer, seller, host, attendee"""
    aliases = ("TOOK_PART_IN", "ATTENDED", "INVOLVED_IN")
    paraphrases = ("Person or Org was a participant in the Event, in a role such as buyer, seller, host or attendee", "Person or Org was involved in the Event, in a role such as buyer, seller, host or attendee")
    role: str | None = None


class HeldAt(Relation[Event, Place]):
    """The Event took place at the Place"""
    aliases = ("TOOK_PLACE_AT", "OCCURRED_AT", "HAPPENED_AT")
    paraphrases = ("The Event happened at the Place", "The Place is where the Event occurred")


BUSINESS = register(Schema(name="business",
    entities=[Person, Org, Object, Place, Coordinate, Event],
    relations=[EmployedBy, ExecutiveOf, BoardMemberOf, MemberOf, FamilyOf, AssociateOf, MetWith,
               CommunicatedWith, LocatedIn, BornIn, OperatesIn, HeadquarteredIn, Founded, OwnsStakeIn,
               SubsidiaryOf, Acquired, HasCoordinate, OwnsObject, TransferredObject, ParticipatedIn, HeldAt]))
