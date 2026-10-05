"""The sports ontology, for evaluation only: it is held out of all training."""

from relweave.schema import register
from relweave.schema.base import Entity, Relation, Schema


class Person(Entity):
    """A human being: athlete, coach, official, author, actor."""


class Org(Entity):
    """A club, team, league, federation or other sporting body."""


class Place(Entity):
    """A country, city, stadium or other location."""


class Event(Entity):
    """A competition, tournament, season or single match."""
    kind: str


class Work(Entity):
    """A book, film, album or other created work."""


class PlaysFor(Relation[Person, Org]):
    """Person is or was a player or athlete on the roster of the club or team Org, a national team being an Org too; a loan is TRANSFERRED_TO and then PLAYS_FOR the new club; a coach is COACHES"""
    aliases = ("PLAYER_FOR", "ATHLETE_AT", "ON_ROSTER_OF")
    paraphrases = ("Person is a player of the club or team Org", "Person played for the club or team Org")
    start: str | None = None
    end: str | None = None


class Coaches(Relation[Person, Org]):
    """Person manages, trains or coaches the team or club Org (head coach, manager, assistant), a national team being an Org too; a player-coach also PLAYS_FOR it"""
    aliases = ("MANAGES", "MANAGER_OF", "TRAINS")
    paraphrases = ("Person is the head coach, manager or assistant coach of the team Org", "Person trains and leads the club or team Org")
    start: str | None = None
    end: str | None = None


class CompetedIn(Relation[Person | Org, Event]):
    """Person or Org took part as a competitor in the Event (entered, played, raced), whatever the result; a team's participation is the team's edge; winning is WON, and a spectator or organiser did not compete"""
    aliases = ("PLAYED_IN", "ENTERED", "TOOK_PART_IN")
    paraphrases = ("Person or Org took part in the competition Event", "Person or Org was a competitor in the Event")


class Won(Relation[Person | Org, Event]):
    """Person or Org won the Event (took the title, the match or first place); a team's win is the team's edge, and players get WON only when the text credits them; finishing second or merely taking part is only COMPETED_IN"""
    aliases = ("TOOK_TITLE", "CHAMPION_OF", "WINNER_OF")
    paraphrases = ("Person or Org won the competition Event", "Person or Org came first in the Event")


class HeldIn(Relation[Event, Place]):
    """The Event took place in or at the most specific Place named (the venue); the city too only if the text separately states it as the location; where the competitors come from is not where it was held"""
    aliases = ("HOSTED_IN", "PLAYED_AT", "TOOK_PLACE_IN")
    paraphrases = ("The Event took place in the Place", "The Event was hosted at the Place")


class Authored(Relation[Person, Work]):
    """Person created the Work (wrote the book or screenplay, composed the album); the creator is AUTHORED and performing on it is PERFORMED_IN, both when the text says both"""
    aliases = ("WROTE", "CREATED_WORK", "COMPOSED")
    paraphrases = ("Person is the author or creator of the Work, for example its writer or composer", "Person made the Work as its creator, and appearing in it does not count")
    date: str | None = None


class PerformedIn(Relation[Person, Work]):
    """Person acted, appeared or performed in the Work (film, series, album as a musician); the creator of the Work is AUTHORED, and a person who both created and performed gets both edges when the text says both"""
    aliases = ("STARRED_IN", "APPEARED_IN", "ACTED_IN")
    paraphrases = ("Person acted or performed in the Work, such as a film, series or album", "Person appears in the Work as a performer, and writing the Work does not count by itself")
    role: str | None = None


class BasedIn(Relation[Org, Place]):
    """The club or body Org has its home, stadium or headquarters in the most specific Place named; the city too only if the text separately states it; the place of a single match is HELD_IN"""
    aliases = ("HOME_IN", "HEADQUARTERED_IN", "HOME_GROUND_IN")
    paraphrases = ("Org has its home or headquarters in the Place", "The Place is the home base of Org")


class TransferredTo(Relation[Person, Org]):
    """Person moved to the target club or team Org by transfer, trade, signing or loan, a national team being an Org too; date is when, and the person then also PLAYS_FOR the Org"""
    aliases = ("SIGNED_FOR", "MOVED_TO", "JOINED")
    paraphrases = ("Person moved to the club or team Org by transfer, signing or loan", "The target Org acquired Person through a transfer, trade or signing, and Person then plays for it")
    date: str | None = None


SPORTS = register(Schema(name="sports",
    entities=[Person, Org, Place, Event, Work],
    relations=[PlaysFor, Coaches, CompetedIn, Won, HeldIn, Authored, PerformedIn, BasedIn, TransferredTo]))
