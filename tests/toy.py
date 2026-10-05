"""A toy schema shared by the tests: three entity types, three relation types (one symmetric, one with a union
source)."""
from relweave.schema import Entity, Relation, Schema


class Person(Entity):
    """A human being."""


class Org(Entity):
    """A company or other organised body."""


class Place(Entity):
    """A location."""


class OperatesIn(Relation[Org, Place]):
    """The Org has operations in the Place."""
    start: str | None = None
    aliases = ("ACTIVE_IN",)


class FamilyOf(Relation[Person, Person]):
    """Family tie."""
    kind: str | None = None
    symmetric = True


class LocatedIn(Relation[Person | Org | Place, Place]):
    """Situated in the Place."""


S = Schema(name="toy", entities=[Person, Org, Place], relations=[OperatesIn, FamilyOf, LocatedIn])
