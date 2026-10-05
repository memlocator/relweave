"""The legal ontology: charges, cases, courts and the people and organisations involved."""

from relweave.schema import register
from relweave.schema.base import Entity, Relation, Schema


class Person(Entity):
    """A human being: defendant, victim, judge, lawyer, investigator."""


class Org(Entity):
    """A court, law firm, prosecutor's office, regulator, police or other agency, or a company party to a case."""


class Place(Entity):
    """A country, region, city or other jurisdiction."""


class Charge(Entity):
    """An alleged offence or legal accusation: fraud, murder, price fixing, breach of contract."""


class Case(Entity):
    """A named lawsuit, trial, prosecution or legal proceeding."""


class ChargedWith(Relation[Person | Org, Charge]):
    """Person or Org was formally accused of the Charge by a prosecutor or authority; the outcome is CONVICTED_OF or ACQUITTED_OF, and suspicion without a charge is INVESTIGATED"""
    aliases = ("ACCUSED_OF", "INDICTED_FOR", "FACES_CHARGE")
    paraphrases = ("A prosecutor or authority formally brought the Charge against the source Person or Org", "Person or Org faces the Charge as a formal accusation, which comes before any verdict")
    date: str | None = None


class ConvictedOf(Relation[Person | Org, Charge]):
    """Person or Org was found guilty of the Charge, or pleaded guilty, by a court; a charge not yet decided is CHARGED_WITH"""
    aliases = ("FOUND_GUILTY_OF", "CONVICTED_FOR", "GUILTY_OF")
    paraphrases = ("A court found the Person or Org guilty of the Charge, or the source pleaded guilty to it", "Person or Org was convicted of the Charge, which is a decided outcome and not a pending charge")
    date: str | None = None


class AcquittedOf(Relation[Person | Org, Charge]):
    """Person or Org was cleared of the Charge by a court (acquitted, charge dismissed after trial); a dropped investigation or a charge withdrawn before trial is not an acquittal"""
    aliases = ("CLEARED_OF", "FOUND_NOT_GUILTY_OF", "EXONERATED_OF")
    paraphrases = ("A court cleared the Person or Org of the Charge", "Person or Org was acquitted, or the Charge against them was dismissed after trial")
    date: str | None = None


class DefendantIn(Relation[Person | Org, Case]):
    """Person or Org is the accused or sued party in the Case; the party that brings a civil suit is not a defendant, and a person harmed in the case is VICTIM"""
    aliases = ("ACCUSED_IN", "STANDS_TRIAL_IN", "SUED_IN")
    paraphrases = ("The accused or sued party in the Case is the Person or Org", "Person or Org stands trial or is sued in the Case")


class HeardBy(Relation[Case, Org]):
    """The Case was tried, heard or decided by the court or tribunal Org; the prosecutor, regulator or lawyers who took part are not the hearing body"""
    aliases = ("TRIED_BY", "DECIDED_BY", "ADJUDICATED_BY")
    paraphrases = ("The Case was tried by the court Org", "The court Org heard the Case")


class RepresentedBy(Relation[Person | Org, Person | Org]):
    """Source is legally represented by the target lawyer or law firm, in a case or generally; legal counsel only, so a business adviser or consultant is not; the target is the advocate and the source the client, and a lawyer and their firm each get an edge when both are named"""
    aliases = ("CLIENT_OF", "COUNSELLED_BY", "HAS_LAWYER")
    paraphrases = ("Source has the target lawyer or firm as legal counsel", "The target acts as attorney for the source")
    case: str | None = None


class Investigated(Relation[Org, Person | Org]):
    """The authority Org (police, prosecutor, regulator) investigated the target Person or Org; audits and inspections are not investigations unless the text calls them one; a formal accusation is CHARGED_WITH, and a court hearing is HEARD_BY"""
    aliases = ("PROBED", "INQUIRED_INTO", "LOOKED_INTO")
    paraphrases = ("The authority Org, such as the police, a prosecutor or a regulator, opened an investigation into the target Person or Org", "The source Org looked into the target Person or Org as a formal investigation, not a mere audit")
    date: str | None = None


class Sentenced(Relation[Person | Org, Person | Org]):
    """The source Person or Org (a court, or the judge named as passing sentence) imposed a sentence or penalty on the target Person or Org; term is the stated punishment (years, fine, probation), and a conviction without a stated sentence is only CONVICTED_OF"""
    aliases = ("PASSED_SENTENCE_ON", "IMPOSED_PENALTY_ON", "PUNISHED")
    paraphrases = ("The court or judge named as source imposed a punishment on the target Person or Org", "The source Person or Org passed a stated sentence, such as years, a fine or probation, on the target")
    term: str | None = None
    date: str | None = None


class Victim(Relation[Person | Org, Case]):
    """Person or Org was harmed by the conduct at issue in the Case (the injured party, the party defrauded, the person killed); the accused is DEFENDANT_IN"""
    aliases = ("VICTIM_IN", "INJURED_PARTY_IN", "HARMED_IN")
    paraphrases = ("Person or Org is the injured party in the Case", "Person or Org was harmed by what the Case is about")


LEGAL = register(Schema(name="legal",
    entities=[Person, Org, Place, Charge, Case],
    relations=[ChargedWith, ConvictedOf, AcquittedOf, DefendantIn, HeardBy, RepresentedBy, Investigated,
               Sentenced, Victim]))
