"""What one chunk yields before the chunks are merged: entities with document-offset mentions, and relations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mention:
    text: str
    start: int | None = None  # document offset; None when the string was not found in the chunk text


@dataclass
class ChunkEntity:
    id: str  # chunk-local, e.g. "e1"
    type: str
    name: str
    mentions: list[Mention]
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class ChunkRelation:
    type: str
    source: str  # chunk-local entity ids
    target: str
    modality: str = "asserted"
    attributes: dict[str, str] = field(default_factory=dict)
    score: float | None = None  # the pair head's margin for this relation; None when it never scored it
    origin: str = "generator"  # "generator" (written by the generator, kept by the head) or "head" (added by it)
    sentence: tuple[int, int] | None = None  # document offsets of the evidence sentence, when known


@dataclass
class ChunkGraph:
    chunk: int  # chunk index
    entities: list[ChunkEntity] = field(default_factory=list)
    relations: list[ChunkRelation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start: int = 0  # the chunk's document offsets
    end: int = 0
