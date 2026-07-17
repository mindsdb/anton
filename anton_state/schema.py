"""Declarative storage schema for state — the single source of truth.

The manifest file (JSON) is read both by artifact code (StateSchema.from_manifest)
and by the publish/provisioning pipeline (subplans #2/#4). One file means the
schema in code and the provisioned table schema cannot diverge by construction.
The file is read without executing any code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

# v1: string keys only. Numeric/binary keys would require support across the
# whole chain (pk: str, validation, begins_with) — deferred.
AttrType = Literal["S"]


class Attr(BaseModel):
    name: str
    type: AttrType = "S"


class Index(BaseModel):
    name: str
    pk: Attr
    sk: Attr | None = None


class StateSchema(BaseModel):
    version: int = 1
    pk: Attr
    sk: Attr | None = None
    gsis: list[Index] = []
    ttl_attribute: str | None = None

    def key_attrs(self) -> set[str]:
        names: set[str] = {self.pk.name}
        if self.sk:
            names.add(self.sk.name)
        for gsi in self.gsis:
            names.add(gsi.pk.name)
            if gsi.sk:
                names.add(gsi.sk.name)
        return names

    @classmethod
    def from_manifest(cls, path: str | Path) -> "StateSchema":
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)

    def to_manifest(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
