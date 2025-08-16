# judgments.py

from __future__ import annotations
from context import Context       # our unified Context (with assumptions & constraints)
from einstein_types import BaseType  # for type‐tags like move_type, frozen_type, etc.
from sympy import sympify

class Judgment:
    """
    A typing judgment of the form:

        Γ ⊢ name = term : type   (if name is provided)
        Γ ⊢ term : type          (if name is empty or None)
    """
    def __init__(self, ctx: Context, name: str, term: str, typ: BaseType,
        parent: Judgment | None = None):
        self.ctx = ctx
        self.name = name.strip() if name else ""  # Normalize empty name
        self.term = sympify(term)
        self.typ = typ
        self.parent = parent     # the judgment from which this one derived


    def __repr__(self):
        if self.name:
            return f"{self.ctx} ⊢ {self.name} = {self.term} : {self.typ}"
        else:
            return f"{self.ctx} ⊢ {self.term} : {self.typ}"

    def __eq__(self, other):
        return (
            isinstance(other, Judgment) and
            self.ctx == other.ctx and
            self.name == other.name and
            self.term == other.term and
            self.typ == other.typ
        )

    def __hash__(self):
        return hash((self.ctx, self.name, self.term, self.typ))

from typing import List
from judgments import Judgment

class ContextWithMemory(Context):
    """
    A memoryful context that records a linear trace of judgments explicitly.
    No global state, no decorators required – you pass `ctx` into every rule.
    """

    def __init__(self):
        self.judgment_count: int = 0
        self.step_log: List[str] = []

    def __repr__(self):
        if not self.step_log:
            return "Ctx(assumptions={})"
        return "Ctx(assumptions={ " + ", ".join(self.step_log) + " })"

    def start(self, j: Judgment) -> str:
        """
        Seed the memory with an existing JudgmentWithName `root`.
        Returns a fresh copy of `root` now living in this memory context.
        """
        # 1) Reset everything
        self.counter = 1
        self.step_log.clear()

        # 2) Record it as the very first step
        start_term = f"{j.term}"
        self.step_log.append(start_term)

        # 3) Now `counter` is 1, so next record() will use step2
        return start_term

    def record(self, term: str, typ: BaseType) -> str:
        """
        Record a new step under this memory‐context,
        appending to judgment_log, step_log, and base.constraints.
        Returns the brand‐new JudgmentWithName object.
        """
        # pick next fresh step name
        self.judgment_count += 1
        name = f"step{self.judgment_count}"

        # record the printable assumption and also push into base-Context.constraints
        recorded_term = f"{term}"
        self.step_log.append(recorded_term)

        return recorded_term
        
    def clone(self):
        # shallow‐copy only the tiny mutable bits
        new = ContextWithMemory()
        new.assumptions = self.assumptions.copy()   # if it’s a set/dict
        new.memory      = self.memory.copy()        # if it’s a list
        return new




