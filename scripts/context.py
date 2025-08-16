# context.py

class Context:
    """
    Holds both:
      1) `assumptions` (set[str]) for typing judgments, and
      2) `constraints` (list[str]) for algebraic simplification.
    """
    def __init__(self,
                 assumptions: set[str] = None,
                 constraints: list[str] = None):
        self.assumptions = set() if (assumptions is None) else set(assumptions)
        self.constraints = constraints or []

    # Add a new semantic assumption (for typing)
    def extend_assumption(self, assumption: str) -> "Context":
        new_assumps = set(self.assumptions)
        new_assumps.add(assumption)
        return Context(new_assumps, list(self.constraints))

    # Add a new symbolic constraint (for SymPy)
    def add_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)

    # Union for typing contexts (unions assumptions; concatenates constraints)
    def union(self, other: "Context") -> "Context":
        new_assumps = self.assumptions.union(other.assumptions)
        new_constraints = list(self.constraints) + list(other.constraints)
        return Context(new_assumps, new_constraints)

    def __eq__(self, other):
        return (isinstance(other, Context)
                and self.assumptions == other.assumptions
                and self.constraints == other.constraints)

    def __hash__(self):
        return hash((frozenset(self.assumptions), tuple(self.constraints)))

    def __repr__(self):
        a = ", ".join(sorted(self.assumptions))
        b = ", ".join(self.constraints)
        return f"Ctx(assumptions={{ {a} }}, constraints=[ {b} ])"

