# synthesis_state.py

"""
Dataclass for a search node (ctx, cursor, depth, trace).
"""

from dataclasses import dataclass
from einstein_types import Art, Emp
from functools import lru_cache


GOAL_TERM = "t - u*x/c**2"
GOAL_TYPE = Art & Emp      # intersection of Art and Emp

@lru_cache(maxsize=None)
def expr_to_str(expr):
    """
    Convert a SymPy Expr to its string form once, then cache.
    Subsequent calls with an equal Expr object are O(1).
    """
    return str(expr)

def _sig(state):
    """Signature we use for hashing & equality: just term + type."""
    return (expr_to_str(state.cursor.term), expr_to_str(state.cursor.typ))

@dataclass(frozen=True)
class State:
    ctx: object
    cursor: object
    depth: int
    trace: tuple[str, ...]

    # ---------------------------------------------------------------------
    # def is_goal(self) -> bool:
    #     return (str(self.cursor.term) == GOAL_TERM
    #             and str(self.cursor.typ) == str(Art & Emp))

    # # Required so we can drop State objects in a set()
    # def __hash__(self):
    #     return hash((str(self.cursor.term), str(self.cursor.typ)))

    def is_goal(self) -> bool:
        return (
            expr_to_str(self.cursor.term) == GOAL_TERM
            and self.cursor.typ == GOAL_TYPE   # compare objects, not strings
        )

    # --- custom equality / hashing (ignore ctx) ---------------------------
    def __hash__(self):
        return hash(_sig(self))

    def __eq__(self, other):
        return isinstance(other, State) and _sig(self) == _sig(other)
