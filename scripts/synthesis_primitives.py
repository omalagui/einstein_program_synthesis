"""
Collect one-step actions that the search can call.
Edit FUNC_STR, SUBS_1, TARGET_JUDGMENT to match your demo script.
"""

from copy import deepcopy

from typing_rules import forward_property_preserving_rule, concept_change_rule, backward_property_preserving_rule
from simplify import sympify_term, simplify_term, substitute_nth_occurrence_mod
from judgments import Judgment
from context import Context
from einstein_types import Art, Emp

# --- CONSTANTS ---------------------------------------------------------------

FUNC_STR = "f(w*tprime - k*y)"          # whatever you used in Step 1
SUBS_1   = {"w": "k*c"}                 # Step 4 in your demo
# Dummy judgment only to feed CHG; you can also import the real one
ctx_tmp       = Context(assumptions={"Stellar aberration"})
TARGET_JUDGMENT = Judgment(ctx_tmp, "", "f(-k*y + t*w - k*u*x/c)", Emp)


# --- PRIMITIVE WRAPPERS ------------------------------------------------------

def _wrap(fn):
    """Return a version that deep-copies ctx before calling `fn`."""
    def _inner(ctx, j, *extra):
        new_ctx = deepcopy(ctx)
        new_j   = fn(new_ctx, j, *extra)
        return new_ctx, new_j
    return _inner

PRIMITIVES = [
    ("FWD",  _wrap(lambda ctx, j: forward_property_preserving_rule(
                              ctx, j, FUNC_STR))),
    ("SIM",  _wrap(sympify_term)),
    ("SIM+", _wrap(simplify_term)),
    ("SUB1", _wrap(lambda ctx, j: substitute_nth_occurrence_mod(
                              ctx, j, SUBS_1, 1))),
    ("CHG",  _wrap(lambda ctx, j: concept_change_rule(
                              ctx, j, TARGET_JUDGMENT))),
    ("BWD",  _wrap(backward_property_preserving_rule)),
]
