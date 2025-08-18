# typing_rules.py

"""
This module defines the core typing rules for the frozen‐light DSL.
Each rule is a Python function that takes one or two existing Judgment objects
and either returns a new Judgment or None if the premises do not hold.

Dependencies (make sure these files are in the same directory):
  - einstein_types.py   (defines move_type, frozen_type, bottom_type, etc.)
  - terms.py           (defines Term, Sin, Wave)
  - judgments.py       (defines Context, Judgment)
"""

from einstein_types import Move, Frozen, Real, Bot, IntersectionType
from judgments import ContextWithMemory, Judgment
from sympy import Symbol, Expr, Function, FunctionClass
from typing import Callable, Any, Dict, Optional
import re
from einstein_types  import Move, Frozen
from functools import wraps
from sympy import symbols, Symbol, Function, sympify, simplify
from sympy.parsing.sympy_parser import parse_expr
from context import Context
from functools import lru_cache


@lru_cache(maxsize=None)
def expr_to_str(expr):
    """
    Convert a SymPy Expr to its string form once, then cache.
    Subsequent calls with an equal Expr object are O(1).
    """
    return str(expr)

def distinction_rule(j1: Judgment, j2: Judgment) -> list[Judgment] | None:
    """
    Distinction‐choice rule:
      If
        j1 = (Γ₁ ⊢ t : A)
        j2 = (Γ₂ ⊢ t : B)
      and t is definitionally the same in both,
      then under Γ₁ ∪ Γ₂, t can only have type A or type ⊥.

    Returns:
      - If j1.term != j2.term: None
      - Else let Γ = j1.ctx.union(j2.ctx).
        • If j1.typ == j2.typ == A, returns [ Judgment(Γ, t, A) ].
        • If j1.typ = A, j2.typ = B with A ≠ B, returns:
              [ Judgment(Γ, t, A),  Judgment(Γ, t, bottom_type) ].
    """

    # 1) Must be the same term
    if sympify(j1.term) != sympify(j2.term):
        return None

    # 2) Merge the contexts
    combined_ctx = j1.ctx.union(j2.ctx)

    # 3a) If the two types agree, no conflict: just return one judgment
    if j1.typ == j2.typ:
        return [Judgment(combined_ctx, j1.term, j1.typ)]

    # 3b) If they disagree, return two possibilities under the merged context:
    #     (Γ ⊢ t : A)  or  (Γ ⊢ t : ⊥)
    return [
        Judgment(combined_ctx, j1.name, j1.term, j1.typ),
        Judgment(combined_ctx, j2.name, j2.term, Bot)
    ]

def concept_change_rule(ctx_mem: ContextWithMemory,
    j1: Judgment,
    j2: Judgment
) -> Judgment:
    """
    Implements the Concept Changing Rule (Section 5.3):

        Γ₁ ⊢ x : A      Γ₂ ⊢ y : B      (and x and y refer to the same Var)
        ────────────────────────────────────────────────────────
           (Γ₁ ∪ Γ₂) ⊢ x : (A ∧ B)

    Preconditions:
      1. j1.term and j2.term are the “same” variable (i.e. same Var.name).
      2. The two contexts Γ₁, Γ₂ do not assign conflicting types to any var.
    Returns:
      A new Judgment in the combined context Γ₁ ∪ Γ₂, with the term = j1.term,
      and the intersection type = (A ∧ B) where A=j1.typ, B=j2.typ.
    """
    # 1) Check that both judgments refer to the same term—i.e. same variable name.
    # if not (isinstance(j1.term, Var) and isinstance(j2.term, Var)):
    #     raise ValueError(
    #         f"Concept Changing Rule only applies to variables. "
    #         f"Got {j1.term} and {j2.term}."
    #     )

    if j1.term != j2.term:
        raise ValueError(
            f"Term mismatch: first says term={j1.term}, second says term={j2.term}. "
            "They must be the same variable."
        )

    # 2) Construct the intersection type A ∧ B
    new_type = IntersectionType(left=j1.typ, right=j2.typ)

    # 3) Merge contexts Γ₁ ∪ Γ₂
    combined_ctx = j1.ctx.union(j2.ctx)

    step = ctx_mem.record(j1.term, new_type)

    # print(f"This is the parent of step 5: {j1.parent}")

    # 4) Return the new judgment: (Γ₁ ∪ Γ₂) ⊢ x : (A ∧ B)
    return Judgment(ctx=combined_ctx, name=j1.name, term=j1.term, typ=new_type)

def forward_property_preserving_rule(
    ctx_mem: ContextWithMemory,
    j:   Judgment,
    func_str: str,
    sympy_symbols: dict = None
) -> Judgment:
    """
    Replace the named variable in func_str using def_j.term,
    record the result in ctx_mem (with def_j as parent), and
    return the new JudgmentWithName.
    """
    # 1) start with a fresh or provided symbol table
    if sympy_symbols is None:
        sympy_symbols = {}

    # 2) make sure the definition-variable itself is in the table
    var = j.name
    if var:
        if f"{var}(" in func_str:
            sympy_symbols.setdefault(var, Function(var))
        else:
            sympy_symbols.setdefault(var, Symbol(var))

    # 3) scan func_str for all other identifiers, too
    tokens = re.findall(r"\b[a-zA-Z_]\w*\b", func_str)
    for tok in tokens:
        # skip ones we've already added
        if tok in sympy_symbols:
            continue
        # heuristically decide Function vs Symbol
        if f"{tok}(" in func_str:
            sympy_symbols[tok] = Function(tok)
        else:
            sympy_symbols[tok] = Symbol(tok)

    # 4) parse the RHS expression
    expr = parse_expr(func_str, local_dict=sympy_symbols)

    # 5) build the replacement (also using that same table)
    replacement = sympify(j.term, locals=sympy_symbols)

    # 6) do the subs
    replaced = expr.subs(sympy_symbols[var], replacement)

    # 7) stringify and record
    new_j = ctx_mem.record(expr_to_str(replaced), j.typ)

    return Judgment(
        ctx    = j.ctx,
        name   = "",
        term   = expr_to_str(replaced),
        typ    = j.typ
    )

def backward_property_preserving_rule(
    ctx_mem: ContextWithMemory,
    j: Judgment
) -> Judgment:
    """
    Step back one generation: drop the last record and return
    a new judgment that reuses the original name & type but
    with the previous term.
    """
    log = ctx_mem.step_log 

    # print(f"This is log: {ctx_mem.step_log}")
    # print(f"This is log: {log}")

    log.pop(-1)

    # print(f"This is log now: {log}")

    # print(f"This is log.shape: {len(log)}")

    last_in_memory = log[-1] #find last term in memory after popping
    # print(f"This is last in memory: {log[-1]}")
    
    last_term = last_in_memory

    # prev_j = ctx.rewind_one()
    # build a fresh JudgmentWithName reusing the name & type
    return Judgment(
        ctx    = j.ctx,
        name   = j.name,
        term   = last_term,
        typ    = j.typ
    )