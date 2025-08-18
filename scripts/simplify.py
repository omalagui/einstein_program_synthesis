# simplify.py

from context   import Context
from sympy import sympify, simplify, Symbol
from judgments import ContextWithMemory, Judgment
from functools import lru_cache

@lru_cache(maxsize=None)
def expr_to_str(expr):
    """
    Convert a SymPy Expr to its string form once, then cache.
    Subsequent calls with an equal Expr object are O(1).
    """
    return str(expr)
    
@lru_cache(maxsize=None)
def sympify_cached(term_str: str):
    return sympify(term_str)


def apply_substitution(j: Judgment, subs: dict) -> Judgment:
    """
    Given a JudgmentWithName j = (Ctx ⊢ name = term : typ), and a substitution dict subs,
    apply substitutions to the term and drop matching constraints.
    """
    # Create all required symbols locally
    local_symbols = {expr_to_str(k): Symbol(expr_to_str(k)) for k in subs.keys()}
    local_symbols.update({expr_to_str(v): Symbol(expr_to_str(v)) for v in subs.values() if isinstance(v, str)})

    # Parse the term into a SymPy expression using these symbols
    expr = sympify_cached(j.term, locals=local_symbols)

    # Apply substitutions and simplify
    for sym, val in subs.items():
        sym_expr = Symbol(expr_to_str(sym))
        val_expr = sympify_cached(val, locals=local_symbols) if isinstance(val, str) else val
        expr = simplify(expr.subs(sym_expr, val_expr))

    # Convert expression back to string
    new_term_str = expr_to_str(expr)

    # Drop constraints whose LHS was substituted
    def lhs_of_constraint(c: str) -> str:
        return c.split("=", 1)[0].strip()

    new_constraints = list(j.ctx.constraints)
    for sym in subs.keys():
        sym_str = expr_to_str(sym)
        new_constraints = [c for c in new_constraints if lhs_of_constraint(c) != sym_str]

    # Build new context and judgment
    new_ctx = Context(
        assumptions=set(j.ctx.assumptions),
        constraints=new_constraints
    )

    return Judgment(new_ctx, j.name, new_term_str, j.typ)

@lru_cache(maxsize=None)
def _pure_simplify(expr_str: str) -> str:
    expr = sympify_cached(expr_str)
    return expr_to_str(simplify(expr))

def simplify_term(ctx_mem: ContextWithMemory, j: Judgment) -> Judgment:
    """
    Simplifies the term of `j`, records the new judgment in ctx_mem
    (with `j` as its parent), and returns it.
    """
    # 1) simplify the Sympy AST of j.term
    # simplified_expr = simplify(sympify(j.term))
    simplified_expr = _pure_simplify(j.term)

    # 2) record it (parent=j, term=str(...), same type j.typ) and get back the new judgment
    new_j = ctx_mem.record(expr_to_str(simplified_expr), j.typ)

    # 3) return that freshly-recorded judgment

    # print(f"This is the parent of step 3: {j.parent}")

    # Return new judgment with the simplified term
    return Judgment(j.ctx, j.name, expr_to_str(simplified_expr), j.typ)

def substitute_nth_occurrence(j: Judgment, subs_dict: dict, occurrence_index: int) -> Judgment:
    """
    Replace the `occurrence_index`-th occurrence (in traversal order) of any symbol in `subs_dict`
    with its corresponding replacement expression.
    """
    expr = sympify_cached(j.term)

    # Count matching symbol hits as we walk the expression tree
    count = 0
    def replace_fn(node):
        nonlocal count
        if isinstance(node, Symbol) and expr_to_str(node) in subs_dict:
            if count == occurrence_index:
                count += 1
                return sympify_cached(subs_dict[expr_to_str(node)])
            count += 1
        return node

    new_expr = expr.replace(lambda x: True, replace_fn)

    return Judgment(j.ctx, j.name, expr_to_str(new_expr), j.typ)

def substitute_nth_occurrence_mod(ctx_mem: ContextWithMemory, j: Judgment, subs_dict: dict, occurrence_index: int) -> Judgment:
    """
    Replaces the `occurrence_index`-th occurrence (modulo total matches)
    of any symbol in `subs_dict` with its replacement.
    """
    expr = sympify_cached(j.term)
    symbols_to_replace = {Symbol(k): sympify_cached(v) for k, v in subs_dict.items()}

    # Step 1: Count total matches
    matches = []
    def find_matches(node):
        if isinstance(node, Symbol) and node in symbols_to_replace:
            matches.append(node)
        for arg in node.args:
            find_matches(arg)
    find_matches(expr)

    if not matches:
        return j  # No substitutions to make

    index_to_replace = occurrence_index % len(matches)
    match_counter = [0]

    # Step 2: Rebuild expression safely
    def replace_exactly_one(node):
        if isinstance(node, Symbol) and node in symbols_to_replace:
            if match_counter[0] == index_to_replace:
                match_counter[0] += 1
                return symbols_to_replace[node]
            match_counter[0] += 1
            return node
        elif node.args:
            new_args = tuple(replace_exactly_one(arg) for arg in node.args)
            return node.func(*new_args)
        else:
            return node

    new_expr = replace_exactly_one(expr)
    new_term = expr_to_str(new_expr)
    # print(f"This is new term: {new_term}")

    step = ctx_mem.record(new_term, j.typ)
    # print(f"This is substitute nth occurrrence mod step: {step}")

    # print(f"This is the parent of step 4: {j.parent}")


    return Judgment(j.ctx, j.name, new_term, j.typ)

def sympify_term(
    ctx_mem: ContextWithMemory,
    j: Judgment,
) -> Judgment:
    """
    Returns a new JudgmentWithName whose term is the sympified form of j.term,
    re-uses j.ctx and j.name, preserves j.typ, records the new term in ctx_mem,
    and links back to the original j as its parent.
    """
    # 1) Simplify the raw string
    # print(f"This is j.term: {j.term}")
    simplified_expr = sympify_cached(j.term)
    new_term = expr_to_str(simplified_expr)

    # 2) Record it in the memory context (under the same type as j)
    step = ctx_mem.record(new_term, j.typ)

    # 3) Return a brand new JudgmentWithName,
    #    pointing at the original j as its parent
    return Judgment(
        ctx    = j.ctx,
        name   = "",
        term   = new_term,
        typ    = j.typ
    )
