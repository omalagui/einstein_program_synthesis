#!/usr/bin/env python3
# pure_bayes_search.py
# Pure Bayesian (Thompson sampling) search — split for import-from-notebook use.

from copy import deepcopy
import random
import difflib
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Optional, List
import time
import cProfile, pstats, io

import numpy as np

from context import Context
from judgments import ContextWithMemory, Judgment
from einstein_types import Art, Emp
from synthesis_state import State
from synthesis_primitives import PRIMITIVES

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MAX_SAMPLES  = 50      # #200 #1000
CHAIN_LENGTH = 8
USE_BAYESIAN = True      # Use Bayesian (Thompson sampling) search

# Trade-off weights (must sum to 1.0)
TERM_WEIGHT  = 0.7
TYPE_WEIGHT  = 0.3

# Goal specification
GOAL_TERM    = "t - u*x/c**2"
GOAL_TYPE    = Art & Emp
TARGET_TERM  = "f(-k*y + t*w - k*u*x/c)"   # for backward-gate checks & closeness

# Which primitives count as “backward”
BACKWARD_NAMES = {"BWD"}

# Toggleable constraints
ENABLE_NO_REPEAT             = True # False
ENABLE_SUB1_CONSTRAINT       = True # False
ENABLE_SIM_SIMPUS_CONSTRAINT = True # False
ENABLE_SIMPUS_SIM_CONSTRAINT = True # False
ENABLE_BACKWARD_GATE         = True # False
ENABLE_BACKWARD_ONLY_MODE    = False

# Uniqueness control for experiments
N_DEDUPES = 500   # #600 #1000

# Toggle per-run logging
VERBOSE = False

# ──────────────────────────────────────────────────────────────
# Helpers & caches
# ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def expr_to_str(expr):
    return str(expr)

# single SequenceMatcher instance; we only change its sequences
_seq = SequenceMatcher()

@lru_cache(maxsize=None)
def closeness_cached(term: str) -> float:
    _seq.set_seqs(term, GOAL_TERM)
    return _seq.ratio()

def step_weight(judgment: Judgment) -> float:
    term_score = closeness_cached(expr_to_str(judgment.term))
    type_score = 1.0 if judgment.typ == GOAL_TYPE else 0.0
    return TERM_WEIGHT * term_score + TYPE_WEIGHT * type_score

def build_initial_state() -> State:
    ctx_mem     = ContextWithMemory()
    ctx_lorentz = Context(assumptions={"Lorentz"})
    j0 = Judgment(
        ctx_lorentz,
        name="tprime",
        term=GOAL_TERM,
        typ=Art,
        parent=None
    )
    ctx_mem.start(j0)
    return State(deepcopy(ctx_mem), j0, depth=0, trace=("SEED",))

# Global caches so our wrapper can look up real objects
_ctx_cache    = {}
_cursor_cache = {}

# A cached primitive-application helper (speeds up repeated calls)
@lru_cache(maxsize=None)
def _apply_primitive_cached(ctx_id: int, cursor_id: int, prim_name: str):
    ctx    = _ctx_cache[ctx_id]
    cursor = _cursor_cache[cursor_id]
    fn     = dict(PRIMITIVES)[prim_name]
    return fn(deepcopy(ctx), cursor)

# ──────────────────────────────────────────────────────────────
# Pure-Bayes single-chain sampler
# ──────────────────────────────────────────────────────────────
def bayesian_chain(initial: State, priors: dict, length: int = CHAIN_LENGTH) -> State:
    st = initial
    backwards_mode = False
    last_op        = None
    used_sub1 = used_sim_simplus = used_simplus_sim = used_chg = False

    for _ in range(length):
        curr = expr_to_str(st.cursor.term)
        candidates = []

        for name, fn in PRIMITIVES:
            if ENABLE_NO_REPEAT and name == last_op and not (backwards_mode and name in BACKWARD_NAMES):
                continue
            if ENABLE_SIM_SIMPUS_CONSTRAINT and (last_op, name) == ("SIM", "SIM+") and used_sim_simplus:
                continue
            if ENABLE_SIMPUS_SIM_CONSTRAINT and (last_op, name) == ("SIM+", "SIM") and used_simplus_sim:
                continue
            if ENABLE_SUB1_CONSTRAINT and name == "SUB1" and used_sub1:
                continue
            if name == "CHG" and used_chg:
                continue
            if ENABLE_BACKWARD_ONLY_MODE and backwards_mode and name not in BACKWARD_NAMES:
                continue
            if ENABLE_BACKWARD_GATE and not backwards_mode and name in BACKWARD_NAMES:
                if curr != TARGET_TERM and not used_chg:
                    continue

            try:
                # register real objects by id for the cached wrapper
                _ctx_cache   [id(st.ctx)]    = st.ctx
                _cursor_cache[id(st.cursor)] = st.cursor
                ctx2, j2 = _apply_primitive_cached(id(st.ctx), id(st.cursor), name)
                candidates.append((name, ctx2, j2))
            except Exception:
                pass

        if not candidates:
            break

        # Thompson sampling across candidates
        thetas = []
        for name, _, _ in candidates:
            if name in BACKWARD_NAMES or name == "CHG":
                thetas.append(1.0)
            else:
                a, b = priors[name]["alpha"], priors[name]["beta"]
                p = random.betavariate(a, b)
                thetas.append(p)

        idx = max(range(len(candidates)), key=lambda i: thetas[i])
        name, ctx2, j2 = candidates[idx]

        # update flags
        backwards_mode  = backwards_mode or (name in BACKWARD_NAMES)
        used_sub1       = used_sub1 or (name == "SUB1")
        used_sim_simplus= used_sim_simplus or ((last_op, name) == ("SIM", "SIM+"))
        used_simplus_sim= used_simplus_sim or ((last_op, name) == ("SIM+", "SIM"))
        used_chg        = used_chg or (name == "CHG")

        # apply
        st = State(ctx2, j2, st.depth + 1, st.trace + (name,))
        last_op = name

        # reward & update posterior
        r = step_weight(j2)
        priors[name]["alpha"] += r
        priors[name]["beta"]  += (1.0 - r)

        if st.is_goal():
            break

    return st

# ──────────────────────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────────────────────
def run_one_experiment(max_samples: int = MAX_SAMPLES, n_dedupe: int = N_DEDUPES) -> Optional[int]:
    """
    Run up to `max_samples` pure-Bayes chains, guaranteeing uniqueness only for the first `n_dedupe` outputs.
    Return the 1-based index at which the goal was first reached, or None.
    """
    initial = build_initial_state()
    seen: set = set()
    count = 0

    while count < max_samples:
        # fresh priors for each chain (keeps exploration alive)
        priors = {name: {"alpha": 1.0, "beta": 1.0} for name, _ in PRIMITIVES}
        st = bayesian_chain(initial, priors, length=CHAIN_LENGTH)

        key = (st.trace, expr_to_str(st.cursor.term), expr_to_str(st.cursor.typ))
        # enforce uniqueness only for the first n_dedupe samples
        if count < n_dedupe and key in seen:
            continue
        seen.add(key)

        count += 1
        if st.is_goal():
            return count

    return None

def run_experiments(n_runs: int,
                    max_samples: int = MAX_SAMPLES,
                    n_dedupe: int = N_DEDUPES,
                    parallel: bool = False) -> List[Optional[int]]:
    """Run `run_one_experiment` n_runs times. Prints a line per run and a final summary."""
    results: List[Optional[int]] = []
    logs: List[str] = []

    for run in range(1, n_runs + 1):
        hit = run_one_experiment(max_samples, n_dedupe)
        status = f"✅ reached at chain {hit}" if hit is not None else "❌ not reached"
        line = f"Run {run:3d}: {status}"
        if VERBOSE:
            print(line)
        else:
            logs.append(line)
        results.append(hit)

    if not VERBOSE:
        print("\n".join(logs))

    succ = sum(1 for r in results if r is not None)
    print(f"\n→ {succ}/{n_runs} runs reached the goal.")
    return results

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()

    N_RUNS = 150
    results = run_experiments(
        N_RUNS,
        max_samples=MAX_SAMPLES,
        n_dedupe=N_DEDUPES,
        parallel=False
    )

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
    # ps.print_stats(20)
    # print("\n=== PROFILING RESULTS ===")
    # print(s.getvalue())

    t1 = time.perf_counter()
    print(f"\n✅ Total elapsed time: {t1 - t0:.2f} seconds")
