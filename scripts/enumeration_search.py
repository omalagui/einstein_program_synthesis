#!/usr/bin/env python3
# enumeration_search.py
# Enumeration search (weighted or greedy), split so notebooks can import & run.

from copy import deepcopy
import random
import difflib
from functools import lru_cache
from typing import Optional, List
import time
import cProfile, pstats, io

from context import Context
from judgments import ContextWithMemory, Judgment
from einstein_types import Art, Emp
from synthesis_state import State
from synthesis_primitives import PRIMITIVES

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MAX_SAMPLES  = 1000
CHAIN_LENGTH = 8
USE_GREEDY   = False   # True = greedy; False = weighted sampling

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
ENABLE_NO_REPEAT             = False  # True #False
ENABLE_SUB1_CONSTRAINT       = False  # True #False
ENABLE_SIM_SIMPUS_CONSTRAINT = False  # True #False
ENABLE_SIMPUS_SIM_CONSTRAINT = False  # True #False
ENABLE_BACKWARD_GATE         = False  # True #False
ENABLE_BACKWARD_ONLY_MODE    = False

# Per-run logging (only in run_experiments)
VERBOSE = True

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def expr_to_str(expr):
    return str(expr)

def closeness(a: str, b: str) -> float:
    """difflib ratio in [0,1]; 1 is identical."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def step_weight(j: Judgment) -> float:
    """Combined score in [0,1] of how close this judgment is to the goal."""
    term_score = closeness(expr_to_str(j.term), GOAL_TERM)
    type_score = 1.0 if j.typ == GOAL_TYPE else 0.0
    return TERM_WEIGHT * term_score + TYPE_WEIGHT * type_score

def build_initial_state() -> State:
    """Seed a new search at the pure Lorentz judgment for relativity of time."""
    ctx_mem     = ContextWithMemory()
    ctx_lorentz = Context(assumptions={"Lorentz"})
    j0 = Judgment(ctx_lorentz, name="tprime", term=GOAL_TERM, typ=Art, parent=None)
    ctx_mem.start(j0)
    return State(deepcopy(ctx_mem), j0, depth=0, trace=("SEED",))

# ──────────────────────────────────────────────────────────────
# Chain builders
# ──────────────────────────────────────────────────────────────
def weighted_chain(initial: State, length: int = CHAIN_LENGTH) -> State:
    """Sample a single chain weighted by combined (term+type) score."""
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
                ctx2, j2 = fn(deepcopy(st.ctx), st.cursor)
                candidates.append((name, ctx2, j2))
            except Exception:
                pass

        if not candidates:
            break

        # compute combined weights
        weights = []
        for name, _, j2 in candidates:
            if name in BACKWARD_NAMES or name == "CHG":
                weights.append(1.0)
            else:
                weights.append(step_weight(j2))

        # choose next primitive
        if sum(weights) == 0:
            idx = random.randrange(len(candidates))
        else:
            idx = random.choices(range(len(candidates)), weights=weights)[0]
        name, ctx2, j2 = candidates[idx]

        # update flags
        backwards_mode  = backwards_mode or (name in BACKWARD_NAMES)
        used_sub1       = used_sub1 or (name == "SUB1")
        used_sim_simplus= used_sim_simplus or ((last_op, name) == ("SIM", "SIM+"))
        used_simplus_sim= used_simplus_sim or ((last_op, name) == ("SIM+", "SIM"))
        used_chg        = used_chg or (name == "CHG")

        # apply it
        st = State(ctx2, j2, st.depth + 1, st.trace + (name,))
        last_op = name

        # early exit
        if st.is_goal():
            break

    return st

def greedy_chain(initial: State, length: int = CHAIN_LENGTH) -> State:
    """Always pick the next step maximizing combined (term+type) score."""
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
                ctx2, j2 = fn(deepcopy(st.ctx), st.cursor)
                candidates.append((name, ctx2, j2))
            except Exception:
                pass

        if not candidates:
            break

        # pick index maximizing score (1.0 for backward/CHG)
        def score_for(i: int) -> float:
            name, _, j2 = candidates[i]
            return 1.0 if (name in BACKWARD_NAMES or name == "CHG") else step_weight(j2)

        best_idx = max(range(len(candidates)), key=score_for)
        name, ctx2, j2 = candidates[best_idx]

        backwards_mode  = backwards_mode or (name in BACKWARD_NAMES)
        used_sub1       = used_sub1 or (name == "SUB1")
        used_sim_simplus= used_sim_simplus or ((last_op, name) == ("SIM", "SIM+"))
        used_simplus_sim= used_simplus_sim or ((last_op, name) == ("SIM+", "SIM"))
        used_chg        = used_chg or (name == "CHG")

        st = State(ctx2, j2, st.depth + 1, st.trace + (name,))
        last_op = name

        if st.is_goal():
            break

    return st

# ──────────────────────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────────────────────
def run_one_experiment(max_samples: int = MAX_SAMPLES) -> Optional[int]:
    """
    Run up to `max_samples` enumeration chains (greedy or weighted).
    Return the 1-based index at which the goal was first reached, or None.
    """
    initial = build_initial_state()
    seen: set = set()
    count = 0
    builder = greedy_chain if USE_GREEDY else weighted_chain

    while count < max_samples:
        st = builder(initial, length=CHAIN_LENGTH)

        # dedupe within this run
        key = (st.trace, expr_to_str(st.cursor.term), expr_to_str(st.cursor.typ))
        if key in seen:
            continue
        seen.add(key)

        count += 1
        if st.is_goal():
            return count

    return None

def run_experiments(n_runs: int,
                    max_samples: int = MAX_SAMPLES,
                    parallel: bool = False) -> List[Optional[int]]:
    """Run `run_one_experiment` n_runs times; print per-run status and summary."""
    results: List[Optional[int]] = []
    logs: List[str] = []

    for i in range(n_runs):
        hit = run_one_experiment(max_samples)
        status = f"✅ reached at chain {hit}" if hit is not None else "❌ not reached"
        line = f"Run {i+1:3d}: {status}"
        if VERBOSE:
            print(line)
        else:
            logs.append(line)
        results.append(hit)

    if not VERBOSE:
        print("\n".join(logs))

    successes = sum(1 for r in results if r is not None)
    print(f"\n→ {successes}/{n_runs} runs reached the goal.")
    return results

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.perf_counter()
    pr = cProfile.Profile(); pr.enable()

    N_RUNS = 150
    results = run_experiments(N_RUNS, max_samples=MAX_SAMPLES, parallel=False)

    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(20)
    print("\n=== PROFILING RESULTS ===")
    print(s.getvalue())

    t1 = time.perf_counter()
    print(f"\n✅ Total elapsed time: {t1 - t0:.2f} seconds")
