import math
import random
import difflib
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from context import Context
from judgments import ContextWithMemory, Judgment
from einstein_types import Art, Emp
from synthesis_state import State
from synthesis_primitives import PRIMITIVES
import time
from functools import lru_cache
import cProfile, pstats, io

import os


# ──────────────────────────────────────────────────────────────
# Search settings
# ──────────────────────────────────────────────────────────────
MAX_SAMPLES_BAYES = 100 # for pure-Bayes data collection
BATCH_SIZE        = 64
EPOCHS            = 3
LR                = 1e-3

MAX_SAMPLES       = 100 # max hybrid chains per experiment
CHAIN_LENGTH      =   8
FEATURE_DIM       = 128

TERM_WEIGHT       = 0.7
TYPE_WEIGHT       = 0.3
GOAL_TERM         = "t - u*x/c**2"
GOAL_TYPE         = Art & Emp
TARGET_TERM       = "f(-k*y + t*w - k*u*x/c)"

MIX_BASE          = 0.1    # annealing start weight on neural

BACKWARD_NAMES            = {"BWD"}
ENABLE_NO_REPEAT          = False # True # False 
ENABLE_SUB1_CONSTRAINT    = False # True # False 
ENABLE_SIM_SIMPUS_CONSTRAINT  = False # True # False 
ENABLE_SIMPUS_SIM_CONSTRAINT  = False # True # False 
ENABLE_BACKWARD_GATE      = False # True # False 
ENABLE_BACKWARD_ONLY_MODE = False

# ─── Toggle per-run logging on/off ────────────────────
VERBOSE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@lru_cache(maxsize=None)
def expr_to_str(expr):
    return str(expr)

# ─── Global caches so our wrapper can look up real objects ───
_ctx_cache    = {}
_cursor_cache = {}

# ─── A cached primitive‐application helper ──────────────────────
@lru_cache(maxsize=None)
def _apply_primitive_cached(ctx_id: int, cursor_id: int, prim_name: str):
    # recover the real objects
    ctx    = _ctx_cache[ctx_id]
    cursor = _cursor_cache[cursor_id]
    # lookup and call the actual primitive
    fn     = dict(PRIMITIVES)[prim_name]
    return fn(deepcopy(ctx), cursor)

# ──────────────────────────────────────────────────────────────
# Helpers 
# ──────────────────────────────────────────────────────────────
def build_initial_state() -> State:
    ctx_mem     = ContextWithMemory()
    ctx_lorentz = Context(assumptions={"Lorentz"})
    j0 = Judgment(ctx_lorentz, name="tprime", term=GOAL_TERM, typ=Art, parent=None)
    ctx_mem.start(j0)
    return State(deepcopy(ctx_mem), j0, depth=0, trace=("SEED",))

@lru_cache(maxsize=None)
def closeness(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def step_weight(j: Judgment) -> float:
    ts = closeness(expr_to_str(j.term), GOAL_TERM)
    ys = 1.0 if j.typ == GOAL_TYPE else 0.0
    return TERM_WEIGHT * ts + TYPE_WEIGHT * ys

def encode_state(state: State) -> torch.Tensor:
    ts = closeness(expr_to_str(state.cursor.term), GOAL_TERM)
    ys = 1.0 if state.cursor.typ == GOAL_TYPE else 0.0
    tn = len(state.trace) / (CHAIN_LENGTH + 1)
    vec = [ts, ys, tn] + [0.0] * (FEATURE_DIM - 3)
    return torch.tensor(vec, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Pure‐Bayes data & policy training 
# ──────────────────────────────────────────────────────────────
def bayesian_chain(initial: State, priors: dict):
    st = initial
    backwards = False
    last_op   = None
    used_sub1 = used_sim_simplus = used_simplus_sim = used_chg = False

    for _ in range(CHAIN_LENGTH):
        curr = expr_to_str(st.cursor.term)
        cands = []
        for name, fn in PRIMITIVES:
            if ENABLE_NO_REPEAT and name==last_op and not (backwards and name in BACKWARD_NAMES):
                continue
            if ENABLE_SUB1_CONSTRAINT and name=="SUB1" and used_sub1:
                continue
            if ENABLE_SIM_SIMPUS_CONSTRAINT and (last_op,name)==("SIM","SIM+") and used_sim_simplus:
                continue
            if ENABLE_SIMPUS_SIM_CONSTRAINT and (last_op,name)==("SIM+","SIM") and used_simplus_sim:
                continue
            if ENABLE_BACKWARD_ONLY_MODE and backwards and name not in BACKWARD_NAMES:
                continue
            if ENABLE_BACKWARD_GATE and not backwards and name in BACKWARD_NAMES:
                if curr!=TARGET_TERM and not used_chg:
                    continue
            try:
                # cctx, cj = fn(deepcopy(st.ctx), st.cursor)

                # first register the current objects by their id:
                _ctx_cache   [id(st.ctx)]    = st.ctx
                _cursor_cache[id(st.cursor)] = st.cursor
                # then call the cached helper instead of fn directly:
                cctx, cj = _apply_primitive_cached(
                    id(st.ctx),
                    id(st.cursor),
                    name
                )

                cands.append((name,cctx,cj))
            except:
                pass

        if not cands:
            break

        # pick by Beta-sampling
        thetas = []
        for name,_,_ in cands:
            if name in BACKWARD_NAMES or name=="CHG":
                thetas.append(1.0)
            else:
                a,b = priors[name]["alpha"], priors[name]["beta"]
                thetas.append(random.betavariate(a,b))
        idx = max(range(len(cands)), key=lambda i: thetas[i])
        name, cctx, cj = cands[idx]

        # update flags
        backwards      |= (name in BACKWARD_NAMES)
        used_sub1      |= (name=="SUB1")
        used_sim_simplus |= (last_op,name)==("SIM","SIM+")
        used_simplus_sim |= (last_op,name)==("SIM+","SIM")
        used_chg       |= (name=="CHG")
        last_op        = name

        st = State(cctx, cj, st.depth+1, st.trace+(name,))
        r  = step_weight(cj)
        priors[name]["alpha"] += r
        priors[name]["beta"]  += (1.0-r)
        if st.is_goal():
            break

    return st, last_op

def collect_bayes_data(n_traces: int = MAX_SAMPLES_BAYES):
    S, A = [], []
    for _ in range(n_traces):
        priors = { name:{"alpha":1.0,"beta":1.0} for name,_ in PRIMITIVES }
        st = build_initial_state()
        for _ in range(CHAIN_LENGTH):
            S.append(encode_state(st).numpy())
            st, action = bayesian_chain(st, priors)
            onehot = np.zeros(len(PRIMITIVES), dtype=np.float32)
            idx    = next(i for i,(n,_) in enumerate(PRIMITIVES) if n == action)
            onehot[idx] = 1.0
            A.append(onehot)
            if st.is_goal():
                break

    return torch.tensor(S), torch.tensor(A)

def train_policy():
    S, A = collect_bayes_data()
    dataset = TensorDataset(S, A)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    net = PolicyNet().to(device)
    opt = optim.Adam(net.parameters(), lr=LR)

    net.train()
    for ep in range(EPOCHS):
        tot = 0
        for s, a in loader:
            s, a = s.to(device), a.to(device)
            opt.zero_grad()
            pred = net(s)
            loss = F.kl_div(pred.log(), a, reduction='batchmean')
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep:2d}  policy_loss={tot/len(loader):.4f}")
    return net


# ──────────────────────────────────────────────────────────────
# Neural policy net & hybrid sampler 
# ──────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc      = nn.Linear(FEATURE_DIM, len(PRIMITIVES))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        return F.softmax(self.fc(self.dropout(x)), dim=-1)

def hybrid_chain(initial: State, net: PolicyNet):
    st = initial
    priors   = { name:{"alpha":1.0,"beta":1.0} for name,_ in PRIMITIVES }
    used_prims= set()
    flags     = {
        'backwards': False,
        'used_sub1': False,
        'used_sim_simplus': False,
        'used_simplus_sim': False,
        'used_chg': False
    }

    net.eval()
    with torch.no_grad():
        for depth in range(CHAIN_LENGTH):
            mix   = MIX_BASE + (1.0 - MIX_BASE)*(depth/(CHAIN_LENGTH-1))
            feat  = encode_state(st).unsqueeze(0).to(device)
            p_net = net(feat).cpu().numpy().squeeze()

            thetas, cands = [], []
            for i,(name,fn) in enumerate(PRIMITIVES):
                if name in used_prims and name not in BACKWARD_NAMES: continue
                if ENABLE_NO_REPEAT and st.trace[-1]==name and not flags['backwards']:
                    continue
                if ENABLE_SUB1_CONSTRAINT and name=="SUB1" and flags['used_sub1']:
                    continue
                if ENABLE_SIM_SIMPUS_CONSTRAINT and (st.trace[-1],name)==("SIM","SIM+") and flags['used_sim_simplus']:
                    continue
                if ENABLE_SIMPUS_SIM_CONSTRAINT and (st.trace[-1],name)==("SIM+","SIM") and flags['used_simplus_sim']:
                    continue
                if ENABLE_BACKWARD_ONLY_MODE and flags['backwards'] and name not in BACKWARD_NAMES:
                    continue
                if ENABLE_BACKWARD_GATE and not flags['backwards'] and name in BACKWARD_NAMES:
                    if expr_to_str(st.cursor.term)!=TARGET_TERM and not flags['used_chg']:
                        continue

                a,b = priors[name]["alpha"], priors[name]["beta"]
                p_b = random.betavariate(a,b)
                th  = mix*p_net[i] + (1.0-mix)*p_b
                thetas.append(th)
                cands.append((name,fn,i))

            if not cands:
                break

            thetas = np.array(thetas)
            thetas /= thetas.sum()
            choice = np.random.choice(len(cands), p=thetas)
            name, fn, _ = cands[choice]

            try:
                 # cctx, cj = fn(deepcopy(st.ctx), st.cursor)

                # first register the current objects by their id:
                _ctx_cache   [id(st.ctx)]    = st.ctx
                _cursor_cache[id(st.cursor)] = st.cursor
                # then call the cached helper instead of fn directly:
                cctx, cj = _apply_primitive_cached(
                    id(st.ctx),
                    id(st.cursor),
                    name
                )

            except:
                continue

            used_prims.add(name)
            if name in BACKWARD_NAMES:      flags['backwards']     = True
            if name=="SUB1":                flags['used_sub1']     = True
            if (st.trace[-1],name)==("SIM","SIM+"):
                                            flags['used_sim_simplus']=True
            if (st.trace[-1],name)==("SIM+","SIM"):
                                            flags['used_simplus_sim']=True
            if name=="CHG":                 flags['used_chg']       = True

            st = State(cctx, cj, st.depth+1, st.trace+(name,))
            r  = step_weight(cj)
            priors[name]["alpha"] += r
            priors[name]["beta"]  += (1.0-r)
            if st.is_goal():
                break

    return st


# ──────────────────────────────────────────────────────────────
# Run one experiment & run experiments
# ──────────────────────────────────────────────────────────────

PRINT_TRACE = os.getenv("BN_TRACE", "0") == "1"

def _format_program_line(count, st):
    ops = " + ".join(st.trace[1:]) if len(st.trace) > 1 else "SEED"
    return f"Program {count}: {ops}  →  {expr_to_str(st.cursor.term)} : {st.cursor.typ}"

def run_one_experiment_paths(net: PolicyNet,
                       max_samples: int = MAX_SAMPLES,
                       print_trace: bool | None = None):
    """
    Run up to `max_samples` hybrid chains.
    Return the 1-based index at which goal was reached, or None.
    """
    init = build_initial_state()
    seen = set()
    count = 0

    while count < max_samples:
        st = hybrid_chain(init, net)
        key = (st.trace, expr_to_str(st.cursor.term), expr_to_str(st.cursor.typ))
        if key in seen:
            continue
        seen.add(key)
        count += 1
        
        if print_trace or PRINT_TRACE:
            print(_format_program_line(count, st), flush=True)

        
        if st.is_goal():
            if print_trace:
                print(f"✅ Hybrid hit goal at {count}")
            return count

    return None

def run_one_experiment(net: PolicyNet,
                       max_samples: int = MAX_SAMPLES):
    """
    Run up to `max_samples` hybrid chains.
    Return the 1-based index at which goal was reached, or None.
    """
    init = build_initial_state()
    seen = set()
    count = 0

    while count < max_samples:
        st = hybrid_chain(init, net)
        key = (st.trace, expr_to_str(st.cursor.term), expr_to_str(st.cursor.typ))
        if key in seen:
            continue
        seen.add(key)
        count += 1
        if st.is_goal():
            return count

    return None

def run_experiments(net: PolicyNet,
                    n_runs: int,
                    max_samples: int = MAX_SAMPLES,
                    parallel: bool = False):
    """
    Run `run_one_experiment` n_runs times, serially or in parallel.
    Prints and returns a list of results (indices or None).
    """

    results = []
    logs    = []
    for i in range(n_runs):
        res    = run_one_experiment(net, max_samples)
        status = f"✅ reached at program {res}" if res else "❌ not reached"
        line   = f"Run {i+1:2d}: {status}"

        if VERBOSE:
            print(line)
        else:
            logs.append(line)

        results.append(res)

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

    # 1) Train your policy net once
    policy_net = train_policy()

    # 2) Run N independent hybrid simulations
    N_RUNS = 150

    # pr = cProfile.Profile()
    # pr.enable()

    results = run_experiments(policy_net, N_RUNS, max_samples=MAX_SAMPLES, parallel=False) 

    # pr.disable()

    # 3) Dump top‐20 cumulative hotspots
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
    # ps.print_stats(20)
    # print("\n=== PROFILING RESULTS ===")
    # print(s.getvalue())

    t1 = time.perf_counter()
    print(f"\n✅ Total elapsed time: {t1 - t0:.2f} seconds")
