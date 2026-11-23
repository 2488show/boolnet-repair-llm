"""Build repair tasks with a known answer, solve them with pyModRev, and score the candidates.

Task = (true network, a corrupted copy, a set of partial steady-state observations of the truth).
pyModRev returns every minimal repair of the corrupted model that fits the observations; each one
is a candidate model. Scoring compares candidates with the truth, rule by rule.
"""
import itertools
import pathlib
import random
import subprocess
import tempfile

import bnet

MAX_MODELS = 200  # more raw repairs than this: recorded as too many, not enumerated
PYMODREV = str(pathlib.Path(__file__).resolve().parents[1] / ".venv" / "bin" / "pymodrev")


# ---------- corruptions (all stay inside monotone functions, so the truth remains reachable) ----

def flip_sign(expr, reg):
    return bnet.TOKEN.sub(lambda m: f"(!{reg})" if m.group(0) == reg else m.group(0), expr)


def and_or_swap(expr):
    """Replace the rule by the plain AND (or OR) of its signed regulators, whichever differs."""
    s = bnet.signs(expr)
    lits = [r if v > 0 else f"!{r}" for r, v in s.items() if v != 0]
    for joiner in (" & ", " | "):
        cand = "(" + joiner.join(lits) + ")"
        if not bnet.same_function(cand, expr):
            return cand
    return None


def drop_regulator(expr, reg, value):
    """Fix `reg` to a constant, so the rule no longer depends on it, and write the result without
    constants as a monotone sum of products over the remaining regulators (stays monotone)."""
    regs = [r for r in bnet.regulators(expr) if r != reg]
    fixed = bnet.TOKEN.sub(lambda m: ("true" if value else "false") if m.group(0) == reg else m.group(0), expr)
    sg = bnet.signs(fixed)
    sg = {r: sg[r] for r in regs if r in sg}
    pts = [bits for bits in itertools.product([False, True], repeat=len(regs))
           if bnet.evaluate(fixed, dict(zip(regs, bits)))]
    if not pts:
        return None  # constant false: not a usable error
    # orient each variable by its sign (inhibitors are "on" when low), keep minimal true points
    def on(bits):
        return frozenset(r for r, b in zip(regs, bits) if b == (sg.get(r, 1) >= 0))
    sets = {on(b) for b in pts}
    mins = [a for a in sets if not any(b < a for b in sets)]
    terms = []
    for a in mins:
        lits = [(r if sg.get(r, 1) >= 0 else f"!{r}") for r in regs if r in a]
        terms.append("(" + " & ".join(lits) + ")" if lits else None)
    if None in terms:
        return None  # constant true
    return "(" + " | ".join(terms) + ")"


def add_regulator(expr, new, positive, use_and):
    """Combine the rule with one extra signed regulator by AND or OR (stays monotone)."""
    lit = new if positive else f"!{new}"
    return f"(({expr}) {'&' if use_and else '|'} {lit})"


def corrupt(net, rng, n_errors=1, kinds=("flip", "and_or")):
    """Return (corrupted net, list of (node, kind, detail)).

    kinds: which kinds of error may be drawn. The default ("flip", "and_or") keeps every earlier
    benchmark byte-identical; "del_reg" removes a regulator from a rule, "add_reg" adds a spurious one.
    """
    net = dict(net)
    log = []
    all_vars = bnet.variables(net)
    targets = [t for t, e in net.items() if bnet.regulators(e)]
    for t in rng.sample(targets, k=min(n_errors, len(targets))):
        regs = bnet.regulators(net[t])
        avail = ["flip"] * ("flip" in kinds) + (["and_or"] if len(regs) >= 2 and "and_or" in kinds else [])
        if "del_reg" in kinds and len(regs) >= 2:
            avail.append("del_reg")
        extra = [v for v in all_vars if v != t and v not in regs]
        if "add_reg" in kinds and extra:
            avail.append("add_reg")
        kind = rng.choice(avail)
        if kind == "del_reg":
            r = rng.choice(regs)
            v = rng.choice([0, 1])
            new = drop_regulator(net[t], r, v)
            if new is None or bnet.same_function(new, net[t]):
                new = drop_regulator(net[t], r, 1 - v)
            if new is None or bnet.same_function(new, net[t]):
                continue  # no usable deletion for this rule
            net[t] = new
            log.append((t, "del_reg", r))
        elif kind == "add_reg":
            x = rng.choice(extra)
            net[t] = add_regulator(net[t], x, rng.random() < 0.5, rng.random() < 0.5)
            log.append((t, "add_reg", x))
        elif kind == "flip":
            r = rng.choice(regs)
            net[t] = flip_sign(net[t], r)
            log.append((t, "flip", r))
        else:
            new = and_or_swap(net[t])
            if new is None:
                r = rng.choice(regs)
                net[t] = flip_sign(net[t], r)
                log.append((t, "flip", r))
            else:
                net[t] = new
                log.append((t, "and_or", ""))
    return net, log


# ---------- observations ----------

def observe(fps, names, rng, n_states, coverage):
    """Pick n_states fixed points and, in each, reveal a random `coverage` fraction of variables.

    Returns (observed list of dicts with only revealed variables, indices of the chosen states).
    """
    chosen = rng.sample(range(len(fps)), k=min(n_states, len(fps)))
    k = max(1, round(coverage * len(names)))
    obs = []
    for i in chosen:
        shown = set(rng.sample(names, k))
        obs.append({v: fps[i][v] for v in names if v in shown})
    return obs, chosen


def consistent(net, obs):
    """True if every observation can be completed to a fixed point of net (exact check)."""
    fps = bnet.fixed_points(net)
    return all(any(all(fp[v] == val for v, val in o.items()) for fp in fps) for o in obs)


# ---------- solving ----------

def solve(net, obs, names, timeout=120):
    """All minimal repairs as candidate networks. Returns (list of nets, status)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        bnet.write_bnet(net, tmp / "model.bnet")
        with open(tmp / "obs.csv", "w") as f:
            f.write("," + ",".join(names) + "\n")
            for i, o in enumerate(obs):
                f.write(f"p{i}," + ",".join(str(o[v]) if v in o else "*" for v in names) + "\n")
        try:
            r = subprocess.run([PYMODREV, "-m", "model.bnet", "-obs", "obs.csv", "steady", "-t", "m",
                                "-s", "3"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [], "timeout"
        if "is consistent" in r.stdout:
            return [], "consistent"
        files = sorted(tmp.glob("model_*.bnet"))
        if not files:
            return [], "no_repair:" + (r.stdout + r.stderr)[-200:].replace("\n", " ")
        if len(files) > MAX_MODELS:
            return [], f"too_many:{len(files)}"
        cands = []
        for f in files:
            c = bnet.read_bnet(f)
            # a node without a rule was turned into a free input by an edge removal: keep its value
            cands.append({t: c.get(t, t) for t in net})
        return dedupe(cands), "ok"


def canonical(expr):
    """Key that is equal for two expressions exactly when they are the same Boolean function."""
    sg = bnet.signs(expr)
    regs = [r for r, s in sg.items() if s != 0]  # variables with no effect are dropped ...
    idle = {r: False for r, s in sg.items() if s == 0}  # ... and held at any fixed value
    rows = []
    for bits in __import__("itertools").product([False, True], repeat=len(regs)):
        rows.append(bool(bnet.evaluate(expr, {**idle, **dict(zip(regs, bits))})))
    return tuple(regs), tuple(rows)


def dedupe(cands):
    """pyModRev can return the same Boolean function written in different ways; keep one of each."""
    seen, out = set(), []
    for c in cands:
        key = tuple((t, canonical(c[t])) for t in sorted(c))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# ---------- scoring ----------

def rules_correct(cand, truth):
    """Which truth rules the candidate reproduces exactly (inputs ignored)."""
    return {t: (t in cand and bnet.same_function(cand[t], truth[t])) for t in truth}


def is_truth(cand, truth):
    return all(rules_correct(cand, truth).values())
