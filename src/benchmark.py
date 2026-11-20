"""Build repair tasks with a known answer, solve them with pyModRev, and score the candidates.

Task = (true network, a corrupted copy, a set of partial steady-state observations of the truth).
pyModRev returns every minimal repair of the corrupted model that fits the observations; each one
is a candidate model. Scoring compares candidates with the truth, rule by rule.
"""
import pathlib
import random
import subprocess
import tempfile

import bnet

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


def corrupt(net, rng, n_errors=1):
    """Return (corrupted net, list of (node, kind, detail))."""
    net = dict(net)
    log = []
    targets = [t for t, e in net.items() if bnet.regulators(e)]
    for t in rng.sample(targets, k=min(n_errors, len(targets))):
        regs = bnet.regulators(net[t])
        kinds = ["flip"] + (["and_or"] if len(regs) >= 2 else [])
        kind = rng.choice(kinds)
        if kind == "flip":
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
        return [bnet.read_bnet(f) for f in files], "ok"


# ---------- scoring ----------

def rules_correct(cand, truth):
    """Which truth rules the candidate reproduces exactly (inputs ignored)."""
    return {t: (t in cand and bnet.same_function(cand[t], truth[t])) for t in truth}


def is_truth(cand, truth):
    return all(rules_correct(cand, truth).values())
