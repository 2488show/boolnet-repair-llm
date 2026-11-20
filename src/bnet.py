"""Minimal Boolean-network toolkit: read/write .bnet, fixed points, truth tables, monotonicity.

A network is a dict  target -> expression string  in BoolNet syntax (!, &, |, parentheses).
Inputs are variables that appear in some expression but have no rule of their own; they are
held constant, so a fixed point of the network is a steady state for one choice of inputs.
"""
import itertools
import re

import numpy as np

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CONST = {"true": "True", "false": "False", "1": "True", "0": "False"}


def read_bnet(path):
    net = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or line.lower().startswith("targets"):
            continue
        tgt, expr = line.split(",", 1)
        net[tgt.strip()] = expr.strip()
    return net


def write_bnet(net, path):
    with open(path, "w") as f:
        f.write("targets, factors\n")
        for t in sorted(net):
            f.write(f"{t}, {net[t]}\n")


def regulators(expr):
    return sorted({v for v in TOKEN.findall(expr) if v.lower() not in CONST})


def variables(net):
    names = set(net)
    for e in net.values():
        names.update(regulators(e))
    return sorted(names)


def inputs(net):
    return [v for v in variables(net) if v not in net]


def _py(expr):
    """BoolNet expression -> Python expression usable with numpy bool arrays and with bools."""
    expr = expr.replace("&&", "&").replace("||", "|")
    expr = TOKEN.sub(lambda m: CONST.get(m.group(0).lower(), f"V['{m.group(0)}']"), expr)
    return expr.replace("!", "~")


def evaluate(expr, values):
    """values: name -> numpy bool array (or bool). Returns array of the same shape."""
    # numpy bools are required: on Python bools, ~True == -2 instead of False
    values = {k: (np.bool_(v) if isinstance(v, bool) else v) for k, v in values.items()}
    out = eval(_py(expr), {"__builtins__": {}, "True": np.True_, "False": np.False_}, {"V": values})
    return np.asarray(out, dtype=bool) if np.ndim(out) else bool(out)


def fixed_points(net, max_vars=24):
    """All fixed points by exhaustive search (fine up to ~24 variables)."""
    names = variables(net)
    n = len(names)
    if n > max_vars:
        raise ValueError(f"{n} variables is too many for exhaustive search")
    idx = np.arange(2 ** n, dtype=np.uint32)
    V = {v: ((idx >> i) & 1).astype(bool) for i, v in enumerate(names)}
    ok = np.ones(2 ** n, dtype=bool)
    for t, e in net.items():
        f = evaluate(e, V)
        if np.ndim(f) == 0:
            f = np.full(2 ** n, bool(f))
        ok &= (f == V[t])
    states = idx[ok]
    return [{v: int((s >> i) & 1) for i, v in enumerate(names)} for s in states]


def truth_table(expr, regs):
    """Tuple of outputs over all assignments of regs (in the given order)."""
    out = []
    for bits in itertools.product([False, True], repeat=len(regs)):
        out.append(bool(evaluate(expr, dict(zip(regs, bits)))))
    return tuple(out)


def same_function(e1, e2):
    regs = sorted(set(regulators(e1)) | set(regulators(e2)))
    return truth_table(e1, regs) == truth_table(e2, regs)


def table_distance(e1, e2):
    """Fraction of input assignments on which two functions differ."""
    regs = sorted(set(regulators(e1)) | set(regulators(e2)))
    a, b = truth_table(e1, regs), truth_table(e2, regs)
    return sum(x != y for x, y in zip(a, b)) / len(a)


def signs(expr):
    """Effect of each regulator: +1 activator, -1 inhibitor, 0 no effect, None = non-monotone."""
    regs = regulators(expr)
    out = {}
    for r in regs:
        others = [x for x in regs if x != r]
        up = down = False
        for bits in itertools.product([False, True], repeat=len(others)):
            ctx = dict(zip(others, bits))
            lo = bool(evaluate(expr, {**ctx, r: False}))
            hi = bool(evaluate(expr, {**ctx, r: True}))
            up |= (hi and not lo)
            down |= (lo and not hi)
        out[r] = None if (up and down) else (1 if up else (-1 if down else 0))
    return out


def is_monotone(net):
    return all(None not in signs(e).values() for e in net.values())
