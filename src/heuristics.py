"""Choosers that use no biological knowledge. If a language model only beats random because it
prefers, say, simpler rules, these baselines will match it, and the anonymous condition will too."""
import random

import bnet


def _differing(task):
    cands, bad = task["candidates"], task["corrupted"]
    return [t for t in bad if any(not bnet.same_function(c[t], cands[0][t]) for c in cands[1:])]


def _literals(expr, node):
    return 0 if expr == node else len(bnet.TOKEN.findall(expr))


def _pick_min(task, score, seed):
    scores = [score(c) for c in task["candidates"]]
    best = min(scores)
    ties = [i for i, s in enumerate(scores) if s == best]
    return random.Random(seed).choice(ties)


def simplest(task, seed):
    """Fewest literals in the rules the candidates disagree on (an input counts as zero)."""
    diff = _differing(task)
    return _pick_min(task, lambda c: sum(_literals(c[t], t) for t in diff), seed)


def closest(task, seed):
    """Smallest change of behaviour relative to the corrupted (starting) model."""
    diff, bad = _differing(task), task["corrupted"]
    return _pick_min(task, lambda c: sum(bnet.table_distance(c[t], bad[t]) for t in diff), seed)


def richest(task, seed):
    """Most literals in the disputed rules: the opposite of `simplest`."""
    diff = _differing(task)
    return _pick_min(task, lambda c: -sum(_literals(c[t], t) for t in diff), seed)


def keep_regulation(task, seed):
    """Avoid turning a node into a free input, then keep as many of the starting model's regulators
    as possible. Encodes the prior "published models rarely drop a regulation"."""
    diff, bad = _differing(task), task["corrupted"]

    def score(c):
        dropped = sum(c[t] == t for t in diff)
        lost = sum(len(set(bnet.regulators(bad[t])) - set(bnet.regulators(c[t]))) for t in diff)
        return (dropped, lost)
    return _pick_min(task, score, seed)
