"""Ask a language model to pick one repair among the solver's equally minimal candidates.

Name conditions (the network and candidates are identical; only the labels change):
  named     real gene/protein names (the "v_" prefix of BBM is dropped)
  anonymous neutral labels g1, g2, ... (no biological knowledge usable)
  shuffled  real names, but permuted across nodes (biological knowledge points the wrong way)
"""
import json
import random
import re

import bnet
import llm

PROMPT = """You are an expert in systems biology and Boolean modelling of gene regulatory and signalling networks.

Below is a Boolean network model. Each line gives the rule that decides whether a node is ON (1) or OFF (0):
! means NOT, & means AND, | means OR. Nodes without a rule are external inputs.

{model}

The rules for {targets} were found to contradict experimental steady-state measurements.
A logic solver found {k} alternative corrections. Every one of them is equally consistent with all
measurements and equally small, so the data cannot decide between them:

{options}

Which correction is most plausible biologically, given what is known about these molecules and how
they regulate each other? Answer with JSON only: {{"choice": <option number>, "reason": "<one sentence>"}}"""


def relabel_map(names, condition, rng):
    clean = {v: re.sub(r"^v_", "", v) for v in names}
    if condition == "named":
        return clean
    if condition == "anonymous":
        order = names[:]
        rng.shuffle(order)
        return {v: f"g{i + 1}" for i, v in enumerate(order)}
    if condition == "shuffled":
        labels = [clean[v] for v in names]
        while True:  # derangement: no node keeps its own name
            perm = labels[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(labels, perm)):
                return dict(zip(names, perm))
    raise ValueError(condition)


def rename(expr, m):
    return bnet.TOKEN.sub(lambda t: m.get(t.group(0), t.group(0)), expr)


def build_prompt(corrupted, candidates, condition, rng):
    names = bnet.variables(corrupted)
    m = relabel_map(names, condition, rng)
    # nodes whose rule differs between candidates or from the corrupted model
    # nodes on which the candidates disagree are the question; a change shared by all candidates
    # is already decided by the data and is shown as part of the model
    changed = sorted({t for t in corrupted
                      if any(not bnet.same_function(c[t], candidates[0][t]) for c in candidates[1:])})
    base = {t: candidates[0][t] for t in corrupted}
    shown = [t for t in sorted(corrupted, key=lambda x: m[x]) if t not in changed]
    model = "\n".join(f"{m[t]} = {rename(base[t], m)}" if base[t] != t else f"{m[t]} = (no rule: external input)"
                      for t in shown)
    model += "\n" + "\n".join(f"{m[t]} = ???   (needs correction)" for t in changed)
    order = list(range(len(candidates)))
    rng.shuffle(order)
    opts = []
    for k, ci in enumerate(order, 1):
        rules = "; ".join(f"{m[t]} = " + (rename(candidates[ci][t], m) if candidates[ci][t] != t else "(no rule: external input)") for t in changed)
        opts.append(f"Option {k}: {rules}")
    prompt = PROMPT.format(model=model, targets=", ".join(m[t] for t in changed), k=len(candidates),
                           options="\n".join(opts))
    return prompt, order


def parse_choice(text, k):
    try:
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        c = int(d["choice"])
        return c if 1 <= c <= k else None
    except Exception:
        return None


def choose(spec, corrupted, candidates, condition, seed, tag=""):
    """Returns (index into candidates or None if unparseable, raw reply)."""
    rng = random.Random(seed)
    prompt, order = build_prompt(corrupted, candidates, condition, rng)
    reply = llm.ask(spec, prompt, temperature=0.0, max_tokens=4096,
                    json_mode=spec.startswith("gemini"), tag=tag)
    c = parse_choice(reply, len(candidates))
    return (order[c - 1] if c else None), reply
