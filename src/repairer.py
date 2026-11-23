"""Stage 3b: the language model repairs the model itself, without the solver's list of candidates.

The model sees the corrupted network and the partial steady-state measurements, is told that one rule
is wrong (not which one), and returns corrected rules. Two modes:
  alone    one answer, nothing is checked before it is scored
  checked  each answer is checked against the measurements (exact enumeration of steady states);
           if it contradicts them, the model is told which measured states fail and tries again,
           up to `max_rounds` answers in total
Names are shown real ("named") or as neutral labels ("anonymous"), as in Stage 2.
"""
import json
import random
import re

import benchmark as B
import bnet
import chooser
import llm

PROMPT = """You are an expert in systems biology and Boolean modelling of gene regulatory and signalling networks.

Below is a Boolean network model. Each line gives the rule that decides whether a node is ON (1) or OFF (0):
! means NOT, & means AND, | means OR. Nodes marked as external inputs have no rule.

{model}

Measurements: the cell was observed in {n} different steady states. In each, only some nodes were measured
(1 = ON, 0 = OFF); unlisted nodes were not measured.

{obs}

The model contradicts these measurements: for at least one measured state there is no steady state of the
model that matches it. Exactly one rule of the model is wrong. The error may concern which regulators the
rule uses, whether a regulator activates or inhibits, or how the regulators are combined.

Correct the model so that it agrees with the measurements and is as plausible as possible biologically,
given what is known about these molecules. Change as little as needed. Use only the node names above.
Answer with JSON only: {{"changes": [{{"node": "<node>", "rule": "<new rule>"}}], "reason": "<one sentence>"}}"""

FEEDBACK = """Your correction was applied:
{applied}
It still contradicts the measurements: no steady state of the corrected model matches measured state(s)
{failed}. Give a different correction, in the same JSON format, starting again from the original model above."""


def render(corrupted, obs, m):
    model = "\n".join(f"{m[t]} = {chooser.rename(corrupted[t], m)}" for t in sorted(corrupted, key=lambda x: m[x]))
    inputs = [v for v in bnet.variables(corrupted) if v not in corrupted]
    if inputs:
        model += "\n" + "\n".join(f"{m[v]} = (external input)" for v in sorted(inputs, key=lambda x: m[x]))
    rows = [f"State {i + 1}: " + ", ".join(f"{m[v]}={o[v]}" for v in sorted(o, key=lambda x: m[x]))
            for i, o in enumerate(obs)]
    return model, "\n".join(rows)


def parse(text, corrupted, inv):
    """-> (dict node -> rule in original names, error message or None)."""
    try:
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        changes = d["changes"]
        assert isinstance(changes, list) and changes
    except Exception:
        return None, "unparseable"
    out = {}
    for c in changes:
        node, rule = inv.get(str(c.get("node", "")).strip()), str(c.get("rule", ""))
        if node is None:
            return None, "unknown node"
        toks = set(bnet.TOKEN.findall(rule)) - {"true", "false", "True", "False", "TRUE", "FALSE"}
        if not toks <= set(inv):
            return None, "unknown name in rule"
        rule = bnet.TOKEN.sub(lambda t: inv.get(t.group(0), t.group(0)), rule)
        try:
            bnet.evaluate(rule, {v: False for v in bnet.regulators(rule)})
        except Exception:
            return None, "invalid expression"
        out[node] = rule
    return out, None


def failed_states(net, obs):
    # a measured variable that the corrected model no longer mentions (an input whose only use was removed)
    # is unconstrained: give it a self-loop so that it can take either value in a fixed point
    net = dict(net)
    for v in {v for o in obs for v in o} - set(bnet.variables(net)):
        net[v] = v
    fps = bnet.fixed_points(net)
    return [i + 1 for i, o in enumerate(obs) if not any(all(fp[v] == val for v, val in o.items()) for fp in fps)]


def repair(spec, task, truth, condition, mode, max_rounds=3, tag=""):
    """Returns a dict with the outcome of the final answer and of every round."""
    corrupted, obs = task["corrupted"], task["observations"]
    names = bnet.variables(corrupted)
    m = chooser.relabel_map(names, condition, random.Random(f"{task['task_id']}-{condition}"))
    inv = {v: k for k, v in m.items()}
    model, obs_txt = render(corrupted, obs, m)
    prompt = PROMPT.format(model=model, n=len(obs), obs=obs_txt)
    rounds = []
    for r in range(1 if mode == "alone" else max_rounds):
        # DeepSeek's reasoning counts against max_tokens; at 8,192 it truncated 28 of 64 pilot answers to empty text
        reply = llm.ask(spec, prompt, temperature=0.0, max_tokens=8192 if spec.startswith("gemini") else 32768, json_mode=not spec.startswith("local"),
                        tag=f"{tag}|r{r + 1}")
        changes, err = parse(reply, corrupted, inv)
        rec = dict(round=r + 1, error=err, reply=reply[:400])
        if changes is not None:
            net = {**corrupted, **changes}
            fail = failed_states(net, obs)
            rec.update(changes=changes, consistent=not fail, failed=fail,
                       truth=B.is_truth(net, truth), n_changed=len(changes),
                       truth_node_changed=any(t in changes for t, _, _ in task["corruption"]))
        rounds.append(rec)
        if mode == "alone" or rec.get("consistent"):
            break
        applied = "\n".join(f"{m[n]} = {chooser.rename(rl, m)}" for n, rl in (changes or {}).items()) or "(no valid change)"
        why = ", ".join(map(str, rec.get("failed", []))) or "(the answer could not be read: " + str(err) + ")"
        prompt = prompt + "\n\nYour previous answer:\n" + reply.strip()[:2000] + "\n\n" + FEEDBACK.format(applied=applied, failed=why)
    last = rounds[-1]
    return dict(rounds=len(rounds), parsed=last.get("error") is None, consistent=bool(last.get("consistent")),
                correct=bool(last.get("truth")), truth_node_changed=bool(last.get("truth_node_changed")),
                history=rounds)
