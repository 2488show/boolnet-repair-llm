"""Stage 3b analysis: three language models repair the model themselves (no candidate list).

Only tasks finished by every model in SPECS, in both name conditions, are used, so every comparison is
paired. Accuracy = the final answer reproduces the true network exactly (every rule the same Boolean
function). "alone" = first answer; "checked" = after up to 3 rounds of consistency feedback.
95% CIs: percentile bootstrap over tasks (2,000 resamples).
Writes results/stage3b_table.csv, results/stage3b_paired.csv and results/stage3b_cost.csv.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECS = {"gemini:gemini-3.8-flash": "Gemini 3.8 Flash (API)",
         "deepseek:deepseek-flash": "DeepSeek V4.1 Flash (API)",
         "codex:gpt-6-luna": "GPT-6-Luna (Codex CLI)"}
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    b = rng.choice(x, size=(2000, len(x))).mean(1)
    return f"{np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}"


d = pd.read_json(ROOT / "runs/stage3b_results.jsonl", lines=True)
d = d[d.spec.isin(SPECS)]
d["group"] = np.where(d.set == "orig", "original (sign / AND-OR)",
                      np.where(d.kind == "add_reg", "added regulator", "removed regulator"))
n_done = d.groupby("task_id").apply(lambda x: len(set(zip(x.spec, x.condition))))
common = set(n_done[n_done == 2 * len(SPECS)].index)
d = d[d.task_id.isin(common)]
print(f"{len(common)} tasks finished by all {len(SPECS)} models in both name conditions")

rows = []
for (g, spec, cond), x in d.groupby(["group", "spec", "condition"]):
    rows.append(dict(group=g, model=SPECS[spec], condition=cond, tasks=len(x),
                     truth_among_solver_repairs=round(x.truth_in_candidates.mean(), 3),
                     alone=round(x.alone_correct.mean(), 3), alone_ci=ci(x.alone_correct),
                     checked=round(x.checked_correct.mean(), 3), checked_ci=ci(x.checked_correct),
                     checked_consistent=round(x.checked_consistent.mean(), 3),
                     mean_rounds=round(x.checked_rounds.mean(), 2)))
tab = pd.DataFrame(rows)
tab.to_csv(ROOT / "results/stage3b_table.csv", index=False)
pd.set_option("display.width", 250)
print(tab.to_string(index=False))

# paired contrasts on "checked" accuracy
w = d.pivot_table(index=["task_id", "group"], columns=["spec", "condition"], values="checked_correct",
                  aggfunc="first").astype(float)
truth_in = d.groupby("task_id").truth_in_candidates.first().astype(float)
rows = []
for g in ["added regulator", "removed regulator", "original (sign / AND-OR)", "all"]:
    x = w if g == "all" else w.xs(g, level="group", drop_level=False)
    t = truth_in.loc[x.index.get_level_values("task_id")].values
    for spec, name in SPECS.items():
        nd = x[(spec, "named")] - x[(spec, "anonymous")]
        rows.append(dict(group=g, contrast=f"{name}: named - anonymous", tasks=len(x), diff=round(nd.mean(), 3), ci=ci(nd)))
        sv = x[(spec, "named")].values - t
        rows.append(dict(group=g, contrast=f"{name} named - truth among solver repairs", tasks=len(x),
                         diff=round(sv.mean(), 3), ci=ci(sv)))
    gem = x[("gemini:gemini-3.8-flash", "named")]
    for spec in list(SPECS)[1:]:
        dd = gem - x[(spec, "named")]
        rows.append(dict(group=g, contrast=f"Gemini named - {SPECS[spec]} named", tasks=len(x), diff=round(dd.mean(), 3), ci=ci(dd)))
pair = pd.DataFrame(rows)
pair.to_csv(ROOT / "results/stage3b_paired.csv", index=False)
print("\nPaired contrasts (checked accuracy):")
print(pair.to_string(index=False))

# answers that did not touch the corrupted rule, unparsable final answers
last = d.history.apply(lambda h: h[-1])
d = d.assign(wrong_rule=last.apply(lambda r: r.get("truth_node_changed") is False),
             unparsed=last.apply(lambda r: r.get("error") is not None))
print("\nFinal answer left the corrupted rule untouched / could not be read:")
print(d.groupby(["spec", "condition"])[["wrong_rule", "unparsed"]].mean().round(3).to_string())

# calls and tokens per model for these tasks (runs/llm_log.jsonl)
ids = {(t, c) for t, c in zip(d.task_id, d.condition)}
cost = {}
for r in map(json.loads, (ROOT / "runs/llm_log.jsonl").open()):
    p = r["tag"].split("|")
    if len(p) >= 4 and p[0] == "s3b" and r["spec"] in SPECS and (p[1], p[2]) in ids:
        c = cost.setdefault(SPECS[r["spec"]], dict(calls=0, tokens_in=0, tokens_out=0))
        c["calls"] += 1
        c["tokens_in"] += r["tokens_in"]
        c["tokens_out"] += r["tokens_out"]
cost = pd.DataFrame(cost).T
cost.to_csv(ROOT / "results/stage3b_cost.csv")
print("\nCalls and tokens (includes calls from earlier passes of the same task):")
print(cost.to_string())
