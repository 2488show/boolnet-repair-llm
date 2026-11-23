"""Stage 3b, single model on all tasks: GPT-6-Luna (Codex CLI) on every solved Stage 3a task and on the
sampled original tasks, both name conditions. Kept apart from the three-model paired table
(scripts/stage3b_analyze.py), which uses only the 112 tasks every model finished.

When a (task, condition) was answered twice (a restart with more threads ran some original tasks twice), the
first record in the file is used; the agreement between the two runs is reported as a repeatability check
(Codex exposes no temperature).
Writes results/stage3b_luna_full.csv and results/stage3b_luna_repeat.csv.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = "codex:gpt-6-luna"
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    b = rng.choice(x, size=(2000, len(x))).mean(1)
    return f"{np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}"


d = pd.read_json(ROOT / "runs/stage3b_results.jsonl", lines=True)
d = d[d.spec == SPEC].reset_index(drop=True)
d["group"] = np.where(d.set == "orig", "original (sign / AND-OR)",
                      np.where(d.kind == "add_reg", "added regulator", "removed regulator"))

rep = d[d.duplicated(["task_id", "condition"], keep=False)].groupby(["task_id", "condition"])
pairs = [(g.iloc[0], g.iloc[1]) for _, g in rep]
repeat = pd.DataFrame([dict(pairs=len(pairs),
                            same_correctness=sum(a.checked_correct == b.checked_correct for a, b in pairs),
                            first_correct=sum(bool(a.checked_correct) for a, _ in pairs),
                            second_correct=sum(bool(b.checked_correct) for _, b in pairs))])
repeat.to_csv(ROOT / "results/stage3b_luna_repeat.csv", index=False)
print("repeat runs of the same (task, condition):")
print(repeat.to_string(index=False))

d = d.drop_duplicates(["task_id", "condition"], keep="first")
rows = []
for (g, cond), x in d.groupby(["group", "condition"]):
    rows.append(dict(group=g, condition=cond, tasks=len(x),
                     truth_among_solver_repairs=round(x.truth_in_candidates.mean(), 3),
                     alone=round(x.alone_correct.mean(), 3), alone_ci=ci(x.alone_correct),
                     checked=round(x.checked_correct.mean(), 3), checked_ci=ci(x.checked_correct),
                     checked_consistent=round(x.checked_consistent.mean(), 3)))
w = d.pivot_table(index=["task_id", "group"], columns="condition", values="checked_correct", aggfunc="first").dropna().astype(float)
t = d.drop_duplicates("task_id").set_index("task_id").truth_in_candidates.astype(float)
for g in ["added regulator", "removed regulator", "original (sign / AND-OR)", "all"]:
    x = w if g == "all" else w.xs(g, level="group", drop_level=False)
    nd = x.named - x.anonymous
    sv = x.named.values - t.loc[x.index.get_level_values("task_id")].values
    rows.append(dict(group=g, condition="named - anonymous (paired)", tasks=len(x), checked=round(nd.mean(), 3), checked_ci=ci(nd)))
    rows.append(dict(group=g, condition="named - truth among solver repairs", tasks=len(x), checked=round(sv.mean(), 3), checked_ci=ci(sv)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/stage3b_luna_full.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
