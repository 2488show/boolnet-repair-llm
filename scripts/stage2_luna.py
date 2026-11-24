"""GPT-6-Luna (Codex CLI) as a Stage 2 chooser, and picking vs repairing within the same model.

1. Stage 2 accuracy of GPT-6-Luna in the three name conditions on the 412 tasks, next to Gemini and the best
   knowledge-free rule, with paired contrasts (an unparsable answer counts as wrong, as for Gemini).
2. On the original tasks that also ran in Stage 3b, GPT-6-Luna picking from the solver's list (Stage 2) vs
   GPT-6-Luna repairing by itself (Stage 3b, first record per task and condition), paired by task.
Writes results/stage2_luna.csv and results/stage3b_pick_vs_repair_luna.csv.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
L = "codex:gpt-6-luna"


def ci(x):
    x = np.asarray(x, dtype=float)
    b = rng.choice(x, size=(2000, len(x))).mean(1)
    return f"{np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}"


s2 = pd.read_json(ROOT / "runs/stage2_results.jsonl", lines=True).drop_duplicates(["task_id", "method"])
w = s2.pivot_table(index="task_id", columns="method", values="correct", aggfunc="first").astype(float)
luna = [f"{L}|named", f"{L}|anonymous", f"{L}|shuffled"]
w = w.dropna(subset=luna)
rows = []
for m in luna + ["gemini:gemini-3.8-flash|named", "keep_regulation"]:
    x = w[m].dropna()
    rows.append(dict(item=m, tasks=len(x), value=round(x.mean(), 3), ci=ci(x)))
for a, b in [(luna[0], luna[1]), (luna[2], luna[1]), (luna[0], "keep_regulation"),
             ("gemini:gemini-3.8-flash|named", luna[0])]:
    x = (w[a] - w[b]).dropna()
    rows.append(dict(item=f"{a} - {b}", tasks=len(x), value=round(x.mean(), 3), ci=ci(x)))
parsed = s2[s2.method.isin(luna)].groupby("method").parsed.mean().round(3)
for m, v in parsed.items():
    rows.append(dict(item=f"{m}: share of answers parsed", tasks=int((s2.method == m).sum()), value=v, ci=""))
t1 = pd.DataFrame(rows)
t1.to_csv(ROOT / "results/stage2_luna.csv", index=False)
pd.set_option("display.width", 220)
print(t1.to_string(index=False))

s3 = pd.read_json(ROOT / "runs/stage3b_results.jsonl", lines=True)
s3 = s3[(s3.spec == L) & (s3.set == "orig")]
if "rep" in s3:
    s3 = s3[s3.rep.fillna(1) == 1]
s3 = s3.drop_duplicates(["task_id", "condition"], keep="first")
rows = []
for cond in ["named", "anonymous"]:
    r = s3[s3.condition == cond].set_index("task_id")
    p = w[f"{L}|{cond}"]
    ids = r.index.intersection(p.index)
    for col in ["alone_correct", "checked_correct"]:
        diff = r.loc[ids, col].astype(float) - p.loc[ids]
        rows.append(dict(condition=cond, repair=col, tasks=len(ids), pick=round(p.loc[ids].mean(), 3),
                         repair_acc=round(r.loc[ids, col].astype(float).mean(), 3), diff=round(diff.mean(), 3), ci=ci(diff)))
t2 = pd.DataFrame(rows)
t2.to_csv(ROOT / "results/stage3b_pick_vs_repair_luna.csv", index=False)
print("\nGPT-6-Luna, same tasks: repairing by itself minus picking from the solver's list")
print(t2.to_string(index=False))
