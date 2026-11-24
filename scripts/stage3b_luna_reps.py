"""Stage 3b repeatability: GPT-6-Luna (Codex CLI) answered every Stage 3b job three times (repeat 1 = the first
record of the full run; repeats 2 and 3 from scripts/stage3b_codex_reps.sh). Codex exposes no temperature.

Per group and name condition: mean "checked" accuracy of each repeat, accuracy of the per-task mean over the
three repeats (with a bootstrap CI over tasks), and how often the three repeats agree (all right, all wrong,
mixed). Paired contrasts on the per-task means: real minus anonymous names, and real names minus the share of
tasks whose truth is among the solver's minimal repairs.
Writes results/stage3b_luna_reps.csv.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    b = rng.choice(x, size=(2000, len(x))).mean(1)
    return f"{np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}"


d = pd.read_json(ROOT / "runs/stage3b_results.jsonl", lines=True)
d = d[(d.spec == "codex:gpt-6-luna") & d.set.isin(["reg", "orig"])].copy()
d["rep"] = d.get("rep", pd.Series(1, index=d.index)).fillna(1).astype(int)
d = d.drop_duplicates(["task_id", "condition", "rep"], keep="first")
d["group"] = np.where(d.set == "orig", "original (sign / AND-OR)",
                      np.where(d.kind == "add_reg", "added regulator", "removed regulator"))
w = d.pivot_table(index=["task_id", "group", "condition"], columns="rep", values="checked_correct",
                  aggfunc="first").dropna().astype(float)
w.columns = [f"rep{c}" for c in w.columns]
w["mean"] = w[["rep1", "rep2", "rep3"]].mean(1)
s = w[["rep1", "rep2", "rep3"]].sum(1)
rows = []
for (g, cond), x in w.groupby(level=["group", "condition"]):
    k = s.loc[x.index]
    rows.append(dict(group=g, condition=cond, tasks=len(x),
                     rep1=round(x.rep1.mean(), 3), rep2=round(x.rep2.mean(), 3), rep3=round(x.rep3.mean(), 3),
                     mean_of_3=round(x["mean"].mean(), 3), mean_ci=ci(x["mean"]),
                     all_right=round((k == 3).mean(), 3), all_wrong=round((k == 0).mean(), 3),
                     mixed=round(((k > 0) & (k < 3)).mean(), 3)))
truth = d.drop_duplicates("task_id").set_index("task_id").truth_in_candidates.astype(float)
m = w["mean"].unstack("condition").dropna()
for g in ["added regulator", "removed regulator", "original (sign / AND-OR)", "all"]:
    x = m if g == "all" else m.xs(g, level="group", drop_level=False)
    nd = x.named - x.anonymous
    sv = x.named.values - truth.loc[x.index.get_level_values("task_id")].values
    rows.append(dict(group=g, condition="named - anonymous (mean of 3, paired)", tasks=len(x),
                     mean_of_3=round(nd.mean(), 3), mean_ci=ci(nd)))
    rows.append(dict(group=g, condition="named - truth among solver repairs", tasks=len(x),
                     mean_of_3=round(sv.mean(), 3), mean_ci=ci(sv)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/stage3b_luna_reps.csv", index=False)
pd.set_option("display.width", 250)
print(out.to_string(index=False))
