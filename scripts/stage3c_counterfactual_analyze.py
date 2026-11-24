"""Counterfactual tasks (scripts/stage3c_counterfactual_build.py), GPT-6-Luna (Codex CLI).

For each condition: accuracy (the answer reproduces the counterfactual truth exactly), and how often the answer
changed the counterfactual node u at all ("touched") or put it back to its published rule ("reverted").
Baseline for touching u: the same model on the base task (same network, same corrupted node, published rule at
u), where changing u is equally unnecessary.
Writes results/stage3c_counterfactual.csv.
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
d = d[d.spec == SPEC]
if "rep" in d:
    d = d[d.rep.fillna(1) == 1]
cf = d[d.set == "cf"].drop_duplicates(["task_id", "condition"])
base = d[d.set == "reg"].drop_duplicates(["task_id", "condition"]).set_index(["task_id", "condition"])
import json  # noqa: E402

meta = {t: json.loads((ROOT / f"data/tasks_cf/{t}.json").read_text()) for t in cf.task_id.unique()}
rows = []
for cond, x in cf.groupby("condition"):
    r = dict(condition=cond, tasks=len(x))
    for col in ["alone_correct", "checked_correct", "alone_touched_cf_node", "alone_reverted_cf_node",
                "checked_touched_cf_node", "checked_reverted_cf_node"]:
        r[col] = round(x[col].astype(float).mean(), 3)
        r[col + "_ci"] = ci(x[col].astype(float))
    # same model, base task: did it change node u there (where u carries the published rule)?
    touched = []
    acc = []
    for _, row in x.iterrows():
        m = meta[row.task_id]
        key = (m["base_task"], cond)
        if key in base.index:
            h = base.loc[key, "history"]
            touched.append(m["cf_node"] in (h[0].get("changes") or {}))
            acc.append(bool(base.loc[key, "checked_correct"]))
    r["base_alone_touched_u"] = round(np.mean(touched), 3)
    r["base_alone_touched_u_ci"] = ci(touched)
    r["base_checked_correct"] = round(np.mean(acc), 3)
    r["base_tasks"] = len(touched)
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/stage3c_counterfactual.csv", index=False)
pd.set_option("display.width", 250)
print(out.T.to_string())
w = cf.pivot_table(index="task_id", columns="condition", values="alone_reverted_cf_node", aggfunc="first").dropna().astype(float)
diff = w.named - w.anonymous
print(f"\nfirst-answer revert, named - anonymous (paired, {len(w)} tasks): {diff.mean():.3f} CI {ci(diff)}")
