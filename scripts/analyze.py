"""Tables and figures for Stage 1 (ambiguity) and Stage 2 (who picks the true repair).

Stage 2 accuracy = share of tasks where the chosen candidate is the true network. Random is the
exact expectation (share of correct candidates). 95% CIs: bootstrap over tasks (2,000 resamples);
differences between methods are paired on the same tasks.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
FIG = pathlib.Path("figures")
FIG.mkdir(exist_ok=True)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    m = rng.integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[m].mean(1), [2.5, 97.5]))


# ---------------- Stage 1 ----------------
s1 = pd.concat([pd.read_csv(f) for f in sorted(pathlib.Path("results").glob("stage1_*_trials.csv"))
                if "stage1_trials" not in f.name], ignore_index=True)
s1["level"] = np.where(s1.n_states.isin([2, 4]), s1.n_states.astype(str) + " states", "all states")
ok = s1[s1.status == "ok"].copy()
ok["ambiguous"] = ok.n_candidates >= 2
t1 = s1.groupby(["level", "coverage"]).agg(trials=("seed", "size"),
                                            undetected=("status", lambda s: (s == "undetected").mean()),
                                            too_many=("status", lambda s: s.str.startswith("too_many").mean()),
                                            timeout=("status", lambda s: (s == "timeout").mean()))
t1 = t1.join(ok.groupby(["level", "coverage"]).agg(
    solved=("seed", "size"), ambiguous=("ambiguous", "mean"), median_candidates=("n_candidates", "median"),
    truth_in_candidates=("truth_found", "mean"), random_pick=("frac_correct", "mean"),
    solver_first=("solver_first_correct", "mean")))
t1.round(3).to_csv("results/stage1_table.csv")
print(t1.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for lv, g in ok.groupby("level"):
    a = g.groupby("coverage").ambiguous.mean()
    axes[0].plot(a.index, a.values, "o-", label=lv)
amb = ok[ok.ambiguous]
for col, lab, st in [("truth_found", "true network among candidates", "o-"),
                     ("frac_correct", "random pick is correct", "s--")]:
    a = amb.groupby("coverage")[col].mean()
    axes[1].plot(a.index, a.values, st, label=lab)
axes[0].set(xlabel="Fraction of variables measured", ylabel="Tasks with ≥2 distinct minimal repairs",
            title="Less measured → more ambiguity")
axes[1].set(xlabel="Fraction of variables measured", ylabel="Share of ambiguous tasks",
            title="Ambiguous tasks: is the truth on the solver's list?", ylim=(0, 1))
for ax in axes:
    ax.invert_xaxis()
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "stage1_ambiguity.png", dpi=110)

# ---------------- Stage 2 ----------------
p = pathlib.Path("runs/stage2_results.jsonl")
if p.exists():
    r = pd.DataFrame(map(json.loads, p.open()))
    r = r.drop_duplicates(["task_id", "method"], keep="last")
    r = r[r.n_states.isin([2, 4]) & r.coverage.isin([0.3, 0.5, 0.7])]  # the Stage 2 grid
    # position bias: which displayed option number did each model answer?
    import re as _re
    llm_rows = r[r.method.str.contains("[|]", regex=True)].copy()
    llm_rows["shown"] = llm_rows.reply.str.extract(r'"choice"\s*:\s*(\d+)', expand=False).astype(float)
    pos = llm_rows.assign(first=llm_rows.shown == 1).groupby("method").agg(
        answered_option_1=("first", "mean"), mean_k=("k", "mean"),
        expected_if_no_bias=("k", lambda k: (1 / k).mean()))
    pos.round(3).to_csv("results/stage2_position_bias.csv")
    print(pos.round(3).to_string())
    chance = r.drop_duplicates("task_id")[["task_id", "network", "coverage", "chance", "k"]]
    rnd = chance.assign(method="random (expected)", correct=chance.chance)
    r = pd.concat([r, rnd], ignore_index=True)
    r["correct"] = r.correct.astype(float)
    rows = []
    for m, g in r.groupby("method"):
        lo, hi = boot_ci(g.correct)
        rows.append(dict(method=m, tasks=len(g), accuracy=g.correct.mean(), lo=lo, hi=hi,
                         unparsed=(~g.parsed.fillna(True).astype(bool)).mean() if "parsed" in g else 0))
    t2 = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    t2.round(3).to_csv("results/stage2_table.csv", index=False)
    print(t2.round(3).to_string(index=False))

    # paired differences on the same tasks
    wide = r.pivot_table(index="task_id", columns="method", values="correct")
    pairs = []
    llms = [m for m in wide if "|" in m]
    for m in llms:
        spec = m.split("|")[0]
        for ref in [f"{spec}|anonymous", f"{spec}|shuffled", "keep_regulation", "richest", "simplest",
                    "closest", "solver_first", "random (expected)"]:
            if ref in wide and ref != m:
                d = (wide[m] - wide[ref]).dropna()
                lo, hi = boot_ci(d)
                pairs.append(dict(method=m, versus=ref, tasks=len(d), difference=d.mean(), lo=lo, hi=hi))
    pd.DataFrame(pairs).round(3).to_csv("results/stage2_differences.csv", index=False)
    print(pd.DataFrame(pairs).round(3).to_string(index=False))

    # per network: is the effect carried by one network?
    pn = r.pivot_table(index="network", columns="method", values="correct", aggfunc="mean")
    pn.insert(0, "tasks", r.drop_duplicates("task_id").groupby("network").size())
    pn.round(3).to_csv("results/stage2_by_network.csv")

    # core figure: overall accuracy with 95% CIs, grouped by kind of chooser
    label = {"gemini:gemini-3.8-flash|named": ("Gemini 3.8 Flash, real names", "#1f6feb"),
             "gemini:gemini-3.8-flash|anonymous": ("Gemini 3.8 Flash, anonymous names", "#6ea8fe"),
             "gemini:gemini-3.8-flash|shuffled": ("Gemini 3.8 Flash, shuffled names", "#b6d4fe"),
             "local:mlx-community/Qwen3-8B-4bit|named": ("Qwen3-8B local, real names", "#d9480f"),
             "local:mlx-community/Qwen3-8B-4bit|anonymous": ("Qwen3-8B local, anonymous", "#fd7e14"),
             "local:mlx-community/Qwen3-8B-4bit|shuffled": ("Qwen3-8B local, shuffled", "#ffc078"),
             "keep_regulation": ("No AI: keep regulators", "#495057"),
             "richest": ("No AI: most complex rule", "#868e96"),
             "solver_first": ("No AI: solver's first answer", "#adb5bd"),
             "simplest": ("No AI: simplest rule", "#ced4da"),
             "random (expected)": ("Random pick (expected)", "#e9ecef")}
    t = t2[t2.method.isin(label)].copy()
    t["order"] = t.method.map(list(label).index)
    t = t.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(t))
    ax.barh(y, t.accuracy, xerr=[t.accuracy - t.lo, t.hi - t.accuracy],
            color=[label[m][1] for m in t.method], edgecolor="#343a40", linewidth=0.5, capsize=3)
    for yi, (a, h, n) in enumerate(zip(t.accuracy, t.hi, t.tasks)):
        ax.text(h + 0.015, yi, f"{a:.2f}  (n={n})", va="center", fontsize=8)
    ax.set_yticks(y, [label[m][0] for m in t.method], fontsize=9)
    ax.set(xlim=(0, 1.12), xlabel="Share of ambiguous tasks where the chosen repair is the true network",
           title="Choosing among equally minimal repairs (15 BBM networks)")
    fig.tight_layout()
    fig.savefig(FIG / "stage2_core.png", dpi=110)

    # secondary figure: by fraction measured, main methods only
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in ["gemini:gemini-3.8-flash|named", "gemini:gemini-3.8-flash|anonymous",
              "gemini:gemini-3.8-flash|shuffled", "keep_regulation", "random (expected)"]:
        a = r[r.method == m].groupby("coverage").correct.mean()
        ax.plot(a.index, a.values, "o-", color=label[m][1] if m != "random (expected)" else "#868e96",
                label=label[m][0])
    ax.invert_xaxis()
    ax.set(xlabel="Fraction of variables measured", ylabel="Chose the true network", ylim=(0, 1),
           title="Accuracy does not depend much on how much was measured")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_by_coverage.png", dpi=110)
