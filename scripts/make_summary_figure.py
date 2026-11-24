"""Summary figure for the README (figures/summary.png), drawn only from the committed result tables.

A  Stage 2: picking among the solver's minimal repairs (412 tasks)         results/stage2_table.csv, stage2_luna.csv
B  Stage 3a: is the true network among the solver's minimal repairs?       results/stage3a_corruption_comparison.csv
C  Stage 3b: added-regulator tasks, the model repairs by itself (77 tasks)  results/stage3b_table.csv
D  OmniPath (no language model) on the 119 tasks it can decide              results/stage2_db_baseline.csv
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = pathlib.Path(__file__).resolve().parents[1] / "results"
BLUE, LIGHT, GREY, DARK = "#3B6EA8", "#9DB8D9", "#B8B8B8", "#4D4D4D"

s2 = pd.read_csv(R / "stage2_table.csv").set_index("method")
lu = pd.read_csv(R / "stage2_luna.csv").set_index("item")
a3 = pd.read_csv(R / "stage3a_corruption_comparison.csv").set_index("kind")
b3 = pd.read_csv(R / "stage3b_table.csv")
db = pd.read_csv(R / "stage2_db_baseline.csv").set_index("subset")

fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
plt.rcParams.update({"font.size": 10})


def bars(a, labels, values, colors, title, ylim=1.0, err=None):
    x = range(len(values))
    a.bar(x, values, color=colors, yerr=err, capsize=3, edgecolor="none")
    a.set_xticks(list(x), labels, rotation=0)
    a.set_ylim(0, ylim)
    a.set_title(title, loc="left", fontsize=11, fontweight="bold")
    a.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(values):
        top = v + (err[1][i] if err is not None else 0)
        a.text(i, top + 0.02, f"{v:.2f}", ha="center", fontsize=9)


# A
rows = [("Gemini\nreal", "gemini:gemini-3.8-flash|named", BLUE), ("Gemini\nanon.", "gemini:gemini-3.8-flash|anonymous", LIGHT),
        ("Gemini\nshuffled", "gemini:gemini-3.8-flash|shuffled", LIGHT), ("Luna", None, GREY),
        ("Qwen3-8B\nreal", "local:mlx-community/Qwen3-8B-4bit|named", GREY),
        ("Keep\nregulators", "keep_regulation", DARK), ("Random", "random (expected)", DARK)]
labels, vals, lo_e, hi_e = [], [], [], []
for lab, m, _ in rows:
    if m is None:
        v = lu.loc["codex:gpt-6-luna|named", "value"]
        lo, hi = map(float, lu.loc["codex:gpt-6-luna|named", "ci"].split("-"))
        lab = "GPT-6-Luna\nreal"
    else:
        v, lo, hi = s2.loc[m, "accuracy"], s2.loc[m, "lo"], s2.loc[m, "hi"]
    labels.append(lab)
    vals.append(v)
    lo_e.append(v - lo)
    hi_e.append(hi - v)
bars(ax[0, 0], labels, vals, [r[2] for r in rows], "A  Pick among the solver's minimal repairs (412 tasks)",
     err=[lo_e, hi_e])
ax[0, 0].tick_params(axis="x", labelsize=8.5)
ax[0, 0].set_ylabel("picks the true network")

# B
kinds = [("Flip a sign", "flip"), ("AND <-> OR", "and_or"), ("Remove a\nregulator", "del_reg"), ("Add a spurious\nregulator", "add_reg")]
bars(ax[0, 1], [k for k, _ in kinds], [a3.loc[c, "truth_among_candidates_solved"] for _, c in kinds],
     [GREY, GREY, BLUE, BLUE], "B  True network among the minimal repairs (solved tasks)")
ax[0, 1].set_ylabel("share of solved tasks")

# C
x = b3[b3.group == "added regulator"]
models = ["Gemini 3.8 Flash (API)", "DeepSeek V4.1 Flash (API)", "GPT-6-Luna (Codex CLI)"]
short = ["Gemini", "DeepSeek", "GPT-6-Luna"]
a = ax[1, 0]
w = 0.38
for i, m in enumerate(models):
    for j, (cond, col) in enumerate([("named", BLUE), ("anonymous", LIGHT)]):
        v = x[(x.model == m) & (x.condition == cond)].checked.iloc[0]
        a.bar(i + (j - 0.5) * w, v, w, color=col, label=("real names" if cond == "named" else "anonymous") if i == 0 else None)
        a.text(i + (j - 0.5) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
ceiling = x.truth_among_solver_repairs.iloc[0]
a.axhline(ceiling, color=DARK, ls="--", lw=1, label=f"truth among solver's repairs ({ceiling:.2f})")
a.set_xticks(range(3), short)
a.set_ylim(0, 1)
a.set_ylabel("repairs to the true network")
a.legend(frameon=False, loc="upper right")
a.set_title("C  Added regulator: the model repairs by itself (77 tasks)", loc="left", fontsize=11, fontweight="bold")
a.spines[["top", "right"]].set_visible(False)

# D
c = db.loc["covered by OmniPath"]
bars(ax[1, 1], ["OmniPath\n(no LLM)", "Gemini\nreal", "Gemini\nanon.", "Keep\nregulators", "Random"],
     [c["db_acc"], c["gemini:gemini-3.8-flash|named"], c["gemini:gemini-3.8-flash|anonymous"], c["keep_regulation"], c["chance"]],
     [DARK, BLUE, LIGHT, DARK, DARK], f"D  Tasks a database can decide ({int(c['tasks'])} of 412)")
ax[1, 1].set_ylabel("picks the true network")

fig.tight_layout()
out = pathlib.Path(__file__).resolve().parents[1] / "figures/summary.png"
fig.savefig(out, dpi=160)
print("wrote", out)
