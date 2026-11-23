"""Stage 3a: compare the regulator add/delete corruptions with the original ones (Stage 1 only).

Original corruptions: results/stage1_newnets, stage1_moreseeds, stage1_5nets trials, restricted to the
same data levels (2 or 4 observed steady states; 70, 50 or 30% of variables measured).
Writes results/stage3a_corruption_comparison.csv.
"""
import pandas as pd

new = pd.read_csv("results/stage1_regcorrupt_trials.csv")
old = pd.concat([pd.read_csv(f"results/{f}_trials.csv")
                 for f in ["stage1_newnets", "stage1_moreseeds", "stage1_5nets"]])
old = old[old.n_states.isin([2, 4]) & old.coverage.isin([0.7, 0.5, 0.3])]
d = pd.concat([new, old])
d["kind"] = d.corrupted.fillna("").str.split(":").str[1]
d["st"] = d.status.str.replace(r"too_many:.*", "too_many", regex=True)
ok = d[d.st == "ok"]
amb = ok[ok.n_candidates >= 2]
g = d.groupby("kind")
out = pd.DataFrame({
    "trials": g.size(),
    "undetected": g.st.apply(lambda s: (s == "undetected").mean()),
    "timeout_or_too_many": g.st.apply(lambda s: s.isin(["timeout", "too_many"]).mean()),
    "solved": ok.groupby("kind").size(),
    "ambiguous_among_solved": ok.groupby("kind").n_candidates.apply(lambda s: (s >= 2).mean()),
    "truth_among_candidates_solved": ok.groupby("kind").truth_found.mean(),
    "truth_among_candidates_ambiguous": amb.groupby("kind").truth_found.mean(),
    "stage2_usable_tasks": ok[ok.n_candidates.between(2, 30) & (ok.truth_found == True)]  # noqa: E712
    .groupby("kind").size(),
}).loc[["flip", "and_or", "del_reg", "add_reg"]]
out.round(3).to_csv("results/stage3a_corruption_comparison.csv")
pd.set_option("display.width", 200)
print(out.round(3).to_string())
print("errors:", new.status.str.startswith("error").sum())
