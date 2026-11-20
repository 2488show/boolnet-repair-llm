"""Stage 1: no AI. How often do the data leave several equally minimal repairs, and is the true
network among them?

For each network, data level (number of observed steady states x fraction of genes measured) and
random seed: corrupt one rule of the true network, observe part of its steady states, let pyModRev
list all minimal repairs, and score them against the truth.
Writes results/stage1_trials.csv and results/stage1_summary.csv.
"""
import argparse
import pathlib
import random
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import benchmark as B  # noqa: E402
import bnet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ids", type=int, nargs="+", default=[100])
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--states", type=int, nargs="+", default=[1, 2, 4, 99])
ap.add_argument("--coverage", type=float, nargs="+", default=[1.0, 0.5])
ap.add_argument("--errors", type=int, default=1)
ap.add_argument("--out", default="results/stage1")
args = ap.parse_args()

survey = pd.read_csv("results/network_survey.csv").set_index("id")
rows = []
for nid in args.ids:
    folder = survey.loc[nid, "folder"]
    truth = bnet.read_bnet(f"vendor/bbm/models/{folder}/model.bnet")
    names = bnet.variables(truth)
    fps = bnet.fixed_points(truth)
    for n_states in args.states:
        for cov in args.coverage:
            for seed in range(args.trials):
                rng = random.Random(f"{nid}-{n_states}-{cov}-{seed}")
                bad, log = B.corrupt(truth, rng, args.errors)
                obs, chosen = B.observe(fps, names, rng, n_states, cov)
                row = dict(network=nid, n_states=min(n_states, len(fps)), coverage=cov, seed=seed,
                           corrupted=";".join(f"{t}:{k}:{d}" for t, k, d in log))
                if B.consistent(bad, obs):
                    row.update(status="undetected", n_candidates=0)
                    rows.append(row)
                    continue
                t0 = time.time()
                cands, status = B.solve(bad, obs, names)
                row.update(status=status, n_candidates=len(cands), solve_s=round(time.time() - t0, 2))
                if cands:
                    ok = [B.is_truth(c, truth) for c in cands]
                    row.update(truth_found=any(ok), frac_correct=sum(ok) / len(ok),
                               solver_first_correct=ok[0])
                rows.append(row)
            sub = [r for r in rows if r["network"] == nid and r["n_states"] == min(n_states, len(fps))
                   and r["coverage"] == cov]
            print(nid, n_states, cov, "| undetected", sum(r["status"] == "undetected" for r in sub),
                  "| ok", sum(r["status"] == "ok" for r in sub),
                  "| mean candidates", round(sum(r.get("n_candidates", 0) for r in sub if r["status"] == "ok")
                                            / max(1, sum(r["status"] == "ok" for r in sub)), 1), flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{args.out}_trials.csv", index=False)
ok = df[df.status == "ok"]
summ = df.groupby(["network", "n_states", "coverage"]).agg(
    trials=("seed", "size"),
    undetected=("status", lambda s: (s == "undetected").mean()),
    solved=("status", lambda s: (s == "ok").mean()),
).join(ok.groupby(["network", "n_states", "coverage"]).agg(
    median_candidates=("n_candidates", "median"),
    share_ambiguous=("n_candidates", lambda s: (s >= 2).mean()),
    truth_found=("truth_found", "mean"),
    random_pick_correct=("frac_correct", "mean"),
    solver_first_correct=("solver_first_correct", "mean"),
))
summ.to_csv(f"{args.out}_summary.csv")
pd.set_option("display.width", 200)
print(summ.round(2).to_string())
print("statuses:", df.status.value_counts().to_dict())
