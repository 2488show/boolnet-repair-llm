"""Stage 1: no AI. How often do the data leave several equally minimal repairs, and is the true
network among them?

For each network, data level (number of observed steady states x fraction of variables measured)
and random seed: corrupt one rule of the true network, observe part of its steady states, let
pyModRev list all minimal repairs (duplicates merged), and score them against the truth.
Every solved task is saved to data/tasks/ so that Stage 2 asks the language models about exactly
the same tasks. Writes <out>_trials.csv and <out>_summary.csv.
"""
import argparse
import json
import pathlib
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import benchmark as B  # noqa: E402
import bnet  # noqa: E402

TASKS = pathlib.Path("data/tasks")


def run_trial(nid, folder, n_states, cov, seed, n_errors, kinds=("flip", "and_or"), tasks_dir="data/tasks"):
    truth = bnet.read_bnet(f"vendor/bbm/models/{folder}/model.bnet")
    names = bnet.variables(truth)
    fps = bnet.fixed_points(truth)
    tag = "" if tuple(kinds) == ("flip", "and_or") else "-" + "+".join(kinds)
    rng = random.Random(f"{nid}-{n_states}-{cov}-{seed}{tag}")
    bad, log = B.corrupt(truth, rng, n_errors, kinds)
    obs, chosen = B.observe(fps, names, rng, n_states, cov)
    row = dict(network=nid, n_states=min(n_states, len(fps)), coverage=cov, seed=seed,
               corrupted=";".join(f"{t}:{k}:{d}" for t, k, d in log))
    if B.consistent(bad, obs):
        row.update(status="undetected", n_candidates=0)
        return row
    t0 = time.time()
    cands, status = B.solve(bad, obs, names)
    row.update(status=status, n_candidates=len(cands), solve_s=round(time.time() - t0, 2))
    if cands:
        ok = [B.is_truth(c, truth) for c in cands]
        row.update(truth_found=any(ok), frac_correct=sum(ok) / len(ok), solver_first_correct=ok[0])
        task_id = f"n{nid}_s{row['n_states']}_c{int(cov * 100)}_r{seed}" + ("" if not tag else "_" + "".join(k[0] + k.split("_")[-1][0] if "_" in k else k[0] for k in kinds))
        row["task_id"] = task_id
        (pathlib.Path(tasks_dir) / f"{task_id}.json").write_text(json.dumps(dict(
            task_id=task_id, network=nid, n_states=row["n_states"], coverage=cov, seed=seed,
            corruption=log, observations=obs, observed_states=chosen, corrupted=bad,
            candidates=cands, correct=ok)))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[100])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--states", type=int, nargs="+", default=[1, 2, 4, 99])
    ap.add_argument("--coverage", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--errors", type=int, default=1)
    ap.add_argument("--kinds", nargs="+", default=["flip", "and_or"], choices=["flip", "and_or", "del_reg", "add_reg"])
    ap.add_argument("--tasks-dir", default="data/tasks")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="results/stage1")
    args = ap.parse_args()

    pathlib.Path(args.tasks_dir).mkdir(parents=True, exist_ok=True)
    survey = pd.read_csv("results/network_survey.csv").set_index("id")
    jobs = [(nid, survey.loc[nid, "folder"], s, c, seed, args.errors, tuple(args.kinds), args.tasks_dir)
            for nid in args.ids for s in args.states for c in args.coverage for seed in range(args.seed_start, args.seed_start + args.trials)]
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_trial, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} trials, {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows).sort_values(["network", "n_states", "coverage", "seed"])
    df.to_csv(f"{args.out}_trials.csv", index=False)
    keys = ["network", "n_states", "coverage"]
    ok = df[df.status == "ok"]
    summ = df.groupby(keys).agg(
        trials=("seed", "size"),
        undetected=("status", lambda s: (s == "undetected").mean()),
        timeout=("status", lambda s: (s == "timeout").mean()),
    ).join(ok.groupby(keys).agg(
        solved=("seed", "size"),
        median_candidates=("n_candidates", "median"),
        share_ambiguous=("n_candidates", lambda s: (s >= 2).mean()),
        truth_found=("truth_found", "mean"),
        random_pick_correct=("frac_correct", "mean"),
        solver_first_correct=("solver_first_correct", "mean"),
    ))
    summ.to_csv(f"{args.out}_summary.csv")
    pd.set_option("display.width", 220)
    print(summ.round(2).to_string())
    print("statuses:", df.status.value_counts().to_dict())
