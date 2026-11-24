"""Stage 3b: the language model repairs without the solver's candidate list; the solver's role is
reduced to checking. Results go to runs/stage3b_results.jsonl (one line per task and condition);
reruns skip finished work.

Only the "checked" mode is run: its first round uses exactly the same prompt at temperature 0 as the
"alone" mode would, so the "alone" result is the first round of each record.

Task sets:
  orig  the Stage 2 tasks (ambiguous, truth among the solver's repairs), so the result can be compared
        with picking from the candidate list on the same tasks
  reg   every solved Stage 3a task (an added or removed regulator), including those where the truth is
        not among the solver's minimal repairs
"""
import argparse
import json
import pathlib
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bnet  # noqa: E402
import repairer  # noqa: E402
from stage2_run import load_tasks, spent_usd  # noqa: E402

OUT = ROOT / "runs/stage3b_results.jsonl"
LEVELS = {(s, c) for s in (2, 4) for c in (0.7, 0.5, 0.3)}
lock = threading.Lock()


def load_reg(folder="data/tasks_reg"):
    out = []
    for f in sorted((ROOT / folder).glob("*.json")):
        out.append(json.loads(f.read_text()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["orig", "reg"])
    ap.add_argument("--spec", default="gemini:gemini-3.8-flash")
    ap.add_argument("--conditions", nargs="+", default=["named", "anonymous"])
    ap.add_argument("--sample", type=int, default=0, help="random sample of N tasks per set (0 = all)")
    ap.add_argument("--same-tasks-as", default="", help="only tasks already finished (both conditions) by this spec")
    ap.add_argument("--rep", type=int, default=1, help="repeat number; repeats >1 are new samples of the same job")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=30.0, help="stop when total Gemini spend (all stages) exceeds this")
    args = ap.parse_args()

    survey = pd.read_csv(ROOT / "results/network_survey.csv").set_index("id")
    truths = {}
    tasks = []
    for s in args.sets:
        ts = load_tasks(LEVELS, 30) if s == "orig" else load_reg("data/tasks_cf" if s == "cf" else "data/tasks_reg")
        if args.sample and len(ts) > args.sample:
            ts = random.Random(f"stage3b-{s}").sample(ts, args.sample)
        tasks += [(s, t) for t in ts]
    if args.same_tasks_as:
        rows = [r for r in map(json.loads, OUT.open()) if r["spec"] == args.same_tasks_as]
        by = {}
        for r in rows:
            by.setdefault(r["task_id"], set()).add(r["condition"])
        keep = {t for t, c in by.items() if set(args.conditions) <= c}
        tasks = [(s, t) for s, t in tasks if t["task_id"] in keep]
    done = set()
    if OUT.exists():
        done = {(r["task_id"], r["condition"]) for r in map(json.loads, OUT.open())
                if r["spec"] == args.spec and r.get("rep", 1) == args.rep}
    print(f"{len(tasks)} tasks x {len(args.conditions)} conditions; done {len(done)}; spent so far ${spent_usd():.2f}", flush=True)

    def truth_of(nid):
        if nid not in truths:
            truths[nid] = bnet.read_bnet(ROOT / f"vendor/bbm/models/{survey.loc[nid, 'folder']}/model.bnet")
        return truths[nid]

    def run(job):
        s, t, cond = job
        if (t["task_id"], cond) in done or (args.spec.startswith("gemini") and spent_usd() > args.budget):
            return
        try:
            r = repairer.repair(args.spec, t, t.get("truth") or truth_of(t["network"]), cond, "checked", args.rounds,
                                tag=f"s3b|{t['task_id']}|{cond}" + (f"|rep{args.rep}" if args.rep > 1 else ""))
        except Exception as e:  # keep going; retried on the next run
            print("error", t["task_id"], cond, type(e).__name__, str(e)[:120], flush=True)
            return
        h0 = r["history"][0]
        row = dict(task_id=t["task_id"], set=s, network=t["network"], n_states=t["n_states"],
                   coverage=t["coverage"], kind=t["corruption"][0][1], condition=cond, spec=args.spec, rep=args.rep,
                   k=len(t["candidates"]), truth_in_candidates=any(t["correct"]),
                   alone_parsed=h0.get("error") is None, alone_consistent=bool(h0.get("consistent")),
                   alone_correct=bool(h0.get("truth")),
                   checked_rounds=r["rounds"], checked_consistent=r["consistent"], checked_correct=r["correct"],
                   history=r["history"])
        if "cf_node" in t:  # counterfactual task: did an answer put node u back to its published rule?
            u, pub = t["cf_node"], t["cf_published_rule"]
            for key, rec in (("alone", r["history"][0]), ("checked", r["history"][-1])):
                ch = rec.get("changes") or {}
                row[f"{key}_touched_cf_node"] = u in ch
                row[f"{key}_reverted_cf_node"] = u in ch and bnet.same_function(ch[u], pub)
        with lock, OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")

    jobs = [(s, t, c) for s, t in tasks for c in args.conditions]
    with ThreadPoolExecutor(args.threads) as ex:
        for n, _ in enumerate(ex.map(run, jobs), 1):
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}  spent ${spent_usd():.2f}", flush=True)
    finished = {(r["task_id"], r["condition"]) for r in map(json.loads, OUT.open())
                if r["spec"] == args.spec and r.get("rep", 1) == args.rep} if OUT.exists() else set()
    left = sum((t["task_id"], c) not in finished for _, t, c in jobs)
    print(f"done; {left} jobs left; total Gemini spend ${spent_usd():.2f}", flush=True)
    sys.exit(0 if left == 0 else 3)
