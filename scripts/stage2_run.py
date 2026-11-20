"""Stage 2: let each method pick one candidate on every ambiguous task, and record whether it is the
true network. Results are appended to runs/stage2_results.jsonl; reruns skip finished work.

Tasks used: saved by Stage 1, at the chosen data levels, with 2-30 distinct candidates and the truth
among them (tasks where the truth is missing cannot be won by any chooser and are counted apart).
"""
import argparse
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import chooser  # noqa: E402
import heuristics  # noqa: E402

OUT = pathlib.Path("runs/stage2_results.jsonl")
PRICE = {"gemini-3.8-flash": (0.75, 3.75)}  # USD per 1M input / output tokens, valid to 2026-12-31
lock = threading.Lock()


def load_tasks(levels, max_cands):
    tasks = []
    for f in sorted(pathlib.Path("data/tasks").glob("*.json")):
        t = json.loads(f.read_text())
        c = round(t["coverage"], 2)
        # "all" = every steady state of the network was observed (all usable networks have > 4)
        is_all = t["n_states"] not in (1, 2, 4)
        if (t["n_states"], c) not in levels and not (is_all and ("all", c) in levels):
            continue
        if 2 <= len(t["candidates"]) <= max_cands and any(t["correct"]):
            tasks.append(t)
    return tasks


def done_keys():
    if not OUT.exists():
        return set()
    return {(r["task_id"], r["method"]) for r in map(json.loads, OUT.open())}


def record(row):
    with lock, OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def spent_usd():
    log = pathlib.Path("runs/llm_log.jsonl")
    total = 0.0
    if log.exists():
        for r in map(json.loads, log.open()):
            model = r["spec"].split(":", 1)[1]
            if model in PRICE:
                pi, po = PRICE[model]
                total += (r["tokens_in"] * pi + r["tokens_out"] * po) / 1e6
    return total


def base_row(t, method):
    return dict(task_id=t["task_id"], network=t["network"], n_states=t["n_states"], coverage=t["coverage"],
                k=len(t["candidates"]), chance=sum(t["correct"]) / len(t["correct"]), method=method)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", required=True, help="n_states:coverage, e.g. 4:0.5 all:0.3")
    ap.add_argument("--llm", nargs="*", default=[], help="specs, e.g. gemini:gemini-3.8-flash")
    ap.add_argument("--conditions", nargs="+", default=["named", "anonymous", "shuffled"])
    ap.add_argument("--max-cands", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (pilot)")
    args = ap.parse_args()

    levels = set()
    for lv in args.levels:
        s, c = lv.split(":")
        levels.add((s if s == "all" else int(s), round(float(c), 2)))
    tasks = load_tasks(levels, args.max_cands)
    if args.limit:
        tasks = tasks[: args.limit]
    done = done_keys()
    print(f"{len(tasks)} tasks; already done {len(done)} task-method pairs", flush=True)

    # baselines without AI: instant
    for t in tasks:
        for name, fn in [("simplest", heuristics.simplest), ("closest", heuristics.closest),
                         ("richest", heuristics.richest), ("keep_regulation", heuristics.keep_regulation)]:
            if (t["task_id"], name) not in done:
                i = fn(t, seed=t["task_id"])
                record({**base_row(t, name), "choice": i, "correct": t["correct"][i]})
        if (t["task_id"], "solver_first") not in done:
            record({**base_row(t, "solver_first"), "choice": 0, "correct": t["correct"][0]})

    def ask(t, spec, cond):
        method = f"{spec}|{cond}"
        if (t["task_id"], method) in done:
            return
        if spec.startswith("gemini") and spent_usd() > args.budget:
            return
        try:
            i, reply = chooser.choose(spec, t["corrupted"], t["candidates"], cond,
                                      seed=f"{t['task_id']}-{cond}", tag=f"s2|{t['task_id']}|{method}")
            record({**base_row(t, method), "choice": i, "parsed": i is not None,
                    "correct": bool(i is not None and t["correct"][i]), "reply": reply[:300]})
        except Exception as e:  # keep going; the task is retried on the next run
            print("error", t["task_id"], method, type(e).__name__, str(e)[:120], flush=True)

    for spec in args.llm:
        threads = args.threads if spec.startswith("gemini") else 1  # one local model server
        jobs = [(t, spec, c) for t in tasks for c in args.conditions]
        with ThreadPoolExecutor(threads) as ex:
            for n, _ in enumerate(ex.map(lambda j: ask(*j), jobs), 1):
                if n % 50 == 0:
                    print(f"{spec}: {n}/{len(jobs)}  spent ${spent_usd():.2f}", flush=True)
        if spec.startswith("gemini") and spent_usd() > args.budget:
            print(f"budget ${args.budget} reached, stopped", flush=True)
    print(f"total Gemini spend so far: ${spent_usd():.2f}")
