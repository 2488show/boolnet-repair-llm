"""Stage 2 pilot: a handful of ambiguous AML tasks, two name conditions, to check the prompt,
the answer parsing, token use and cost before scaling up. Not a result."""
import json
import pathlib
import random
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import benchmark as B  # noqa: E402
import bnet  # noqa: E402
import chooser  # noqa: E402

SPEC = sys.argv[1] if len(sys.argv) > 1 else "gemini:gemini-3.8-flash"
N_TASKS = 8
survey = pd.read_csv("results/network_survey.csv").set_index("id")
truth = bnet.read_bnet(f"vendor/bbm/models/{survey.loc[100, 'folder']}/model.bnet")
names, fps = bnet.variables(truth), bnet.fixed_points(truth)

tasks, seed = [], 0
while len(tasks) < N_TASKS and seed < 200:
    rng = random.Random(f"pilot-{seed}")
    seed += 1
    bad, log = B.corrupt(truth, rng, 1)
    obs, _ = B.observe(fps, names, rng, 4, 0.5)
    if B.consistent(bad, obs):
        continue
    cands, status = B.solve(bad, obs, names)
    ok = [B.is_truth(c, truth) for c in cands]
    if status == "ok" and 2 <= len(cands) <= 30 and any(ok):
        tasks.append(dict(seed=seed, bad=bad, cands=cands, ok=ok, log=log))
        print(f"task {len(tasks)}: {len(cands)} candidates, {sum(ok)} correct, corrupted {log}", flush=True)

rows = []
for i, t in enumerate(tasks):
    for cond in ("named", "anonymous"):
        idx, reply = chooser.choose(SPEC, t["bad"], t["cands"], cond, seed=f"{t['seed']}-{cond}",
                                    tag=f"pilot-{i}-{cond}")
        rows.append(dict(task=i, condition=cond, k=len(t["cands"]), chance=sum(t["ok"]) / len(t["ok"]),
                         parsed=idx is not None, correct=bool(idx is not None and t["ok"][idx]),
                         reply=reply[:160].replace("\n", " ")))
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print(df.groupby("condition")[["parsed", "correct", "chance"]].mean().round(2))
log = [json.loads(l) for l in open("runs/llm_log.jsonl") if '"pilot-' in l]
print("calls", len(log), "tokens in", sum(x["tokens_in"] for x in log), "out", sum(x["tokens_out"] for x in log),
      "mean s", round(sum(x["seconds"] for x in log) / max(1, len(log)), 1))
