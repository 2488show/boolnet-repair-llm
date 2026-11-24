"""Counterfactual tasks: does the model override the data with what it believes the published model says?

For a Stage 3b task (published network T, corrupted rule at node t), a counterfactual truth T' is made by flipping
the sign of one regulator in the rule of another node u (u != t). T' is a self-consistent Boolean network, so the
steady-state measurements are regenerated from T'; then the same corruption as in the original task is applied
to node t of T'. The model sees T' with the corrupted rule and the measurements of T', exactly as in Stage 3b.

The correct answer restores only node t. Changing node u back to its published rule is never needed by the data
(T' fits the measurements) and is scored separately as a "revert". Reverts with real names but not with anonymous
names show that prior knowledge of these molecules overrides the given model. This does not separate memory of
the specific paper from general biological knowledge; both point to the published sign.

Writes data/tasks_cf/<task_id>_cf.json (one per base task; skipped if no valid u is found).
Usage: stage3c_counterfactual_build.py --n 200 [--seed 0]
"""
import argparse
import json
import pathlib
import random
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import benchmark as B  # noqa: E402
import bnet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

survey = pd.read_csv(ROOT / "results/network_survey.csv").set_index("id")
base = [json.loads(f.read_text()) for f in sorted((ROOT / "data/tasks_reg").glob("*.json"))]
base = [t for t in base if t["corruption"][0][1] == "add_reg"]  # the corruption the solver cannot undo
rng = random.Random(f"cf-{args.seed}")
rng.shuffle(base)
out_dir = ROOT / "data/tasks_cf"
out_dir.mkdir(parents=True, exist_ok=True)
made = skipped = 0
for t in base:
    if made >= args.n:
        break
    truth = bnet.read_bnet(ROOT / f"vendor/bbm/models/{survey.loc[t['network'], 'folder']}/model.bnet")
    names = bnet.variables(truth)
    target = t["corruption"][0][0]
    r = random.Random(t["task_id"])
    nodes = [u for u in truth if u != target and bnet.regulators(truth[u])]
    r.shuffle(nodes)
    ok = False
    for u in nodes:
        regs = bnet.regulators(truth[u])
        reg = r.choice(regs)
        cf = dict(truth)
        cf[u] = B.flip_sign(truth[u], reg)
        if bnet.same_function(cf[u], truth[u]):
            continue
        fps = bnet.fixed_points(cf)
        if len(fps) < 2:
            continue
        obs, chosen = B.observe(fps, names, r, t["n_states"], t["coverage"])
        corrupted = dict(cf)
        corrupted[target] = t["corrupted"][target]
        if B.consistent(corrupted, obs):
            continue  # the error must be visible in the data
        ok = True
        break
    if not ok:
        skipped += 1
        continue
    task = dict(task_id=t["task_id"] + "_cf", base_task=t["task_id"], network=t["network"], n_states=t["n_states"],
                coverage=t["coverage"], seed=t["seed"], corruption=t["corruption"], observations=obs,
                observed_states=chosen, corrupted=corrupted, candidates=[], correct=[], truth=cf,
                cf_node=u, cf_regulator=reg, cf_published_rule=truth[u], cf_rule=cf[u])
    (out_dir / f"{task['task_id']}.json").write_text(json.dumps(task))
    made += 1
print(f"made {made} counterfactual tasks, skipped {skipped}")
