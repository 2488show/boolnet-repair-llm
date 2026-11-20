"""Stage 1a: which BBM networks are usable benchmarks?

Usable = 10-24 variables (exhaustive fixed-point search), every rule monotone (pyModRev repairs
within monotone functions, so the true answer must be reachable), and at least 3 fixed points
(steady-state observations are the experimental data; with fewer there is too little to observe).
Writes results/network_survey.csv.
"""
import json
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import bnet  # noqa: E402

rows = []
for d in sorted(p for p in pathlib.Path("vendor/bbm/models").iterdir() if p.is_dir()):
    meta = json.loads((d / "metadata.json").read_text())
    if not (10 <= meta["variables"] <= 24 and meta.get("inputs", 0) <= 3):
        continue
    net = bnet.read_bnet(d / "model.bnet")
    n = len(bnet.variables(net))
    t0 = time.time()
    mono = bnet.is_monotone(net)
    fps = bnet.fixed_points(net) if n <= 24 else []
    year = meta.get("url-publication", "")
    rows.append(dict(id=meta["id"], name=meta["name"], n_vars=n, n_inputs=len(bnet.inputs(net)),
                     regulations=meta.get("regulations"), monotone=mono, fixed_points=len(fps),
                     seconds=round(time.time() - t0, 1), folder=d.name, publication=year))
    print(rows[-1]["id"], rows[-1]["name"][:30], n, mono, len(fps), flush=True)

df = pd.DataFrame(rows)
df["usable"] = df.monotone & (df.fixed_points >= 3)
df.to_csv("results/network_survey.csv", index=False)
print(f"\n{df.usable.sum()} of {len(df)} networks usable")
print(df[df.usable][["id", "name", "n_vars", "n_inputs", "fixed_points"]].to_string(index=False))
