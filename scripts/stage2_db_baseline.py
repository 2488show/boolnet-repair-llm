"""Knowledge without a language model: pick among the solver's repairs using OmniPath.

OmniPath (omnipathdb.org, human + mouse, datasets omnipath, tf_target, kinaseextra, pathwayextra; downloaded to
data/omnipath/interactions.tsv) lists directed, signed molecular interactions. A candidate repair is scored on the
rules that differ between candidates:
  +1 for every regulator r -> t in the candidate rule that OmniPath supports with the same sign,
  -1 for every regulator whose OmniPath sign is the opposite (only opposite-sign evidence),
  -1 for every OmniPath-supported edge from a network variable to t that the candidate rule leaves out.
The highest-scoring candidates are tied; accuracy is the exact expected accuracy of a uniform pick among them.

Network variables are mapped to human gene symbols: exact symbol match after removing the BBM "v_" prefix, plus
the hand-written ALIAS table below (complexes and families map to several genes; an edge is supported if any gene
pair is). Non-mammalian networks (Drosophila, yeast, E. coli, lac operon) are not mapped; the chicken network is
mapped through human orthologs. Tasks where no candidate differs in score are reported as "not covered".
Writes results/stage2_db_baseline.csv and results/stage2_db_baseline_by_task.csv.
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import bnet  # noqa: E402
from stage2_run import load_tasks  # noqa: E402

ALIAS = {
    # 3 mammalian cell cycle
    "CycD1": ["CCND1"], "CycE1": ["CCNE1"], "ERa": ["ESR1"], "ErbB1": ["EGFR"], "ErbB1_2": ["EGFR", "ERBB2"],
    "ErbB1_3": ["EGFR", "ERBB3"], "ErbB2_3": ["ERBB2", "ERBB3"], "MEK1": ["MAP2K1"], "cMYC": ["MYC"],
    "p21": ["CDKN1A"], "p27": ["CDKN1B"], "pRB": ["RB1"],
    # 10 cardiac development
    "Foxc1_2": ["FOXC1", "FOXC2"], "GATAs": ["GATA4", "GATA5", "GATA6"], "Nkx2_5": ["NKX2-5"],
    # 100 AML
    "akt": ["AKT1", "AKT2", "AKT3"], "arf": ["CDKN2A"], "erk": ["MAPK1", "MAPK3"],
    # 174 hepatocellular carcinoma
    "AKT": ["AKT1", "AKT2", "AKT3"], "Bcatenin_memb": ["CTNNB1"], "Bcatenin_nuc": ["CTNNB1"], "Ecadherin": ["CDH1"],
    "GLI": ["GLI1", "GLI2"], "MEK": ["MAP2K1", "MAP2K2"], "NOTCH": ["NOTCH1"], "SMAD": ["SMAD2", "SMAD3", "SMAD4"],
    "SOS_GRB2": ["SOS1", "GRB2"], "TGFBR": ["TGFBR1", "TGFBR2"], "miR200": [],
    # 177 myeloid progenitors
    "EGR_NAB": ["EGR1", "EGR2", "NAB2"], "EKLF": ["KLF1"], "FOG1": ["ZFPM1"], "PU1": ["SPI1"], "SCL": ["TAL1"],
    # 182 Boolean cell cycle (mammalian)
    "CycA": ["CCNA2"], "CycB": ["CCNB1"], "CycD": ["CCND1"], "CycE": ["CCNE1"], "E2F": ["E2F1"], "Rb2": ["RBL2"],
    "UbcH10": ["UBE2C"], "p27_1_b1": ["CDKN1B"], "p27_1_b2": ["CDKN1B"], "p27_2": ["CDKN1B"],
    # 185 chicken sex determination (human orthologs)
    "AROMA": ["CYP19A1"], "B_CATENIN": ["CTNNB1"], "DMRT1_b1": ["DMRT1"], "DMRT1_b2": ["DMRT1"], "ESTRO": ["ESR1"],
    # 208 hematopoiesis ageing
    "CDK46CycD": ["CDK4", "CDK6", "CCND1"], "CIPKIP": ["CDKN1A", "CDKN1B"],
    # 278 stochasticity in hematopoiesis
    "AMPK": ["PRKAA1", "PRKAA2"], "HIF1": ["HIF1A"], "p53": ["TP53"], "SOD": ["SOD1", "SOD2"],
}
NOT_MAMMALIAN = {21, 63, 99, 104, 212}

op = pd.read_csv(ROOT / "data/omnipath/interactions.tsv", sep="\t")
op = op[op.is_directed.astype(str) == "True"]
SIGN = {}
for s, t, st, inh in zip(op.source_genesymbol.str.upper(), op.target_genesymbol.str.upper(),
                         op.is_stimulation.astype(str) == "True", op.is_inhibition.astype(str) == "True"):
    a = SIGN.setdefault((s, t), set())
    if st:
        a.add(1)
    if inh:
        a.add(-1)
    if not st and not inh:
        a.add(0)
SYMS = {g for pair in SIGN for g in pair}


def genes(var):
    v = re.sub(r"^v_", "", var)
    if v in ALIAS:
        return [g.upper() for g in ALIAS[v]]
    u = v.upper()
    return [u] if u in SYMS else []


def edge(r, t):
    """Set of signs OmniPath gives for any gene pair of variables r -> t (empty: no evidence)."""
    out = set()
    for a in genes(r):
        for b in genes(t):
            out |= SIGN.get((a, b), set())
    return out


def score(rule, t, variables):
    sg = bnet.signs(rule)
    s = 0
    for r, v in sg.items():
        if v in (None, 0):
            continue
        e = edge(r, t)
        if v in e:
            s += 1
        elif e and (-v) in e and v not in e:
            s -= 1
    for r in variables:
        if r != t and r not in sg and (edge(r, t) - {0}):
            s -= 1
    return s


survey = pd.read_csv(ROOT / "results/network_survey.csv").set_index("id")
tasks = load_tasks({(s, c) for s in (2, 4) for c in (0.7, 0.5, 0.3)}, 30)
rows = []
for t in tasks:
    cands = t["candidates"]
    variables = bnet.variables(t["corrupted"])
    changed = sorted({n for n in t["corrupted"]
                      if any(not bnet.same_function(c[n], cands[0][n]) for c in cands[1:])})
    mapped = t["network"] not in NOT_MAMMALIAN
    sc = [sum(score(c[n], n, variables) for n in changed) for c in cands] if mapped else [0] * len(cands)
    best = [i for i, v in enumerate(sc) if v == max(sc)]
    rows.append(dict(task_id=t["task_id"], network=t["network"], k=len(cands), mapped=mapped,
                     covered=mapped and len(best) < len(cands),
                     db_acc=np.mean([t["correct"][i] for i in best]),
                     chance=np.mean(t["correct"])))
d = pd.DataFrame(rows)
s2 = pd.read_json(ROOT / "runs/stage2_results.jsonl", lines=True).drop_duplicates(["task_id", "method"])
w = s2.pivot_table(index="task_id", columns="method", values="correct", aggfunc="first").astype(float)
d = d.join(w[["gemini:gemini-3.8-flash|named", "gemini:gemini-3.8-flash|anonymous", "keep_regulation",
              "codex:gpt-6-luna|named"]], on="task_id")
d.to_csv(ROOT / "results/stage2_db_baseline_by_task.csv", index=False)

rng = np.random.default_rng(0)


def ci(x):
    x = np.asarray(x, dtype=float)
    b = rng.choice(x, size=(2000, len(x))).mean(1)
    return f"{np.percentile(b, 2.5):.2f}-{np.percentile(b, 97.5):.2f}"


out = []
for name, x in [("all tasks", d), ("mammalian networks", d[d.mapped]), ("covered by OmniPath", d[d.covered])]:
    r = dict(subset=name, tasks=len(x))
    for col in ["db_acc", "chance", "keep_regulation", "gemini:gemini-3.8-flash|named",
                "gemini:gemini-3.8-flash|anonymous", "codex:gpt-6-luna|named"]:
        r[col] = round(x[col].mean(), 3)
    r["gemini_named_minus_db"] = round((x["gemini:gemini-3.8-flash|named"] - x.db_acc).mean(), 3)
    r["gemini_named_minus_db_ci"] = ci(x["gemini:gemini-3.8-flash|named"] - x.db_acc)
    r["db_minus_keep_regulation"] = round((x.db_acc - x.keep_regulation).mean(), 3)
    r["db_minus_keep_regulation_ci"] = ci(x.db_acc - x.keep_regulation)
    out.append(r)
out = pd.DataFrame(out)
out.to_csv(ROOT / "results/stage2_db_baseline.csv", index=False)
pd.set_option("display.width", 250)
print(out.T.to_string())
print("\ncovered tasks per network:", d[d.covered].network.value_counts().to_dict())
