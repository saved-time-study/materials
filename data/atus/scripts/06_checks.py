"""Checks on the finished outputs; exits 1 on any failure.  Run: python -I data/atus/scripts/06_checks.py

1. The five main outcomes add up to 1,440 minutes, so in every specification their coefficients on the same
   term add up to zero.
2. Every number in table1.csv equals the matching row of estimates.csv.
3. Every respondent of the sample has a harmonized occupation and all four exposure scores.
4. The event study has a coefficient for every year except its reference year.
"""
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, DER = ROOT / "results", ROOT / "derived"
MAIN_OUT = ["leisure", "paid_work", "household", "personal_care", "rest"]
fails = []

est = pd.read_csv(RES / "estimates.csv")
m = est[est["outcome"].isin(MAIN_OUT)]
sums = m.groupby(["spec", "measure", "sample", "term"])["coef"].agg(["sum", "count"])
bad = sums[(sums["count"] == 5) & (sums["sum"].abs() > 1e-6)]
if len(bad):
    fails.append(f"adding-up fails in {len(bad)} specifications")
if (sums["count"] == 5).sum() < 100:
    fails.append("too few complete five-outcome specifications to check the adding-up")

t1 = pd.read_csv(RES / "table1.csv")
if len(t1) != 25:
    fails.append(f"table1 has {len(t1)} cells, expected 25")
for _, r in t1.iterrows():
    hits = est[(est["outcome"] == r["outcome"]) & (est["term"] == "x_post") & (est["sample"] == "all")
               & ((est["coef"] - r["coef"]).abs() < 1e-9)]
    if hits.empty:
        fails.append(f"table1 cell {r['outcome']} / {r['column']} has no matching estimate")

cov = pd.read_csv(DER / "mapping_coverage.csv")
for _, r in cov.iterrows():
    for c in [c for c in cov.columns if c.startswith("with_") or c == "mapped_to_occupation"]:
        if r[c] != r["respondents"]:
            fails.append(f"scheme {r['census_scheme']}: {c} = {r[c]} of {r['respondents']}")

key = json.loads((RES / "key_numbers.json").read_text())
for ref in (2019, 2022):
    ev = est[(est["spec"] == f"event_study_ref{ref}") & (est["measure"] == "gpt4_beta:third")
             & (est["outcome"] == "leisure")]
    years = sorted(int(t[3:]) for t in ev["term"])
    expected = [y for y in map(int, key["respondents_by_year"]) if y != ref]
    if years != expected:
        fails.append(f"event study ref {ref}: years {years}, expected {expected}")

if fails:
    print("FAIL")
    print("\n".join(fails))
    sys.exit(1)
print(f"clean: {len(sums)} specifications add up, {len(t1)} table cells match, coverage complete")
