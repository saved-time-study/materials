"""Exposure to generative AI for every ATUS occupation, on one harmonized occupation scheme.

ATUS codes occupations with the 2010 Census scheme before 2020 and the 2018 Census scheme from 2020 on.
Each Census code is mapped to its SOC codes (2010 SOC for 2010 Census codes, 2018 SOC for 2018 codes).
A 2010 Census code and a 2018 Census code are linked when their SOC codes overlap after the BLS 2010-to-2018
SOC crosswalk; each connected group of linked codes is one harmonized occupation ("hocc"). Every respondent in
a harmonized occupation gets the same exposure score, so a change of coding scheme in 2020 cannot move a
person's exposure.

Exposure measures, each the mean over the SOC codes of the occupation:
  gpt4_beta   Eloundou, Manning, Mishkin and Rock (2024), GPT-4 rating, beta = E1 + 0.5 * E2 (primary)
  human_beta  the same paper's human rating, beta
  lm_aioe     Felten, Raj and Seamans (2023), language-modeling AI occupational exposure
  aioe        Felten, Raj and Seamans (2021), AI occupational exposure
The harmonized occupation's score averages its 2018 Census codes, weighted by the ATUS final weights of the
2021-2025 respondents in each code (unweighted when a code has no respondents).

Run after 02_build_sample.py:  python -I data/atus/scripts/03_exposure.py
Writes data/atus/derived/occupation_exposure.csv, census_code_map.csv and mapping_coverage.csv.
"""
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "derived"
MEASURES = ["gpt4_beta", "human_beta", "lm_aioe", "aioe"]


def census_list(path, sheet):
    d = pd.read_excel(RAW / path, sheet_name=sheet, header=None, dtype=str).iloc[:, 1:4]
    d.columns = ["title", "code", "soc"]
    d = d.dropna(subset=["code", "soc"])
    d["code"] = d["code"].str.strip()
    d["soc"] = d["soc"].str.strip()
    d = d[d["code"].str.fullmatch(r"\d{4}")]
    return d.drop_duplicates("code").reset_index(drop=True)


def expand(pattern, universe):
    """SOC codes in `universe` that a Census SOC entry stands for (exact, X wildcard, or broad group)."""
    p = pattern.strip()
    if "X" in p:
        rx = re.compile("^" + p.replace("X", r"\d") + "$")
        return [u for u in universe if rx.match(u)]
    if p in universe:
        return [p]
    stem = p
    while stem.endswith("0") and len(stem) > 3:
        stem = stem[:-1]
        hits = [u for u in universe if u.startswith(stem)]
        if hits:
            return hits
    return []


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    c10 = census_list("2010-occ-codes-with-crosswalk-from-2002-2011.xls", "2010OccCodeList")
    c18 = census_list("2018-occupation-code-list-and-crosswalk.xlsx", "2018 Census Occ Code List")

    xw = pd.read_excel(RAW / "soc_2010_to_2018_crosswalk.xlsx", header=None, dtype=str).iloc[9:, [0, 2]]
    xw.columns = ["soc10", "soc18"]
    xw = xw.dropna().apply(lambda s: s.str.strip())
    xw = xw[xw["soc10"].str.fullmatch(r"\d{2}-\d{4}") & xw["soc18"].str.fullmatch(r"\d{2}-\d{4}")]
    u10, u18 = sorted(set(xw["soc10"])), sorted(set(xw["soc18"]))
    to18 = xw.groupby("soc10")["soc18"].apply(list).to_dict()

    # Scores on 2018 SOC
    g = pd.read_csv(RAW / "gpts_occ_level.csv")
    g["soc18"] = g["O*NET-SOC Code"].str[:7]
    s18 = g.groupby("soc18")[["dv_rating_beta", "human_rating_beta"]].mean()
    s18.columns = ["gpt4_beta", "human_beta"]
    lm = pd.read_excel(RAW / "Language_Modeling_AIOE_and_AIIE.xlsx", sheet_name="LM AIOE")
    lm = lm.rename(columns={"SOC Code": "soc10", "Language Modeling AIOE": "lm_aioe"})[["soc10", "lm_aioe"]]
    ai = pd.read_excel(RAW / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
    ai = ai.rename(columns={"SOC Code": "soc10", "AIOE": "aioe"})[["soc10", "aioe"]]
    s10 = lm.merge(ai, on="soc10", how="outer").set_index("soc10")
    felten18 = xw.merge(s10, left_on="soc10", right_index=True).groupby("soc18")[["lm_aioe", "aioe"]].mean()
    s18 = s18.join(felten18, how="outer")

    rows = []
    uf = UnionFind()
    for vint, table, univ in [(2010, c10, u10), (2018, c18, u18)]:
        for _, r in table.iterrows():
            socs = expand(r["soc"], univ)
            socs18 = sorted({t for s in socs for t in to18.get(s, [])}) if vint == 2010 else socs
            key = f"{vint}:{r['code']}"
            uf.find(key)
            for s in socs18:
                uf.union(key, "soc:" + s)
            sc = s18.reindex(socs18)[MEASURES].mean() if socs18 else pd.Series(np.nan, index=MEASURES)
            fallback = ""
            # "All other" codes (XX-XX99) have no O*NET rating; they take the mean of their SOC
            # broad group (first five characters), else of their minor group (first four).
            stem = (socs18[0] if socs18 else r["soc"].strip())
            for n in (5, 4):
                if sc.isna().any() and re.match(r"\d{2}-\d", stem):
                    near = s18[s18.index.str.startswith(stem[:n])][MEASURES].mean()
                    if near.notna().any():
                        sc = sc.fillna(near)
                        fallback = f"group {stem[:n]}"
            rows.append({"vintage": vint, "code": int(r["code"]), "title": r["title"], "soc_entry": r["soc"],
                         "n_soc2018": len(socs18), "score_from": fallback or "own SOC codes", **sc.to_dict()})
    cmap = pd.DataFrame(rows)
    cmap["key"] = cmap["vintage"].astype(str) + ":" + cmap["code"].astype(str).str.zfill(4)
    roots = cmap["key"].map(uf.find)
    cmap["hocc"] = pd.factorize(roots)[0] + 1

    smp = pd.read_parquet(OUT / "sample.parquet")
    smp["key"] = smp["occ_vintage"].astype(str) + ":" + smp["TEIO1OCD"].astype(int).astype(str).str.zfill(4)
    w18 = smp[smp["TUYEAR"] >= 2021].groupby("key")["weight"].sum()
    cmap["w"] = cmap["key"].map(w18).fillna(0)

    def comp_score(d):
        d18 = d[(d["vintage"] == 2018)]
        base = d18 if len(d18) else d
        out = {}
        for m in MEASURES:
            x = base.dropna(subset=[m])
            if x.empty:
                out[m] = np.nan
            elif x["w"].sum() > 0:
                out[m] = np.average(x[m], weights=x["w"])
            else:
                out[m] = x[m].mean()
        t = d18.sort_values("w", ascending=False)["title"]
        out["hocc_title"] = t.iloc[0] if len(t) else d["title"].iloc[0]
        out["n_codes_2010"] = int((d["vintage"] == 2010).sum())
        out["n_codes_2018"] = int((d["vintage"] == 2018).sum())
        return pd.Series(out)

    hx = cmap.groupby("hocc").apply(comp_score).reset_index()
    cmap.drop(columns=["w"]).to_csv(OUT / "census_code_map.csv", index=False, lineterminator="\n")
    hx.to_csv(OUT / "occupation_exposure.csv", index=False, lineterminator="\n")

    smp = smp.merge(cmap[["key", "hocc"]], on="key", how="left")
    smp = smp.merge(hx[["hocc"] + MEASURES], on="hocc", how="left")
    cov = []
    for vint in [2010, 2018]:
        d = smp[smp["occ_vintage"] == vint]
        cov.append({"census_scheme": vint, "respondents": len(d),
                    "mapped_to_occupation": int(d["hocc"].notna().sum()),
                    **{f"with_{m}": int(d[m].notna().sum()) for m in MEASURES}})
    cov = pd.DataFrame(cov)
    cov.to_csv(OUT / "mapping_coverage.csv", index=False, lineterminator="\n")
    sizes = cmap.groupby("hocc").size()
    print(cov.to_string())
    print("harmonized occupations:", cmap["hocc"].nunique(), "| largest has", sizes.max(), "Census codes")
    print("occupations in sample:", smp["hocc"].nunique())
    print(hx[MEASURES].corr().round(2))


if __name__ == "__main__":
    main()
