"""Build the analysis sample from the ATUS 2003-2025 multi-year files.

One row per respondent: employed (TELFS 1 or 2), aged 18 to 64, diary years 2015 to 2025 except 2020.
Writes data/atus/derived/sample.parquet (not committed; rebuilt from raw) and
data/atus/derived/sample_counts.csv (committed).

Run from the repository root:  python -I data/atus/scripts/02_build_sample.py
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "derived"
YEARS = [y for y in range(2015, 2026) if y != 2020]

# Outcomes: minutes on the diary day, built from the six-digit activity columns of the summary file.
# Each outcome is a list of code prefixes; a column t<code> counts when its code starts with a prefix.
LEISURE_PARTS = {
    "tv": ["120303", "120304"],                       # television and movies
    "socializing": ["1201", "1202", "120501", "120502"],  # socializing, communicating, social events
    "sports": ["13"],                                   # sports, exercise and recreation
    "computer_games": ["120307", "120308"],           # playing games; computer use for leisure
    "relaxing": ["120301"],                             # relaxing, thinking
}
MAIN = {
    "personal_care": ["01"],   # includes sleep
    "household": ["02"],
    "paid_work": ["05"],       # work and work-related activities
    "leisure": ["12", "13"],   # socializing, relaxing and leisure; sports, exercise and recreation
}


def minutes(df, cols, prefixes):
    keep = [c for c in cols if any(c[1:].startswith(p) for p in prefixes)]
    return df[keep].sum(axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RAW / "atussum-0325" / "atussum_0325.dat")
    s = s[s["TUYEAR"].isin(YEARS)].copy()
    tcols = [c for c in s.columns if c.startswith("t") and c[1:].isdigit()]
    total = s[tcols].sum(axis=1)
    assert (total == 1440).all(), "every diary must add up to 1,440 minutes"

    out = s[["TUCASEID", "TUYEAR", "TEAGE", "TESEX", "PEEDUCA", "TELFS", "TUDIARYDAY", "TRHOLIDAY",
             "TUFNWGTP", "TU20FWGT"]].copy()
    for name, pref in MAIN.items():
        out[name] = minutes(s, tcols, pref)
    out["rest"] = 1440 - out[list(MAIN)].sum(axis=1)
    for name, pref in LEISURE_PARTS.items():
        out["leis_" + name] = minutes(s, tcols, pref)
    out["leis_other"] = out["leisure"] - out[["leis_" + k for k in LEISURE_PARTS]].sum(axis=1)
    out["work_main_other_jobs"] = minutes(s, tcols, ["0501"])

    r = pd.read_csv(RAW / "atusresp-0325" / "atusresp_0325.dat",
                    usecols=["TUCASEID", "TEIO1OCD", "TUMONTH", "TRDTOCC1"])
    out = out.merge(r, on="TUCASEID", how="left", validate="1:1")

    # Worked at home on the diary day: any minutes of work at a main or other job (0501xx)
    # whose location was the respondent's home or yard (TEWHERE = 1).
    a = pd.read_csv(RAW / "atusact-0325" / "atusact_0325.dat",
                    usecols=["TUCASEID", "TUACTDUR24", "TRCODEP", "TEWHERE"])
    a = a[a["TUCASEID"].isin(out["TUCASEID"])]
    a["code"] = a["TRCODEP"].astype(str).str.zfill(6)
    job = a[a["code"].str.startswith("0501")]
    home = job[job["TEWHERE"] == 1].groupby("TUCASEID")["TUACTDUR24"].sum().rename("work_home_min")
    out = out.merge(home, on="TUCASEID", how="left")
    out["work_home_min"] = out["work_home_min"].fillna(0)
    out["wfh_day"] = (out["work_home_min"] > 0).astype(int)
    out["work_day"] = (out["work_main_other_jobs"] > 0).astype(int)

    counts = [("diaries 2015-2025 without 2020", len(out))]
    out = out[out["TELFS"].isin([1, 2])]
    counts.append(("employed (TELFS 1 or 2)", len(out)))
    out = out[(out["TEAGE"] >= 18) & (out["TEAGE"] <= 64)]
    counts.append(("aged 18 to 64", len(out)))
    out = out[out["TEIO1OCD"] > 0]
    counts.append(("with an occupation code", len(out)))
    out["weight"] = out["TUFNWGTP"]
    out = out[out["weight"] > 0]
    counts.append(("with a positive final weight", len(out)))

    # Census occupation vintage: 2010 codes for diaries before 2020, 2018 codes from 2020 on.
    out["occ_vintage"] = np.where(out["TUYEAR"] < 2020, 2010, 2018)
    out["female"] = (out["TESEX"] == 2).astype(int)
    out["educ"] = pd.cut(out["PEEDUCA"], [0, 38, 39, 42, 99],
                         labels=["less_than_hs", "hs", "some_college", "ba_plus"]).astype(str)
    out["weekend"] = out["TUDIARYDAY"].isin([1, 7]).astype(int)
    out["holiday"] = (out["TRHOLIDAY"] == 1).astype(int)

    out.to_parquet(OUT / "sample.parquet", index=False)
    pd.DataFrame(counts, columns=["step", "respondents"]).to_csv(OUT / "sample_counts.csv", index=False,
                                                                 lineterminator="\n")
    by_year = out.groupby("TUYEAR").size().rename("respondents")
    by_year.to_csv(OUT / "sample_by_year.csv", lineterminator="\n")
    print(pd.DataFrame(counts))
    print(by_year)


if __name__ == "__main__":
    main()
