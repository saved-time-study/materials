"""Reproduce the ChatGPT event study of Jiang, Park, Xiao and Zhang, "AI and the Extended Workday" (NBER w33536,
February 2025, section 4.1 and Table 2, column 1), then extend it to 2024 and 2025 and to other baseline years.

Their specification, from the text of the paper, and what we do where the text leaves a choice:
- Sample: ATUS respondents aged 16 to 65, without full-time students under 25 and people in the military
  industries (Census industry 9590 and 9600-9900) or the tech industries (6470-6780, 7380-7460, 7490); employed or
  unemployed (TELFS 1 to 4), with an occupation code. For the unemployed, whose ATUS occupation is missing, they take
  the occupation and industry from the respondent's CPS interview; we do the same (PEIO1OCD, PEIO1ICD, PRDTIND1 of
  the respondent's line in the ATUS-CPS file).
- Outcomes: weekly hours (daily minutes / 60 * 7, capped at 168, winsorized at the 1st and 99th percentiles of this
  script's sample, 2015-2025 without 2020, after every exclusion). Work: main job (050101, 050103, 050189) and work-related activities other than socializing at
  work (050202, 050203, 050204, 050289); other jobs (050102) and travel for work are excluded. Leisure: socializing,
  relaxing and leisure (12), sports and recreation (13), travel for those activities (1812, 1813), and the time spent
  sleeping, grooming, on personal activities and eating (0101, 0102, 0104, 0105, 0199, 11) above 7 hours a day.
- Exposure: their generative AI exposure is their own GPT-4o labeling of O*NET 27.0 tasks with a rubric adapted
  from Eloundou et al. (a task score of 1 for direct exposure and 0.5 for exposure through LLM-powered software,
  though one sentence of their appendix describes the occupation score as the share of tasks with either kind),
  as a percentile rank (1 to 100) across occupations. Their labels are not published with the paper; we use the
  GPT-4 rating published by Eloundou et al. (beta, E1 + 0.5 * E2), ranked across the harmonized occupations present
  in this script's sample (370).
  The interquartile shift is 50 percentile points, as in their text.
- Occupation unit: they use David Dorn's occ1990dd; we use our harmonized occupations (03_exposure.py).
- Model: weighted least squares (TUFNWGTP) of weekly hours on exposure percentile x POST, with controls for age,
  number of children under 18, and indicators for sex, education, marital status and race, and fixed effects for
  occupation, state x year, industry x year (TRDTIND1, 51 industries), year x month and day of the week. Standard
  errors clustered two ways, by occupation and by state.

Run after 03_exposure.py:  python -I data/atus/scripts/07_jpxz_replication.py
Writes results/jpxz_estimates.csv, results/jpxz_key_numbers.json and results/figureS4_jpxz_event_study.png.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fe import fe_wls, wald  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, DER, RES = ROOT / "raw", ROOT / "derived", ROOT / "results"
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
WORK = ["050101", "050103", "050189", "050202", "050203", "050204", "050289"]
ESP = ["0101", "0102", "0104", "0105", "0199", "11"]
LEIS = ["12", "13", "1812", "1813"]
MEASURES = ["gpt4_beta", "human_beta", "lm_aioe"]
IQR = 50


def cols_for(columns, prefixes):
    return [c for c in columns if c.startswith("t") and c[1:].isdigit() and any(c[1:].startswith(p) for p in prefixes)]


def build():
    s = pd.read_csv(RAW / "atussum-0325" / "atussum_0325.dat")
    s = s[s["TUYEAR"].isin(YEARS)].copy()
    tc = [c for c in s.columns if c.startswith("t") and c[1:].isdigit()]
    work = s[cols_for(tc, WORK)].sum(axis=1)
    esp = s[cols_for(tc, ESP)].sum(axis=1)
    leis = s[cols_for(tc, LEIS)].sum(axis=1) + np.maximum(esp - 420, 0)
    d = s[["TUCASEID", "TUYEAR", "TEAGE", "TESEX", "PEEDUCA", "PTDTRACE", "TELFS", "TRCHILDNUM", "TUDIARYDAY",
           "TUFNWGTP"]].copy()
    d["work_wk"] = np.minimum(work / 60 * 7, 168)
    d["leisure_wk"] = np.minimum(leis / 60 * 7, 168)

    r = pd.read_csv(RAW / "atusresp-0325" / "atusresp_0325.dat",
                    usecols=["TUCASEID", "TEIO1OCD", "TEIO1ICD", "TRDTIND1", "TUMONTH", "TESCHENR", "TESCHFT"])
    d = d.merge(r, on="TUCASEID", how="left", validate="1:1")
    keep = set(d["TUCASEID"])
    parts = []
    for ch in pd.read_csv(RAW / "atuscps-0325" / "atuscps_0325.dat", chunksize=200_000,
                          usecols=["TUCASEID", "TULINENO", "GESTFIPS", "PEMARITL", "PEIO1OCD", "PEIO1ICD", "PRDTIND1"]):
        ch = ch[(ch["TULINENO"] == 1) & ch["TUCASEID"].isin(keep)]
        parts.append(ch.drop(columns="TULINENO"))
    cps = pd.concat(parts).drop_duplicates("TUCASEID")
    d = d.merge(cps, on="TUCASEID", how="left", validate="1:1")

    emp = d["TELFS"].isin([1, 2])
    d["occ"] = np.where(emp, d["TEIO1OCD"], d["PEIO1OCD"])
    d["ind"] = np.where(emp, d["TEIO1ICD"], d["PEIO1ICD"])
    d["ind_recode"] = np.where(emp, d["TRDTIND1"], d["PRDTIND1"])
    steps = [("diaries 2015-2025 without 2020", len(d))]
    d = d[(d["TEAGE"] >= 16) & (d["TEAGE"] <= 65)]
    steps.append(("aged 16 to 65", len(d)))
    d = d[d["TELFS"].isin([1, 2, 3, 4])]
    steps.append(("employed or unemployed", len(d)))
    d = d[~((d["TEAGE"] < 25) & (d["TESCHENR"] == 1) & (d["TESCHFT"] == 1))]
    steps.append(("without full-time students under 25", len(d)))
    d = d[d["occ"] > 0]
    steps.append(("with an occupation code", len(d)))
    ind = d["ind"]
    tech = ind.between(6470, 6780) | ind.between(7380, 7460) | (ind == 7490)
    mil = (ind == 9590) | ind.between(9600, 9900)
    d = d[~tech & ~mil]
    steps.append(("outside the tech and military industries", len(d)))

    cmap = pd.read_csv(DER / "census_code_map.csv")
    hx = pd.read_csv(DER / "occupation_exposure.csv")
    vintage = pd.Series(np.where(d["TUYEAR"] < 2020, "2010", "2018"), index=d.index)
    d["key"] = vintage + ":" + d["occ"].astype(int).astype(str).str.zfill(4)
    d = d.merge(cmap[["key", "hocc"]], on="key", how="left").merge(hx[["hocc"] + MEASURES], on="hocc", how="left")
    d = d.dropna(subset=["hocc", "gpt4_beta", "GESTFIPS"])
    d = d[d["ind_recode"] > 0]
    steps.append(("with an exposure score, a state and an industry", len(d)))
    for m in MEASURES:
        occ = d.drop_duplicates("hocc").set_index("hocc")[m]
        d["pct_" + m] = d["hocc"].map(occ.rank(pct=True) * 100)
    for v in ["work_wk", "leisure_wk"]:
        lo, hi = d[v].quantile([0.01, 0.99])
        d[v] = d[v].clip(lo, hi)
    d["female"] = (d["TESEX"] == 2).astype(float)
    d["educ5"] = pd.cut(d["PEEDUCA"], [0, 38, 39, 42, 43, 99], labels=["lths", "hs", "some", "ba", "grad"]).astype(str)
    d["race4"] = d["PTDTRACE"].map({1: "white", 2: "black", 4: "asian"}).fillna("other")
    d["marital"] = d["PEMARITL"].astype(int).astype(str)
    dum = pd.get_dummies(d[["educ5", "race4", "marital"]], drop_first=True).astype(float)
    d = pd.concat([d.reset_index(drop=True), dum.reset_index(drop=True)], axis=1)
    d["state_year"] = d["GESTFIPS"].astype(int).astype(str) + "_" + d["TUYEAR"].astype(str)
    d["ind_year"] = d["ind_recode"].astype(int).astype(str) + "_" + d["TUYEAR"].astype(str)
    d["year_month"] = d["TUYEAR"].astype(str) + "_" + d["TUMONTH"].astype(str)
    controls = ["TEAGE", "TRCHILDNUM", "female"] + list(dum.columns)
    return d, controls, steps


FES = ["hocc", "state_year", "ind_year", "year_month", "TUDIARYDAY"]
ROWS = []


def run(d, controls, spec, years, terms, measure="gpt4_beta"):
    s = d[d["TUYEAR"].isin(years)].copy()
    xs = []
    for name, yrs in terms.items():
        s[name] = s["pct_" + measure] * s["TUYEAR"].isin(yrs)
        xs.append(name)
    out = {}
    for o in ["work_wk", "leisure_wk"]:
        res = fe_wls(s, o, xs + controls, fe=FES, w="TUFNWGTP", cluster=["hocc", "GESTFIPS"])
        out[o] = res
        for t in xs:
            r = res.loc[t]
            ROWS.append({"spec": spec, "measure": measure, "outcome": o, "term": t, "coef": r["coef"], "se": r["se"],
                         "t": r["t"], "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "iqr_hours_week": IQR * r["coef"], "iqr_ci_lo": IQR * r["ci_lo"],
                         "iqr_ci_hi": IQR * r["ci_hi"], "iqr_minutes_day": IQR * r["coef"] * 60 / 7,
                         "n": res.attrs["n"], "occupations": res.attrs["cluster_counts"][0],
                         "states": res.attrs["cluster_counts"][1]})
    return out


def main():
    d, controls, steps = build()
    # 1. Their years and their POST: 2022 against 2023
    for m in MEASURES:
        run(d, controls, "replication_2022_2023", [2022, 2023], {"post": [2023]}, measure=m)
    # 2. Add 2024 and 2025: one POST term, and one term for each year after 2022
    run(d, controls, "extended_2022_2025_one_post", [2022, 2023, 2024, 2025], {"post": [2023, 2024, 2025]})
    run(d, controls, "extended_2022_2025_by_year", [2022, 2023, 2024, 2025],
        {"y2023": [2023], "y2024": [2024], "y2025": [2025]})
    # 3. The same model against the years before the pandemic: 2015-2019 against 2024-2025
    run(d, controls, "prepandemic_2015_2019_vs_2024_2025", [2015, 2016, 2017, 2018, 2019, 2024, 2025],
        {"post": [2024, 2025]})
    run(d, controls, "prepandemic_2015_2019_vs_2023", [2015, 2016, 2017, 2018, 2019, 2023], {"post": [2023]})
    # 4. Event studies in their model, 2015-2025, against 2022 and against 2019
    tests = []
    for ref in (2022, 2019):
        terms = {f"y{y}": [y] for y in YEARS if y != ref}
        res = run(d, controls, f"event_study_ref{ref}", YEARS, terms)
        for o, r in res.items():
            pre = [f"y{y}" for y in (2015, 2016, 2017, 2018) if y != ref]
            F, p = wald(r, pre)
            tests.append({"reference_year": ref, "outcome": o, "years_tested_jointly_zero": "2015-2018", "F": F,
                          "p": p})

    est = pd.DataFrame(ROWS)
    est.to_csv(RES / "jpxz_estimates.csv", index=False, float_format="%.10g", lineterminator="\n")
    key = {
        "sample_steps": steps,
        "respondents_by_year": {int(k): int(v) for k, v in d.groupby("TUYEAR").size().items()},
        "replication_n_2022_2023": int(d["TUYEAR"].isin([2022, 2023]).sum()),
        "paper_table2_col1": {"work_coef": 0.063, "work_t": 2.12, "leisure_coef": -0.064, "leisure_t": -1.92,
                              "n_work": 8094, "n_leisure": 8335, "iqr_work_hours_week": 3.15,
                              "iqr_leisure_hours_week": -3.20},
        "mean_hours_week_2022": {o: float(np.average(d.loc[d["TUYEAR"] == 2022, o],
                                                     weights=d.loc[d["TUYEAR"] == 2022, "TUFNWGTP"]))
                                 for o in ["work_wk", "leisure_wk"]},
        "pretrend_tests": tests,
    }
    (RES / "jpxz_key_numbers.json").write_text(json.dumps(key, indent=2) + "\n", newline="\n")
    figure(est)
    show = est[est["spec"] != "event_study_ref2019"]
    show = show[~show["spec"].str.startswith("event") | show["term"].isin(["y2019", "y2023", "y2024", "y2025"])]
    print(pd.DataFrame(steps))
    print(show[["spec", "measure", "outcome", "term", "coef", "t", "iqr_hours_week", "iqr_ci_lo", "iqr_ci_hi", "n"]]
          .round(3).to_string())
    print(pd.DataFrame(tests).round(3).to_string())


def figure(est):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True)
    for ax, o, title in zip(axes, ["work_wk", "leisure_wk"], ["Work", "Leisure"]):
        for ref, color, off in [(2022, "#c8553d", -0.12), (2019, "#2b2b2b", 0.12)]:
            s = est[(est["spec"] == f"event_study_ref{ref}") & (est["outcome"] == o)].copy()
            s["year"] = s["term"].str[1:].astype(int)
            s = pd.concat([s, pd.DataFrame([{"year": ref, "iqr_hours_week": 0, "iqr_ci_lo": 0, "iqr_ci_hi": 0}])])
            s = s.sort_values("year")
            x = [YEARS.index(y) + off for y in s["year"]]
            ax.errorbar(x, s["iqr_hours_week"], yerr=[s["iqr_hours_week"] - s["iqr_ci_lo"],
                                                       s["iqr_ci_hi"] - s["iqr_hours_week"]],
                        fmt="o", ms=3.5, color=color, elinewidth=0.9, label=f"against {ref}")
        ax.axhline(0, color="#9a9a9a", lw=0.7)
        ax.axvspan(YEARS.index(2022) + 0.5, len(YEARS) - 0.5, color="#c8553d", alpha=0.07, lw=0)
        ax.set_xticks(range(len(YEARS)))
        ax.set_xticklabels([str(y)[2:] for y in YEARS])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel("Diary year")
    axes[0].set_ylabel("Weekly hours, 75th minus 25th\npercentile of exposure")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, ["Reference year 2022 (their baseline)", "Reference year 2019"], frameon=False, fontsize=7.5,
               loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RES / "figureS4_jpxz_event_study.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
