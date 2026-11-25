"""Every estimate of the ATUS analysis. Writes data/atus/results/estimates.csv (one row per coefficient),
results/pretrend_tests.csv, results/descriptives.csv and results/key_numbers.json.

Design. Employed respondents aged 18 to 64, 2015-2025 without 2020. Weighted least squares with the ATUS
final weight (TUFNWGTP, rescaled to mean 1 within each year so every year counts by its number of diaries),
harmonized-occupation fixed effects, year fixed effects, and controls for age, age squared, sex, education
(four groups), day of the week, holiday, month of the diary, and whether the person worked at home that day.
Standard errors are clustered by harmonized occupation.

Main difference in differences: diaries from 2019, 2021 and 2022 (before) against 2024 and 2025 (after),
with 2023 given its own term. Exposure enters as (a) the continuous score, standardized to mean 0 and
standard deviation 1 across workers, or (b) top third against bottom third of workers (middle third dropped).
Event study: exposure times each year, 2022 the reference year.

Run after 03_exposure.py:  python -I data/atus/scripts/04_analysis.py
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fe import fe_wls, wald  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
RES = ROOT / "results"
MEASURES = ["gpt4_beta", "human_beta", "lm_aioe", "aioe"]
MAIN_OUT = ["leisure", "paid_work", "household", "personal_care", "rest"]
PARTS = ["leis_tv", "leis_socializing", "leis_sports", "leis_computer_games", "leis_relaxing", "leis_other"]
ALL_OUT = MAIN_OUT + PARTS
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
REF = 2022
MDE_FACTOR = 2.80  # (1.96 + 0.84) standard errors: 80 percent power, two-sided 5 percent test


def load():
    d = pd.read_parquet(DER / "sample.parquet")
    cmap = pd.read_csv(DER / "census_code_map.csv")
    hx = pd.read_csv(DER / "occupation_exposure.csv")
    d["key"] = d["occ_vintage"].astype(str) + ":" + d["TEIO1OCD"].astype(int).astype(str).str.zfill(4)
    d = d.merge(cmap[["key", "hocc"]], on="key", how="left").merge(hx[["hocc"] + MEASURES], on="hocc")
    d["w"] = d["weight"] / d.groupby("TUYEAR")["weight"].transform("mean")
    for m in MEASURES:
        mu = np.average(d[m], weights=d["w"])
        sd = np.sqrt(np.average((d[m] - mu) ** 2, weights=d["w"]))
        d["z_" + m] = (d[m] - mu) / sd
        # thirds of workers, by weighted rank of the occupation's score
        o = d.sort_values(m)
        cum = o["w"].cumsum() / o["w"].sum()
        cut1 = o.loc[cum >= 1 / 3, m].iloc[0]
        cut2 = o.loc[cum >= 2 / 3, m].iloc[0]
        d["third_" + m] = np.where(d[m] < cut1, 1, np.where(d[m] < cut2, 2, 3))
    d["age2"] = d["TEAGE"] ** 2 / 100
    ctrl = pd.get_dummies(d[["educ"]].astype(str), drop_first=True).astype(float)
    ctrl = ctrl.join(pd.get_dummies(d["TUDIARYDAY"], prefix="dow", drop_first=True).astype(float))
    ctrl = ctrl.join(pd.get_dummies(d["TUMONTH"], prefix="mon", drop_first=True).astype(float))
    yrs = pd.get_dummies(d["TUYEAR"], prefix="yr").astype(float)
    d = pd.concat([d, ctrl, yrs], axis=1)
    controls = ["TEAGE", "age2", "female", "holiday"] + list(ctrl.columns)
    return d, controls


ROWS = []


def record(res, spec, measure, outcome, sample, terms, base=None):
    for t in terms:
        if t not in res.index:
            continue
        r = res.loc[t]
        ROWS.append({"spec": spec, "measure": measure, "outcome": outcome, "sample": sample, "term": t,
                     "coef": r["coef"], "se": r["se"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                     "mde": MDE_FACTOR * r["se"], "n": res.attrs["n"], "clusters": res.attrs["clusters"],
                     "pre_mean": base})


def did(d, controls, outcome, measure, kind, spec, sample_name, mask=None, pre=(2019, 2021, 2022),
        post=(2024, 2025), extra=(2023,), wfh=True):
    """One difference in differences. kind 'z' = per standard deviation; 'third' = top vs bottom third."""
    s = d[d["TUYEAR"].isin(list(pre) + list(post) + list(extra))]
    if mask is not None:
        s = s[mask.loc[s.index]]
    if kind == "third":
        s = s[s["third_" + measure].isin([1, 3])].copy()
        expo = (s["third_" + measure] == 3).astype(float)
    else:
        s = s.copy()
        expo = s["z_" + measure]
    s["x_post"] = expo * s["TUYEAR"].isin(post)
    xs = ["x_post"]
    for y in extra:
        s[f"x_{y}"] = expo * (s["TUYEAR"] == y)
        xs.append(f"x_{y}")
    yrs = [f"yr_{y}" for y in sorted(s["TUYEAR"].unique())][1:]
    ctl = controls + (["wfh_day"] if wfh else [])
    res = fe_wls(s, outcome, xs + yrs + ctl, fe="hocc", w="w", cluster="hocc")
    base = np.average(s.loc[s["TUYEAR"].isin(pre), outcome], weights=s.loc[s["TUYEAR"].isin(pre), "w"])
    record(res, spec, f"{measure}:{kind}", outcome, sample_name, xs, base)
    return res


def event(d, controls, outcome, measure, kind, ref):
    s = d.copy()
    if kind == "third":
        s = s[s["third_" + measure].isin([1, 3])].copy()
        expo = (s["third_" + measure] == 3).astype(float)
    else:
        expo = s["z_" + measure]
    xs = []
    for y in YEARS:
        if y == ref:
            continue
        s[f"ev_{y}"] = expo * (s["TUYEAR"] == y)
        xs.append(f"ev_{y}")
    yrs = [f"yr_{y}" for y in YEARS[1:]]
    res = fe_wls(s, outcome, xs + yrs + controls + ["wfh_day"], fe="hocc", w="w", cluster="hocc")
    record(res, f"event_study_ref{ref}", f"{measure}:{kind}", outcome, "all", xs)
    out = []
    groups = {"2015-2018": [f"ev_{y}" for y in (2015, 2016, 2017, 2018) if y != ref],
              "2015-2021 except reference": [f"ev_{y}" for y in YEARS if y < REF and y != ref],
              "2021-2022 except reference": [f"ev_{y}" for y in (2021, 2022) if y != ref],
              "2024-2025": ["ev_2024", "ev_2025"]}
    for label, terms in groups.items():
        F, p = wald(res, terms)
        out.append({"reference_year": ref, "measure": f"{measure}:{kind}", "outcome": outcome,
                    "years_tested_jointly_zero": label, "F": F, "p": p, "q": len(terms),
                    "clusters": res.attrs["clusters"]})
    return out


def main():
    RES.mkdir(exist_ok=True)
    d, controls = load()
    tests = []

    # 1. Main difference in differences, all measures, both forms, every outcome
    for m in MEASURES:
        for kind in ["z", "third"]:
            for o in ALL_OUT + ["work_main_other_jobs", "wfh_day"]:
                did(d, controls, o, m, kind, "main", "all", wfh=(o != "wfh_day"))

    # 2. Sensitivity: no work-at-home control; longer before period (2015-2022 without 2020)
    for kind in ["z", "third"]:
        for o in MAIN_OUT:
            did(d, controls, o, "gpt4_beta", kind, "no_wfh_control", "all", wfh=False)
            did(d, controls, o, "gpt4_beta", kind, "long_pre", "all",
                pre=(2015, 2016, 2017, 2018, 2019, 2021, 2022))

    # 3. Subsamples: weekdays, weekends; worked at home, worked only away from home, did not work
    subs = {
        "weekday": d["weekend"] == 0,
        "weekend": d["weekend"] == 1,
        "worked_at_home": d["wfh_day"] == 1,
        "worked_only_away": (d["work_day"] == 1) & (d["wfh_day"] == 0),
        "did_not_work": d["work_day"] == 0,
    }
    for name, mask in subs.items():
        for kind in ["z", "third"]:
            for o in ALL_OUT:
                did(d, controls, o, "gpt4_beta", kind, "subsample", name, mask=mask,
                    wfh=name in ("weekday", "weekend"))

    # 4. Placebo dates before ChatGPT (continuous GPT-4 score and top vs bottom third)
    placebos = {
        "placebo_2018_19_vs_2015_16": dict(pre=(2015, 2016), post=(2018, 2019), extra=()),
        "placebo_2019_vs_2015_18": dict(pre=(2015, 2016, 2017, 2018), post=(2019,), extra=()),
        "placebo_2022_vs_2021": dict(pre=(2021,), post=(2022,), extra=()),
    }
    for name, kw in placebos.items():
        for kind in ["z", "third"]:
            for o in MAIN_OUT + PARTS:
                did(d, controls, o, "gpt4_beta", kind, name, "all", **kw)

    # 2b. Pre-pandemic baseline: 2015-2019 against 2024-2025, with 2021, 2022 and 2023 each on their own
    for m in MEASURES:
        for kind in ["z", "third"]:
            for o in ALL_OUT + ["wfh_day"]:
                did(d, controls, o, m, kind, "prepandemic_baseline", "all",
                    pre=(2015, 2016, 2017, 2018, 2019), extra=(2021, 2022, 2023), wfh=(o != "wfh_day"))
    for name, mask in {"weekday": d["weekend"] == 0, "weekend": d["weekend"] == 1}.items():
        for kind in ["z", "third"]:
            for o in MAIN_OUT:
                did(d, controls, o, "gpt4_beta", kind, "prepandemic_baseline", name, mask=mask,
                    pre=(2015, 2016, 2017, 2018, 2019), extra=(2021, 2022, 2023))

    # 5. Event study, 2015-2025, with 2019 (last year before the pandemic) and 2022 (last year before
    #    ChatGPT) as reference years
    for ref in (2019, REF):
        for m in MEASURES:
            for kind in ["z", "third"]:
                outs = ALL_OUT if m == "gpt4_beta" else MAIN_OUT
                for o in outs:
                    tests.extend(event(d, controls, o, m, kind, ref))

    est = pd.DataFrame(ROWS)
    # Holm adjustment across the four outcomes that are free to move (leisure, paid work, household,
    # personal care; "rest" is fixed by the other four because the day adds up to 1,440 minutes)
    fam = est["outcome"].isin(MAIN_OUT[:4])
    est["p_holm_4"] = np.nan
    for _, idx in est[fam].groupby(["spec", "measure", "sample", "term"]).groups.items():
        p = est.loc[idx, "p"].sort_values()
        adj = np.minimum(1, np.maximum.accumulate(p.to_numpy() * (len(p) - np.arange(len(p)))))
        est.loc[p.index, "p_holm_4"] = adj
    est.to_csv(RES / "estimates.csv", index=False, float_format="%.10g", lineterminator="\n")
    pd.DataFrame(tests).to_csv(RES / "pretrend_tests.csv", index=False, float_format="%.6g", lineterminator="\n")

    # Descriptives: weighted means by third of GPT-4 exposure and period
    d["period"] = np.select([d["TUYEAR"].isin([2019, 2021, 2022]), d["TUYEAR"].isin([2024, 2025]),
                             d["TUYEAR"] == 2023], ["before", "after", "2023"], "2015-2018")
    desc = []
    for (third, period), g in d.groupby(["third_gpt4_beta", "period"]):
        row = {"third": int(third), "period": period, "respondents": len(g),
               "occupations": g["hocc"].nunique()}
        for o in ALL_OUT + ["wfh_day", "work_day"]:
            row[o] = np.average(g[o], weights=g["w"])
        desc.append(row)
    desc = pd.DataFrame(desc)
    desc.to_csv(RES / "descriptives.csv", index=False, float_format="%.6g", lineterminator="\n")
    yearly = []
    for (yr, third), g in d.groupby(["TUYEAR", "third_gpt4_beta"]):
        yearly.append({"year": yr, "third": int(third), "respondents": len(g),
                       **{o: np.average(g[o], weights=g["w"]) for o in MAIN_OUT + ["wfh_day"]}})
    pd.DataFrame(yearly).to_csv(RES / "yearly_means_by_third.csv", index=False, float_format="%.6g",
                                lineterminator="\n")

    # Key numbers
    main = est[(est["spec"] == "main") & (est["sample"] == "all") & (est["term"] == "x_post")]
    adding = {k: float(v) for k, v in main[main["outcome"].isin(MAIN_OUT)].groupby("measure")["coef"].sum().items()}
    third_cut = {m: [float(d.loc[d["third_" + m] == t, m].min()) for t in (2, 3)] for m in MEASURES}
    key = {
        "respondents_total": int(len(d)),
        "respondents_by_year": {int(k): int(v) for k, v in d.groupby("TUYEAR").size().items()},
        "occupations": int(d["hocc"].nunique()),
        "main_did_sample": int(d["TUYEAR"].isin([2019, 2021, 2022, 2023, 2024, 2025]).sum()),
        "third_cutpoints": third_cut,
        "sd_of_measure": {m: float(np.sqrt(np.cov(d[m], aweights=d["w"]))) for m in MEASURES},
        "adding_up_check_sum_of_main_coefs": adding,
        "share_worked_at_home_by_third_before_after": desc.set_index(["third", "period"])["wfh_day"]
        .round(4).to_dict().__repr__(),
    }
    (RES / "key_numbers.json").write_text(json.dumps(key, indent=2) + "\n", newline="\n")
    print(main[main["outcome"].isin(MAIN_OUT)][["measure", "outcome", "coef", "ci_lo", "ci_hi", "p", "mde"]]
          .round(2).to_string())
    t = pd.DataFrame(tests)
    print(t[t["measure"].str.startswith("gpt4") & t["outcome"].isin(MAIN_OUT)].round(3).to_string())
    pp = est[(est["spec"] == "prepandemic_baseline") & (est["sample"] == "all") & (est["term"] == "x_post")]
    print(pp[pp["outcome"].isin(MAIN_OUT)][["measure", "outcome", "coef", "ci_lo", "ci_hi", "p", "mde"]]
          .round(2).to_string())
    print(json.dumps(key, indent=1)[:1500])


if __name__ == "__main__":
    main()
