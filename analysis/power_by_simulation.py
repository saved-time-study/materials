"""Power by simulation: how often each reading of H1 to H5 occurs under each pattern of true effects.

Run from the repository's root (about ten minutes on one processor thread):
    python analysis/power_by_simulation.py
    python analysis/power_by_simulation.py --draws 20000 --curve-draws 10000 --sensitivity-draws 2000

Each simulated data set has the registered allocation (1,706 analyzable participants a group, groups filled evenly
within each of 12 countries); replacement of excluded participants is assumed to reach exactly that number, so only
analyzable participants are drawn. Every data set is analyzed with analyze.py's own functions: the outcome-neutral
criteria, the regression with country fixed effects and HC2 errors, the reading rules and the criterion gates. A
failed criterion makes its tests "not interpretable", which counts as a miss: the share with the expected reading is
the probability that the criteria pass and the test gives the reading.

Five parts:
1. Calibration: one very large data set for each pattern, to measure the standard deviations and differences that the
   simulator produces (so that the planted effects are shown in points, minutes and d).
2. The five patterns at --draws data sets each (20,000; the Monte Carlo interval is about plus or minus 0.003 near
   0.96), under both forms of the bound; the registered form is observed_sd. For each pattern: the share of data sets
   in which each criterion failed, the share in which the AI-saved group was less surprised than the found-break group
   by more than 0.15 pooled SD (a limit on reading H3 and H5, not a criterion), the share of each reading of H3 and H5
   as a pair, and, for each hypothesis, the share with each reading.
3. Power curves of each equivalence test: the share of data sets in which both one-sided tests are significant at
   .01, at true differences of 0, 0.05, 0.10 and 0.15 d (--curve-draws data sets each), beside the normal-theory value.
4. Sensitivity: the no-effect pattern when the true standard deviation of each outcome differs from the one behind
   the fixed bounds (29 points; 3.0 minutes).
5. The outcome-neutral criteria by normal theory: the true value at which each criterion fails with probability 0.05
   and 0.50 at the registered sample.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")  # one thread: nothing heavy on this computer

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy import optimize, stats

import analyze as an
import registered_constants as rc
import simulate as sim

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
CURVE_DIFFERENCES = (0.0, 0.05, 0.10, 0.15)  # true differences in d


def wilson(successes, n, level=0.95):
    return an.wilson(successes, n, level)


def claimed_power(hypothesis, expected):
    result = {v: k for k, v in rc.HYPOTHESES[hypothesis]["readings"].items()}[expected]
    return rc.CLAIMED_POWER_EQUIVALENCE if result == "equivalent" else rc.CLAIMED_POWER_DIFFERENCE


def verdict(lower, upper, level):
    if lower >= level:
        return f"{level:.2f} or more"
    if upper < level:
        return f"below {level:.2f}"
    return f"cannot be told apart from {level:.2f} at this number of draws"


def run_pattern(pattern, assumptions, draws, seed, batch=250, forms=rc.BOUND_FORMS):
    """Simulate `draws` data sets under `pattern` (a dict of simulate.PATTERNS) and count the readings of H1 to H5."""
    calibration = sim.calibrate(assumptions)
    group_index, country_index = sim.allocation()
    model = an.CountryFixedEffectsOLS(group_index, country_index)
    rng = np.random.default_rng(seed)
    counts = {form: {h: Counter() for h in rc.HYPOTHESES} for form in forms}
    equivalence_hits = {form: Counter() for form in forms}
    significant_and_equivalent = {form: Counter() for form in forms}
    estimate_sum, se_sum = Counter(), Counter()
    pairs = Counter()  # where AI-saved time lies: H3 and H5 read as a pair (registered_constants.PAIR_RULES)
    check_failures, any_failure, less_surprised = Counter(), 0, 0
    done = 0
    while done < draws:
        b = min(batch, draws - done)
        d = sim.draw_outcomes(rng, group_index, country_index, b, pattern, assumptions, calibration)
        checks = an.outcome_neutral_checks(model, d["fun_share"], d["paid_work_minutes"],
                                           (d["surprise_unexpected"] + d["surprise_surprising"]) / 2,
                                           d["says_ai"], d["says_not_reviewed"], d["saw_label"])
        failed_any = np.zeros(b, dtype=bool)
        for check, passed in checks["passed"].items():
            check_failures[check] += int((~passed).sum())
            failed_any |= ~passed
        any_failure += int(failed_any.sum())
        less_surprised += int(checks["surprise_ai_saved_minus_found"]["ai_saved_less_surprised"].sum())
        outcomes = {"fun_share": d["fun_share"], "paid_work_minutes": d["paid_work_minutes"]}
        for form in forms:
            tests = an.confirmatory_tests(model, outcomes, form)
            gated = an.apply_check_gates({h: t["reading"] for h, t in tests.items()}, checks["passed"])
            for h, t in tests.items():
                values, n = np.unique(gated[h].astype(str), return_counts=True)
                counts[form][h].update(dict(zip(values, n.tolist())))
                equivalence_hits[form][h] += int((t["p_equivalence"] < rc.ALPHA).sum())
                both = (t["p_two_sided"] < rc.ALPHA) & (t["p_equivalence"] < rc.ALPHA)
                significant_and_equivalent[form][h] += int(both.sum())
                if form == rc.BOUND_FORM:
                    estimate_sum[h] += float(t["estimate"].sum())
                    se_sum[h] += float(t["se"].sum())
            if form == rc.BOUND_FORM:
                values, n = np.unique(an.pair_reading(gated["H1"], gated["H3"], gated["H5"]).astype(str),
                                      return_counts=True)
                pairs.update(dict(zip(values, n.tolist())))
        done += b

    unknown = {r for form in forms for h in rc.HYPOTHESES for r in counts[form][h] if r not in rc.READINGS}
    assert not unknown, f"readings outside registered_constants.READINGS: {unknown}"
    result = {"draws": draws, "seed": seed,
              "criterion_failure_share": {c: check_failures[c] / draws for c in rc.CHECK_GATES},
              "any_criterion_failed_share": any_failure / draws,
              "ai_saved_less_surprised_than_found_share": less_surprised / draws,
              "mean_estimate": {h: estimate_sum[h] / draws for h in rc.HYPOTHESES},
              "mean_se": {h: se_sum[h] / draws for h in rc.HYPOTHESES},
              "pair_reading_share": {k: pairs[k] / draws for k in rc.PAIR_ORDER if pairs[k]},
              "by_bound_form": {}}
    assert set(pairs) <= set(rc.PAIR_ORDER), set(pairs) - set(rc.PAIR_ORDER)
    if pattern.get("expected_pair"):
        hits = pairs[pattern["expected_pair"]]
        result["expected_pair"] = pattern["expected_pair"]
        result["expected_pair_share"] = hits / draws
        result["expected_pair_interval"] = list(wilson(hits, draws))
    for form in forms:
        rows = {}
        for h in rc.HYPOTHESES:
            rows[h] = {"share_by_reading": {r: counts[form][h][r] / draws for r in rc.READINGS if counts[form][h][r]},
                       "equivalence_test_power": equivalence_hits[form][h] / draws,
                       "share_significant_and_equivalent": significant_and_equivalent[form][h] / draws}
            expected = pattern.get("expected", {}).get(h)
            if expected:
                hits = counts[form][h][expected]
                lo, hi = wilson(hits, draws)
                claim = pattern.get("power_claim", False)
                rows[h].update({
                    "expected_reading": expected, "share_expected": hits / draws, "interval": [lo, hi],
                    "claimed_in_table_1": claimed_power(h, expected) if claim else None,
                    "verdict_against_0_95": verdict(lo, hi, rc.REQUIRED_POWER) if claim else "not powered",
                    "verdict_against_0_96": verdict(lo, hi, rc.TARGET_POWER) if claim else "not powered"})
        result["by_bound_form"][form] = rows
    return result


def calibration_table(assumptions, scale=200, seed=1):
    """One data set 200 times the registered size for each pattern: standard deviations and planted differences."""
    cal = sim.calibrate(assumptions)
    group_index, country_index = sim.allocation(scale=scale)
    slots_mean = 1 + (rc.VIDEO_PICKS - 1) * assumptions.paid_work_extra_slot_probability
    out = {"calibration": asdict(cal), "participants_per_group": int(rc.PARTICIPANTS_PER_GROUP * scale),
           "intended_step": {"fun_share": cal.fun_share_step,
                             "paid_work_minutes": rc.SLOT_MINUTES * slots_mean * cal.paid_work_step,
                             "d": assumptions.effect_d}}
    rng = np.random.default_rng(seed)
    for name, pattern in sim.PATTERNS.items():
        d = sim.draw_outcomes(rng, group_index, country_index, 1, pattern, assumptions, cal)
        entry = {"within_country_sd": {}, "group_means": {}, "planted": {},
                 "share_left_early": float(d["left_early"].mean())}
        for outcome in rc.OUTCOMES:
            y = d[outcome][:, 0]
            sds, means = {}, {}
            for k, group in enumerate(rc.GROUPS):
                in_group = group_index == k
                residual = np.concatenate([y[in_group & (country_index == c)] - y[in_group & (country_index == c)].mean()
                                           for c in range(len(rc.COUNTRIES))])
                sds[group] = float(np.sqrt((residual ** 2).sum() / (len(residual) - len(rc.COUNTRIES))))
                means[group] = float(y[in_group].mean())
            entry["within_country_sd"][outcome] = sds
            entry["group_means"][outcome] = means
        for h, spec in rc.HYPOTHESES.items():
            o = spec["outcome"]
            diff = entry["group_means"][o][spec["minuend"]] - entry["group_means"][o][spec["subtrahend"]]
            pooled = np.sqrt((entry["within_country_sd"][o][spec["minuend"]] ** 2
                              + entry["within_country_sd"][o][spec["subtrahend"]] ** 2) / 2)
            entry["planted"][h] = {"difference": diff, "unit": rc.OUTCOME_UNITS[o], "d": float(diff / pooled)}
        out[name] = entry
    return out


def normal_theory_equivalence_power(true_d, bound_in_sd=rc.SMALLEST_EFFECT_D, n=rc.PARTICIPANTS_PER_GROUP):
    se = np.sqrt(2 / n)
    z = stats.norm.ppf(1 - rc.ALPHA)
    return float(max(0.0, stats.norm.cdf((bound_in_sd - true_d) / se - z)
                     + stats.norm.cdf((bound_in_sd + true_d) / se - z) - 1))


def curve_patterns(true_d):
    """Two configurations whose contrasts sit at true_d: A for H1, H2, H4 and H5 (H5 at minus true_d), B for H3."""
    s = true_d / rc.SMALLEST_EFFECT_D
    a = {"fun_share_steps": {"planned_break": 0, "found_break": s, "ai_saved": 0, "ai_saved_labeled": s},
         "paid_work_steps": {"planned_break": 0, "found_break": 0, "ai_saved": s, "ai_saved_labeled": s}}
    b = {"fun_share_steps": {"planned_break": 0, "found_break": 0, "ai_saved": s, "ai_saved_labeled": s},
         "paid_work_steps": dict(sim.ZERO)}
    return {"A": (a, ("H1", "H2", "H4", "H5")), "B": (b, ("H3",))}


def power_curves(assumptions, draws, seed):
    rows = []
    for i, true_d in enumerate(CURVE_DIFFERENCES):
        row = {"true_d": true_d, "normal_theory": normal_theory_equivalence_power(true_d), "simulated": {}}
        for j, (_, (pattern, hypotheses)) in enumerate(curve_patterns(true_d).items()):
            r = run_pattern(pattern, assumptions, draws, seed + 10 * i + j, forms=(rc.BOUND_FORM,))
            for h in hypotheses:
                share = r["by_bound_form"][rc.BOUND_FORM][h]["equivalence_test_power"]
                lo, hi = wilson(round(share * draws), draws)
                row["simulated"][h] = {"share": share, "interval": [lo, hi], "draws": draws,
                                       "share_by_reading": r["by_bound_form"][rc.BOUND_FORM][h]["share_by_reading"]}
        rows.append(row)
        print(f"curve at d = {true_d}: done")
    return rows


def sensitivity(draws, seed):
    """The no-effect pattern when the true SD of each outcome differs from the SD behind the fixed bound."""
    settings = [  # (fun share SD in points, paid work SD in minutes, extra-slot probability that reaches that SD)
        (25.0, 2.0, 0.5),
        (29.0, 3.0, 0.5),
        (33.0, 4.0, 0.9),
    ]
    rows = []
    for i, (fun_sd, work_sd, extra) in enumerate(settings):
        a = replace(sim.Assumptions(), fun_share_within_country_sd=fun_sd, paid_work_within_country_sd=work_sd,
                    paid_work_extra_slot_probability=extra)
        r = run_pattern(sim.PATTERNS["no_effects"], a, draws, seed + i)
        rows.append({
            "fun_share_sd": fun_sd, "paid_work_sd": work_sd,
            "normal_theory_fixed": {
                "fun_share": normal_theory_equivalence_power(0.0, rc.FUN_SHARE_BOUND / fun_sd),
                "paid_work_minutes": normal_theory_equivalence_power(0.0, rc.PAID_WORK_MINUTES_BOUND / work_sd)},
            "simulated": {form: {h: {"share": r["by_bound_form"][form][h]["share_expected"],
                                     "interval": r["by_bound_form"][form][h]["interval"]}
                                 for h in rc.HYPOTHESES} for form in rc.BOUND_FORMS},
        })
    return rows


def criterion_thresholds(n=rc.PARTICIPANTS_PER_GROUP):
    """Normal theory: the true value at which each criterion fails with probability 0.05 and with probability 0.50."""
    z95 = stats.norm.ppf(1 - (1 - rc.CHECK_INTERVAL_LEVEL) / 2)
    se_d = np.sqrt(2 / n)

    def solve(fail_probability, f, lo, hi):
        return float(optimize.brentq(lambda x: f(x) - fail_probability, lo, hi))

    def share_fail_below(p, minimum):  # P(observed share < minimum) at a true share p
        return stats.norm.cdf((minimum - p) / np.sqrt(p * (1 - p) / n))

    def share_fail_above(p, maximum):
        return stats.norm.sf((maximum - p) / np.sqrt(p * (1 - p) / n))

    def label_fail(delta, unlabeled=0.20):  # P(observed difference < 20 points or lower end <= 0)
        p1 = unlabeled + delta
        se = np.sqrt(p1 * (1 - p1) / n + unlabeled * (1 - unlabeled) / n)
        return stats.norm.cdf((max(rc.LABEL_MIN_DIFFERENCE, z95 * se) - delta) / se)

    def floor_fail(rate):  # paid-work floor, with the simulator's form of the minutes: mean 6a, variance 40a - 36a^2
        mean, var = 6 * rate, 40 * rate - 36 * rate ** 2
        return stats.norm.cdf((rc.PAID_WORK_MIN_SHARE_OF_FREE_MINUTES * rc.FREE_MINUTES - mean) / np.sqrt(var / n))

    out = {}
    surprise_cut = rc.SURPRISE_MIN_D + z95 * se_d
    out["surprise (each group, in d of the surprise score)"] = {
        str(p): float(surprise_cut + stats.norm.ppf(1 - p) * se_d) for p in (0.05, 0.5)}
    out["attribution, AI-saved part (true share in each AI-saved group)"] = {
        str(p): solve(1 - np.sqrt(1 - p), lambda x: share_fail_below(x, rc.ATTRIBUTION_MIN_SHARE_AI_GROUPS), 0.5, 0.999)
        for p in (0.05, 0.5)}
    out["attribution, found-break part: share choosing the assistant (true share)"] = {
        str(p): solve(p, lambda x: share_fail_above(x, rc.ATTRIBUTION_MAX_SHARE_FOUND_GROUP), 0.01, 0.5)
        for p in (0.05, 0.5)}
    out["attribution, found-break part: share choosing 'would not be reviewed' (true share)"] = {
        str(p): solve(p, lambda x: share_fail_below(x, rc.ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED), 0.3, 0.99)
        for p in (0.05, 0.5)}
    out["label (true difference in share, unlabeled at 0.20)"] = {
        str(p): solve(p, label_fail, 0.0, 0.79) for p in (0.05, 0.5)}
    se_fun = rc.FUN_SHARE_SD / np.sqrt(n)
    out["fun-share floor (true planned mean, points; the ceiling is symmetric at 100 minus this)"] = {
        str(p): float(rc.FUN_SHARE_FLOOR + stats.norm.ppf(1 - p) * se_fun) for p in (0.05, 0.5)}
    out["paid-work floor (true share of the planned group's free minutes)"] = {
        str(p): 0.6 * solve(p, floor_fail, 0.01, 0.5) for p in (0.05, 0.5)}
    return out


def report_text(out):
    lines = []
    add = lines.append
    add("Power by simulation for H1 to H5 (analysis/power_by_simulation.py)")
    add(f"Each data set: {rc.PARTICIPANTS_PER_GROUP} analyzable participants a group in 12 countries; alpha "
        f"{rc.ALPHA:.4f} a test; bounds d = {rc.SMALLEST_EFFECT_D} (registered form: {rc.BOUND_FORM}). Readings "
        f"include the criterion gates: a failed criterion makes its tests not interpretable, which counts as a miss.")
    add(f"Time: {out['seconds']:.0f} seconds on one processor thread.")
    add("")
    cal = out["calibration"]
    step = cal["intended_step"]
    a = sim.Assumptions()
    add("1. What the simulator produces (one data set 200 times the registered size for each pattern)")
    add(f"   Intended step between two groups: {step['fun_share']:.2f} points of fun share, "
        f"{step['paid_work_minutes']:.3f} minutes of paid work, each d = {step['d']:.2f} of the pooled within-country "
        f"SD. Intended within-country SD: {a.fun_share_within_country_sd:.0f} points and "
        f"{a.paid_work_within_country_sd:.1f} minutes (leavers included, at 0 minutes) in groups at step 0; "
        f"{100 * a.leave_early_rate:.0f}% leave early in every group.")
    for name in sim.PATTERNS:
        e = cal[name]
        sd_f = e["within_country_sd"]["fun_share"]
        sd_w = e["within_country_sd"]["paid_work_minutes"]
        add(f"   {name}: within-country SD of the fun share by group "
            + ", ".join(f"{v:.1f}" for v in sd_f.values()) + " points; of paid-work minutes "
            + ", ".join(f"{v:.2f}" for v in sd_w.values()) + f" minutes; {100 * e['share_left_early']:.1f}% left early.")
        add("      Planted differences: " + "; ".join(
            f"{h} {p['difference']:.2f} {p['unit']} (d = {p['d']:.3f})" for h, p in e["planted"].items()) + ".")
    add("")
    add("2. The five patterns: share of data sets with each reading, and the share with the expected reading with its "
        "95% Monte Carlo interval")
    for name, r in out["patterns"].items():
        pattern = sim.PATTERNS[name]
        add(f"   Pattern {name} ({r['draws']} data sets). {pattern['description']}")
        add("      Share of data sets in which each criterion failed: " + ", ".join(
            f"{c} {v:.4f}" for c, v in r["criterion_failure_share"].items())
            + f"; any criterion {r['any_criterion_failed_share']:.4f}.")
        add(f"      Share in which the AI-saved group was less surprised than the found-break group, with the interval "
            f"below minus 0.15 pooled SD (a limit on reading H3 and H5, not a criterion): "
            f"{r['ai_saved_less_surprised_than_found_share']:.4f}.")
        add("      Mean estimate across data sets (mean SE): " + "; ".join(
            f"{h} {r['mean_estimate'][h]:.2f} ({r['mean_se'][h]:.2f}) "
            f"{rc.OUTCOME_UNITS[rc.HYPOTHESES[h]['outcome']]}" for h in rc.HYPOTHESES) + ".")
        add(f"      H3 and H5 read as a pair (registered bound form): expected {r['expected_pair']}: "
            f"{r['expected_pair_share']:.3f} [{r['expected_pair_interval'][0]:.3f}, {r['expected_pair_interval'][1]:.3f}]."
            " All pair readings: " + "; ".join(f"{k} {v:.3f}" for k, v in r["pair_reading_share"].items()) + ".")
        for form in rc.BOUND_FORMS:
            add(f"      Bound form: {form}")
            for h, row in r["by_bound_form"][form].items():
                shares = ", ".join(f"{k} {v:.3f}" for k, v in row["share_by_reading"].items())
                claim = (f"Table 1 claims {row['claimed_in_table_1']:.3f}; {row['verdict_against_0_95']}; "
                         f"{row['verdict_against_0_96']}" if row["claimed_in_table_1"] is not None
                         else "not powered (reported only)")
                add(f"         {h}: expected {row['expected_reading']}: {row['share_expected']:.3f} "
                    f"[{row['interval'][0]:.3f}, {row['interval'][1]:.3f}]; {claim}. All readings: {shares}. "
                    f"Significant and inside the bounds at once: {row['share_significant_and_equivalent']:.4f}.")
    add("")
    add("3. Power curves of the equivalence tests (both one-sided tests significant at .01; registered bound form), "
        "with the normal-theory value at the registered sample")
    for row in out["power_curves"]:
        add(f"   True difference {row['true_d']:.2f} d (normal theory {row['normal_theory']:.3f}): " + "; ".join(
            f"{h} {v['share']:.3f} [{v['interval'][0]:.3f}, {v['interval'][1]:.3f}]"
            for h, v in sorted(row["simulated"].items())) + f" ({next(iter(row['simulated'].values()))['draws']} "
            f"data sets).")
    add("")
    add("4. Sensitivity: share of no-effect data sets with the expected reading when the true SD differs")
    for row in out["sensitivity"]:
        add(f"   Fun-share SD {row['fun_share_sd']:.0f} points, paid-work SD {row['paid_work_sd']:.1f} minutes "
            f"(normal theory with the fixed bound: fun share {row['normal_theory_fixed']['fun_share']:.3f}, "
            f"paid work {row['normal_theory_fixed']['paid_work_minutes']:.3f})")
        for form in rc.BOUND_FORMS:
            add(f"      {form}: " + "; ".join(
                f"{h} {v['share']:.3f} [{v['interval'][0]:.3f}, {v['interval'][1]:.3f}]"
                for h, v in row["simulated"][form].items()))
    add("")
    add("5. The outcome-neutral criteria by normal theory: the true value at which each fails with probability 0.05 and "
        "0.50, at the registered sample")
    for name, values in out["criterion_thresholds"].items():
        add(f"   {name}: fails with probability 0.05 at {values['0.05']:.3f}, and 0.50 at {values['0.5']:.3f}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Power by simulation for H1 to H5.")
    parser.add_argument("--draws", type=int, default=20000, help="data sets for each pattern (default 20,000)")
    parser.add_argument("--curve-draws", type=int, default=10000, help="data sets for each point of the power curves")
    parser.add_argument("--sensitivity-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=sim.BASE_SEED)
    parser.add_argument("--out-name", default="power_by_simulation", help="stem of the two output files")
    args = parser.parse_args(argv)
    start = time.time()
    assumptions = sim.Assumptions()
    out = {"assumptions": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(assumptions).items()},
           "registered": {"participants_per_group": rc.PARTICIPANTS_PER_GROUP, "alpha": rc.ALPHA,
                          "bound_form": rc.BOUND_FORM, "smallest_effect_d": rc.SMALLEST_EFFECT_D},
           "calibration": calibration_table(assumptions), "patterns": {}}
    for i, (name, pattern) in enumerate(sim.PATTERNS.items()):
        t0 = time.time()
        out["patterns"][name] = run_pattern(pattern, assumptions, args.draws, args.seed + 200 + i)
        print(f"{name}: {args.draws} data sets in {time.time() - t0:.1f} seconds")
    out["power_curves"] = power_curves(assumptions, args.curve_draws, args.seed + 400)
    out["sensitivity"] = sensitivity(args.sensitivity_draws, args.seed + 300)
    out["criterion_thresholds"] = criterion_thresholds()
    out["seconds"] = time.time() - start
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{args.out_name}.json").write_text(json.dumps(an.jsonable(out), indent=2), encoding="utf-8")
    text = report_text(out)
    (OUTPUT / f"{args.out_name}.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
