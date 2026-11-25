"""The registered analysis of the experiment, as Table 1 of the Stage 1 protocol states it.

Run from the repository's root:
    python analysis/analyze.py analysis/output/simulated_no_effects.csv

It reads one file in the survey's export format (one row per participant who started) and runs, in this order:
1. the exclusions (only those registered: a failed attention check before assignment, not finishing the session, and
   withdrawing the answers at the debriefing; no one is excluded on any answer given after assignment), with the
   table of attrition by group and country;
2. the outcome-neutral criteria (registered_constants.CHECK_GATES), with the tests that each makes not interpretable
   when it fails;
3. H1 to H5: one linear regression for each outcome, on group (all four groups, the planned break as reference) with
   country fixed effects and HC2 errors (the share of fun is computed over the picks a participant made, and a
   participant who made no pick has no share, so is left out of H1, H3, H4 and H5 only and stays in H2 and every
   other outcome; the report gives the number by group); each hypothesis is a contrast of two groups from that
   regression, with a two-sided test at .01 and two one-sided tests at .01 against the bounds of plus and minus
   d = 0.15 (d times the pooled SD of the two groups' raw values); then the reading of each test, and the reading of
   H3 and H5 as a pair by the first rule of registered_constants.PAIR_RULES (Supplementary Table S4) that holds;
4. the planned secondary analyses, as estimates with 95 percent intervals and no test.
It writes a results file (JSON) and a plain-text report, and prints the reading of every test.

Each test first gives a result (registered_constants, "The readings"): higher or lower when the two-sided test is
significant at .01, equivalent when both one-sided tests are (the 98 percent interval lies inside the bounds), and
unresolved when neither is. A difference hypothesis reads significance first, and H3 reads equivalence first; at the
registered sample both cannot happen at once. Each hypothesis names each result (registered_constants.HYPOTHESES,
"readings"):
    H1 found minus planned, fun share:       higher supported, lower opposite, equivalent;
    H2 AI-saved minus found, paid minutes:   higher supported (the work account), lower (the goal-progress account),
                                             equivalent;
    H3 AI-saved minus planned, fun share:    equivalent supported, higher than planned, lower than planned;
    H4 labeled minus unlabeled, fun share:   higher (a named saving treated as a gain), lower (the label reminds
                                             people of the work), equivalent (the label does not change it);
    H5 AI-saved minus found, fun share:      lower supported (the work account), higher opposite (the goal-progress
                                             account), equivalent (the windfall account);
and every hypothesis can read unresolved. A failed criterion replaces the reading with "not interpretable"; the test's
estimate and intervals are still reported.

The regression is fitted twice: by statsmodels, and by a few lines of numpy that fit many simulated data sets at
once (power_by_simulation.py uses them). The script stops if the two disagree.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import registered_constants as rc

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

SUPPORTED, OPPOSITE, EQUIVALENT, UNRESOLVED = "supported", "opposite", "equivalent", "unresolved"
HIGHER, LOWER = "higher", "lower"
HIGHER_THAN_PLANNED, LOWER_THAN_PLANNED = "higher than planned", "lower than planned"
NOT_INTERPRETABLE = rc.NOT_INTERPRETABLE

PICK_COLUMNS = [f"video_pick_{j + 1}" for j in range(rc.VIDEO_PICKS)]
SLOT_COLUMNS = [f"slot_{j + 1}_use" for j in range(rc.VIDEO_PICKS)]
SURPRISE_ITEMS = ["surprise_unexpected", "surprise_surprising"]
RECOGNITION_COLUMNS = list(rc.RECOGNITION_SENTENCES)
# Every column that the analysis uses must be present for every analyzed participant (the survey requires each
# answer). Three have no value by design for some participants and are checked separately: fun_share (no value for a
# participant who made no pick), minutes_video_page_hidden (no value without video minutes) and employment_status
# (from Prolific's export, which may have expired).
FILLED_COLUMNS = list(rc.PICK_FILLED_COLUMNS)
REQUIRED_COLUMNS = (PICK_COLUMNS + FILLED_COLUMNS + SLOT_COLUMNS
                    + ["left_early", "paid_work_minutes", "ai_use", "how_review_ended"]
                    + SURPRISE_ITEMS + RECOGNITION_COLUMNS)
COLUMNS_WITH_GAPS = ["fun_share", "minutes_video_page_hidden"]
ROUND = 9  # decimals kept when a regression estimate is compared with a threshold, so 0.2 stays 0.2


class DataError(ValueError):
    """The file cannot be analyzed as registered without a decision that the protocol does not make."""


# =================================================================================================================
# 1. Exclusions and the attrition table
# =================================================================================================================
def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def to_codes(series, answers):
    """Map the screen's option text to the export code; codes pass through unchanged."""
    inverse = {label.strip().lower(): code for code, label in answers.items()}
    return series.map(lambda v: v if v in answers or pd.isna(v) else inverse.get(str(v).strip().lower(), v))


def flags(raw):
    passed = as_bool(raw["passed_attention_check"])
    finished = as_bool(raw["finished"])
    withdrew = as_bool(raw["withdrew"]) if "withdrew" in raw else pd.Series(False, index=raw.index)
    assigned = raw["group"].isin(rc.GROUPS)
    return passed, finished, withdrew, assigned


def attrition_table(raw, analyzed):
    """Assigned, not finishing, withdrawing and leaving early, by group and by group and country."""
    passed, finished, withdrew, assigned = flags(raw)
    started = raw.loc[passed & assigned]

    def counts(rows, kept):
        n_assigned = int(len(rows))
        not_finished = int((~as_bool(rows["finished"])).sum())
        withdrawn = int((as_bool(rows["finished"]) & (as_bool(rows["withdrew"]) if "withdrew" in rows else False)).sum())
        return {"assigned": n_assigned, "did_not_finish": not_finished, "withdrew": withdrawn,
                "analyzed": int(len(kept)), "left_early": int(kept["left_early"].sum()),
                "share_did_not_finish_or_withdrew": (not_finished + withdrawn) / n_assigned if n_assigned else 0.0}

    by_group = {g: counts(started[started["group"] == g], analyzed[analyzed["group"] == g]) for g in rc.GROUPS}
    by_group_country = {
        g: {c: counts(started[(started["group"] == g) & (started["country"] == c)],
                      analyzed[(analyzed["group"] == g) & (analyzed["country"] == c)])
            for c in sorted(started["country"].dropna().unique())}
        for g in rc.GROUPS}
    shares = [v["share_did_not_finish_or_withdrew"] for v in by_group.values()]
    gap = max(shares) - min(shares)
    return {"by_group": by_group, "by_group_and_country": by_group_country,
            "largest_gap_between_groups": gap, "threshold": rc.ATTRITION_DIFFERENCE_THRESHOLD,
            "threshold_crossed": bool(round(gap, ROUND) > rc.ATTRITION_DIFFERENCE_THRESHOLD)}


def apply_exclusions(raw):
    """Keep participants who passed the attention check, were assigned, finished and did not withdraw."""
    passed, finished, withdrew, assigned = flags(raw)
    keep = passed & assigned & finished & ~withdrew
    data = raw.loc[keep].copy()
    by_group = {g: int((data["group"] == g).sum()) for g in rc.GROUPS}
    summary = {
        "rows_read": int(len(raw)),
        "failed_attention_check": int((~passed).sum()),
        "assigned": int((passed & assigned).sum()),
        "did_not_finish": int((passed & assigned & ~finished).sum()),
        "withdrew": int((passed & assigned & finished & withdrew).sum()),
        "finished_but_no_group_recorded": int((passed & finished & ~assigned).sum()),
        "analyzed": int(keep.sum()),
        "analyzed_by_group": by_group,
        "registered_per_group": rc.PARTICIPANTS_PER_GROUP,
        "every_group_at_least_registered_size": all(v >= rc.PARTICIPANTS_PER_GROUP for v in by_group.values()),
    }

    # No one is excluded after assignment: a finisher with no group, or a missing answer, stops the analysis instead
    # of dropping the row.
    if summary["finished_but_no_group_recorded"]:
        raise DataError(f"{summary['finished_but_no_group_recorded']} participants passed the attention check and "
                        f"finished but have no group among {rc.GROUPS}; the analysis stops here instead of "
                        f"dropping them.")
    absent = [c for c in REQUIRED_COLUMNS + COLUMNS_WITH_GAPS if c not in data]
    if absent:
        raise DataError(f"the export has no column {absent}")
    data["ai_use"] = to_codes(data["ai_use"], rc.AI_USE_ANSWERS)
    data["how_review_ended"] = to_codes(data["how_review_ended"], rc.ATTRIBUTION_ANSWERS)
    if "employment_status" not in data:
        data["employment_status"] = np.nan
    data["employment_status"] = to_codes(data["employment_status"], rc.EMPLOYMENT_ANSWERS)
    missing = {c: int(data[c].isna().sum()) for c in REQUIRED_COLUMNS}
    gaps = {c: m for c, m in missing.items() if m}
    if gaps:
        raise DataError(f"answers are missing among participants who finished: {gaps}. No one is excluded after "
                        f"assignment, so the analysis stops here.")
    summary["employment_unknown"] = int((~data["employment_status"].isin(rc.EMPLOYMENT_ANSWERS)).sum())

    picks = data[PICK_COLUMNS]
    if not picks.isin(set(rc.FUN_VIDEOS) | set(rc.SELF_IMPROVEMENT_VIDEOS)).all(axis=None):
        raise DataError("a video pick is not one of the ten registered videos")
    if (picks.nunique(axis=1) != rc.VIDEO_PICKS).any():
        raise DataError("a participant has the same video among the five picks twice")
    is_fun = picks.isin(rc.FUN_VIDEOS).to_numpy()
    filled = np.column_stack([as_bool(data[c]).to_numpy() for c in FILLED_COLUMNS])
    made = rc.VIDEO_PICKS - filled.sum(axis=1)
    fun_share = data["fun_share"].to_numpy(float)
    if not np.array_equal(np.isnan(fun_share), made == 0):
        raise DataError("fun_share must be empty for a participant who made no pick, and only for them")
    with np.errstate(invalid="ignore", divide="ignore"):
        recomputed = 100.0 * (is_fun & ~filled).sum(axis=1) / made
    if not np.allclose(recomputed[made > 0], fun_share[made > 0]):
        raise DataError("the exported fun_share does not match the picks the participant made (filled picks do not "
                        "count)")

    uses = data[SLOT_COLUMNS]
    if not uses.isin(rc.SLOT_USES).all(axis=None):
        raise DataError(f"a slot use is not one of {rc.SLOT_USES}")
    uses = uses.to_numpy()
    n_left = (uses == "left_early").sum(axis=1)
    if not np.isin(n_left, (0, rc.VIDEO_PICKS)).all():
        raise DataError("leaving early replaces all five slots; a participant has it on some slots only")
    left = as_bool(data["left_early"]).to_numpy()
    if not np.array_equal(left, n_left == rc.VIDEO_PICKS):
        raise DataError("left_early does not match the five slot uses")
    minutes = data["paid_work_minutes"].to_numpy(float)
    if not np.allclose(minutes, (uses == "paid_work").sum(axis=1) * rc.SLOT_MINUTES):
        raise DataError("paid_work_minutes must be 2 minutes for each slot given to paid extra work")
    if ((made < rc.VIDEO_PICKS) & ~(uses == "watch").all(axis=1)).any():
        raise DataError("a participant with picks filled at random never saw the slots, so every slot plays a video")
    watch = uses == "watch"
    video_minutes = watch.sum(axis=1) * float(rc.SLOT_MINUTES)
    hidden = data["minutes_video_page_hidden"].to_numpy(float)
    if np.isnan(hidden[video_minutes > 0]).any():
        raise DataError("minutes_video_page_hidden is missing for a participant who gave minutes to videos")
    if (np.nan_to_num(hidden[video_minutes == 0]) != 0).any():
        raise DataError("minutes_video_page_hidden must be empty for a participant who left or gave every slot to work")
    hidden = np.where(video_minutes > 0, hidden, np.nan)
    known = ~np.isnan(hidden)
    if ((hidden[known] < 0) | (hidden[known] > video_minutes[known] + 1e-9)).any():
        raise DataError("minutes_video_page_hidden must lie between 0 and the minutes given to videos")
    for column in SURPRISE_ITEMS:
        lo, hi = rc.SURPRISE_RANGE
        values = data[column].to_numpy(float)
        if ((values < lo) | (values > hi) | (values % 1 != 0)).any():
            raise DataError(f"{column} must be a whole number from {lo} to {hi}")
    if not data["how_review_ended"].isin(rc.ATTRIBUTION_ANSWERS).all():
        raise DataError(f"how_review_ended must be one of {list(rc.ATTRIBUTION_ANSWERS.values())}")
    if not data["ai_use"].isin(rc.AI_USE_ANSWERS).all():
        raise DataError(f"ai_use must be one of {list(rc.AI_USE_ANSWERS.values())}")

    data["fun_share"] = fun_share
    data["picks_made"] = made
    data["made_all_five_picks"] = made == rc.VIDEO_PICKS
    data["paid_work_minutes"] = minutes
    data["left_early"] = left
    data["surprise"] = data[SURPRISE_ITEMS].astype(float).mean(axis=1)
    data["says_ai_finished"] = (data["how_review_ended"] == rc.ATTRIBUTION_ANSWER_AI).astype(float)
    data["says_not_reviewed"] = (data["how_review_ended"] == rc.ATTRIBUTION_ANSWER_NOT_REVIEWED).astype(float)
    for column in RECOGNITION_COLUMNS:
        data[column] = as_bool(data[column]).astype(float)
    data["saw_label"] = data[rc.LABEL_SENTENCE]
    # The four uses of the 10 minutes, in minutes (they sum to 10 for every participant).
    data["fun_videos_watched"] = (watch & is_fun).sum(axis=1) * float(rc.SLOT_MINUTES)
    data["self_improvement_videos_watched"] = (watch & ~is_fun).sum(axis=1) * float(rc.SLOT_MINUTES)
    data["paid_work"] = minutes
    data["left_early_minutes"] = np.where(left, float(rc.FREE_MINUTES), 0.0)
    total = data[["fun_videos_watched", "self_improvement_videos_watched", "paid_work", "left_early_minutes"]].sum(axis=1)
    if not np.allclose(total, rc.FREE_MINUTES):
        raise DataError("the four uses of the free minutes do not add up to 10")
    data["video_minutes"] = video_minutes
    data["minutes_video_page_hidden"] = hidden
    data["hidden_share_of_video_minutes"] = hidden / np.maximum(video_minutes, 1e-12)  # no value without video minutes
    # A participant who made no pick has no share of fun: left out of H1, H3, H4 and H5 only (registered_constants).
    summary["no_pick_made"] = int((made == 0).sum())
    summary["no_pick_made_by_group"] = {g: int(((made == 0) & (data["group"] == g).to_numpy()).sum())
                                        for g in rc.GROUPS}
    summary["some_picks_filled_at_random"] = int(((made > 0) & (made < rc.VIDEO_PICKS)).sum())
    return data.reset_index(drop=True), summary


# =================================================================================================================
# 2. The regression: outcome on group (planned break as reference) and country fixed effects, HC2 errors.
#    Y may hold many outcome columns at once (one per simulated data set).
# =================================================================================================================
class CountryFixedEffectsOLS:
    def __init__(self, group_index, country_index, covariates=None):
        group_index = np.asarray(group_index)
        _, country_codes = np.unique(np.asarray(country_index), return_inverse=True)
        n, n_groups, n_countries = len(group_index), len(rc.GROUPS), country_codes.max() + 1
        columns = [np.ones(n)] + [(group_index == j).astype(float) for j in range(1, n_groups)]
        columns += [(country_codes == j).astype(float) for j in range(1, n_countries)]
        if covariates is not None:
            columns += [np.asarray(c, dtype=float) for c in np.atleast_2d(np.asarray(covariates, dtype=float))]
        x = np.column_stack(columns)
        self.x = x
        self.projector = np.linalg.solve(x.T @ x, x.T)  # (X'X)^-1 X'
        self.leverage = np.einsum("ij,ji->i", x, self.projector)
        self.df = n - x.shape[1]
        self.group_index = group_index

    def fit(self, y):
        return _Fit(self, np.asarray(y, dtype=float).reshape(len(self.group_index), -1))


class _Fit:
    def __init__(self, model, y):
        self.model, self.y = model, y
        self.residuals = y - model.x @ (model.projector @ y)

    def contrast(self, minuend, subtrahend):
        """Estimate and HC2 standard error of the minuend group minus the subtrahend group."""
        c = np.zeros(self.model.x.shape[1])
        for group, sign in ((minuend, 1.0), (subtrahend, -1.0)):
            j = rc.GROUPS.index(group)
            if j > 0:
                c[j] += sign
        weights = c @ self.model.projector
        estimate = weights @ self.y
        variance = (weights ** 2 / (1.0 - self.model.leverage)) @ self.residuals ** 2
        return estimate, np.sqrt(variance)


def model_for(data, covariates=None):
    return CountryFixedEffectsOLS(data["group"].map(rc.GROUPS.index).to_numpy(), data["country"].to_numpy(),
                                  covariates)


# =================================================================================================================
# 3. The reading rules (each works on one test or on many simulated tests at once)
# =================================================================================================================
def two_sided_p(estimate, se, df):
    return 2 * stats.t.sf(np.abs(estimate / se), df)


def equivalence_p(estimate, se, df, bound):
    """Larger p of the two one-sided tests: against a difference of -bound or less, and of +bound or more."""
    p_above_lower = stats.t.sf((estimate + bound) / se, df)
    p_below_upper = stats.t.cdf((estimate - bound) / se, df)
    return np.maximum(p_above_lower, p_below_upper)


def test_result(estimate, se, df, bound, equivalence_first=False, alpha=None):
    """higher, lower, equivalent or unresolved, from the two-sided test and the two one-sided tests at alpha."""
    alpha = rc.ALPHA if alpha is None else alpha
    significant = two_sided_p(estimate, se, df) < alpha
    equivalent = equivalence_p(estimate, se, df, bound) < alpha
    by_sign = np.where(significant & (estimate > 0), HIGHER, np.where(significant, LOWER, UNRESOLVED))
    if equivalence_first:
        return np.where(equivalent, EQUIVALENT, by_sign)
    return np.where(significant, by_sign, np.where(equivalent, EQUIVALENT, UNRESOLVED))


def read_hypothesis(h, estimate, se, df, bound, alpha=None):
    """The reading of hypothesis h: its result named by registered_constants.HYPOTHESES[h]['readings']."""
    spec = rc.HYPOTHESES[h]
    result = test_result(np.asarray(estimate, float), np.asarray(se, float), df, np.asarray(bound, float),
                         spec["kind"] == "equivalence", alpha)
    names = spec["readings"]
    return np.vectorize(names.__getitem__, otypes=[object])(result), result


def pair_rule_holds(rule, h1, h3, h5):
    """Where one rule of registered_constants.PAIR_RULES (Supplementary Table S4) holds; h1, h3, h5 are str arrays."""
    _, h1_condition, h3_results, h5_results, join, _ = rule
    every = np.ones(h3.shape, dtype=bool)
    on_h3 = every if h3_results is rc.PAIR_ANY else np.isin(h3, h3_results)
    on_h5 = every if h5_results is rc.PAIR_ANY else np.isin(h5, h5_results)
    holds = (on_h3 | on_h5) if join == "or" else (on_h3 & on_h5)
    if h1_condition == "supported":
        holds = holds & (h1 == SUPPORTED)
    elif h1_condition == "not supported":
        holds = holds & (h1 != SUPPORTED)
    return holds


def pair_reading(h1, h3, h5, rule_numbers=False):
    """H3 and H5 read as a pair, given H1; works on arrays.

    The rules of registered_constants.PAIR_RULES (Supplementary Table S4) are tried in their order, and the first
    that holds gives the reading. With rule_numbers=True, the number of that rule is returned instead.
    """
    h1, h3, h5 = (np.asarray(x).astype(str) for x in (h1, h3, h5))
    h1 = np.broadcast_to(h1, h3.shape)
    conditions = [pair_rule_holds(rule, h1, h3, h5) for rule in rc.PAIR_RULES]
    if rule_numbers:
        return np.select(conditions, [rule[0] for rule in rc.PAIR_RULES], default=0)
    return np.select(conditions, [rule[-1] for rule in rc.PAIR_RULES], default=rc.PAIR_NO_RULE).astype(object)


def pooled_sd(y_a, y_b):
    """SD pooled over two groups' raw values: ((n1 - 1) s1^2 + (n2 - 1) s2^2) / (n1 + n2 - 2), square root."""
    n1, n2 = y_a.shape[0], y_b.shape[0]
    return np.sqrt(((n1 - 1) * y_a.var(axis=0, ddof=1) + (n2 - 1) * y_b.var(axis=0, ddof=1)) / (n1 + n2 - 2))


def equivalence_bound(outcome, y_minuend, y_subtrahend, bound_form):
    """The bound in points or minutes: d = 0.15 x the pooled SD (registered), or the fixed value (comparison)."""
    draws = y_minuend.shape[1]
    if bound_form == "fixed":
        return np.full(draws, rc.FIXED_BOUNDS[outcome])
    if bound_form == "observed_sd":
        return rc.SMALLEST_EFFECT_D * pooled_sd(y_minuend, y_subtrahend)
    raise ValueError(f"unknown bound form {bound_form!r}")


# =================================================================================================================
# 4. The outcome-neutral criteria
# =================================================================================================================
def outcome_neutral_checks(model, fun_share, paid_work_minutes, surprise, says_ai, says_not_reviewed, saw_label):
    """Criteria a to d. Every input has shape (participants, draws); every result has one value a draw."""
    g = model.group_index
    t_check = stats.t.ppf(1 - (1 - rc.CHECK_INTERVAL_LEVEL) / 2, model.df)
    shape = (len(g), -1)
    member = {group: g == k for k, group in enumerate(rc.GROUPS)}

    # (a) Surprise: the 95 percent interval of each difference from the planned group lies above 0.15 pooled SD.
    surprise_fit = model.fit(surprise)
    surprise_result = {}
    for group in rc.SURPRISE_GROUPS:
        est, se = surprise_fit.contrast(group, rc.REFERENCE_GROUP)
        minimum = rc.SURPRISE_MIN_D * pooled_sd(surprise_fit.y[member[group]], surprise_fit.y[member[rc.REFERENCE_GROUP]])
        lower = est - t_check * se
        surprise_result[group] = {"difference": est, "lower": lower, "upper": est + t_check * se,
                                  "minimum": minimum, "passed": lower > minimum}
    est, se = surprise_fit.contrast("ai_saved", "found_break")
    limit = rc.SURPRISE_WINDFALL_LIMIT_D * pooled_sd(surprise_fit.y[member["ai_saved"]],
                                                     surprise_fit.y[member["found_break"]])
    ai_minus_found = {"difference": est, "lower": est - t_check * se, "upper": est + t_check * se,
                      "limit": -limit, "ai_saved_less_surprised": est + t_check * se < -limit}

    # (b) Attribution, on the observed shares.
    says_ai = np.asarray(says_ai, dtype=float).reshape(shape)
    says_not_reviewed = np.asarray(says_not_reviewed, dtype=float).reshape(shape)
    share_ai = {group: says_ai[member[group]].mean(axis=0) for group in rc.GROUPS}
    share_not_reviewed = {group: says_not_reviewed[member[group]].mean(axis=0) for group in rc.GROUPS}
    attribution_ai_saved = ((share_ai["ai_saved"] >= rc.ATTRIBUTION_MIN_SHARE_AI_GROUPS)
                            & (share_ai["ai_saved_labeled"] >= rc.ATTRIBUTION_MIN_SHARE_AI_GROUPS))
    attribution_found = ((share_ai["found_break"] <= rc.ATTRIBUTION_MAX_SHARE_FOUND_GROUP)
                         & (share_not_reviewed["found_break"] >= rc.ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED))

    # (c) The label: at least 20 points more, and the 95 percent interval of the difference above zero.
    label_est, label_se = model.fit(saw_label).contrast("ai_saved_labeled", "ai_saved")
    label_lower = label_est - t_check * label_se
    label_passed = (np.round(label_est, ROUND) >= rc.LABEL_MIN_DIFFERENCE) & (label_lower > 0)

    # (d) Floor and ceiling, in the planned-break group.
    planned = member[rc.REFERENCE_GROUP]
    fun = np.asarray(fun_share, dtype=float).reshape(shape)
    planned_mean = np.nanmean(fun[planned], axis=0)  # over those with a share of fun (at least one pick made)
    work = np.asarray(paid_work_minutes, dtype=float).reshape(shape)
    paid_share = work[planned].sum(axis=0) / (planned.sum() * rc.FREE_MINUTES)

    passed = {
        "surprise_found_break": surprise_result["found_break"]["passed"],
        "surprise_ai_saved": surprise_result["ai_saved"]["passed"],
        "surprise_ai_saved_labeled": surprise_result["ai_saved_labeled"]["passed"],
        "attribution_ai_saved": attribution_ai_saved,
        "attribution_found_break": attribution_found,
        "label": label_passed,
        "fun_share_floor_ceiling": (planned_mean >= rc.FUN_SHARE_FLOOR) & (planned_mean <= rc.FUN_SHARE_CEILING),
        "paid_work_floor": paid_share >= rc.PAID_WORK_MIN_SHARE_OF_FREE_MINUTES,
    }
    return {
        "surprise": surprise_result,
        "surprise_ai_saved_minus_found": ai_minus_found,
        "attribution": {"share_choosing_ai": share_ai, "share_choosing_not_reviewed": share_not_reviewed},
        "label": {"difference": label_est, "lower": label_lower, "upper": label_est + t_check * label_se},
        "fun_share_floor_ceiling": {"planned_break_mean": planned_mean},
        "paid_work_floor": {"planned_break_share_of_free_minutes": paid_share},
        "passed": passed,
    }


def apply_check_gates(readings, passed):
    """A failed criterion makes the tests tied to it in registered_constants.CHECK_GATES not interpretable."""
    gated = {h: np.asarray(r).astype(object) for h, r in readings.items()}
    for check, hypotheses in rc.CHECK_GATES.items():
        failed = ~np.asarray(passed[check], dtype=bool)
        for h in hypotheses:
            gated[h] = np.where(failed, NOT_INTERPRETABLE, gated[h])
    return gated


# =================================================================================================================
# 5. H1 to H5
# =================================================================================================================
def confirmatory_tests(model, outcomes, bound_form=None):
    """H1 to H5 on outcomes = {"fun_share": Y, "paid_work_minutes": Y}, each Y of shape (participants, draws).

    model is one CountryFixedEffectsOLS for both outcomes, or a dict {outcome: model} when the outcomes have different
    rows (a participant who made no pick has no share of fun but has minutes of paid work).
    """
    bound_form = bound_form or rc.BOUND_FORM
    models = model if isinstance(model, dict) else {o: model for o in rc.OUTCOMES}
    fits = {o: models[o].fit(outcomes[o]) for o in rc.OUTCOMES}
    results = {}
    for h, spec in rc.HYPOTHESES.items():
        m, fit = models[spec["outcome"]], fits[spec["outcome"]]
        g = m.group_index
        est, se = fit.contrast(spec["minuend"], spec["subtrahend"])
        bound = equivalence_bound(spec["outcome"], fit.y[g == rc.GROUPS.index(spec["minuend"])],
                                  fit.y[g == rc.GROUPS.index(spec["subtrahend"])], bound_form)
        reading, result = read_hypothesis(h, est, se, m.df, bound)
        results[h] = {
            "estimate": est, "se": se, "df": m.df, "bound": bound, "n": len(g),
            "p_two_sided": two_sided_p(est, se, m.df),
            "p_equivalence": equivalence_p(est, se, m.df, bound),
            "result": result, "reading": reading,
        }
    return results


def confirmatory_on(data, bound_form=None):
    """H1 to H5 on a data frame, each outcome on the participants who have a value for it."""
    models, outcomes = {}, {}
    for o in rc.OUTCOMES:
        rows = data[data[o].notna()]
        models[o] = model_for(rows)
        outcomes[o] = rows[o].to_numpy(float)
    return confirmatory_tests(models, outcomes, bound_form)


def statsmodels_contrasts(data, formula_extra="", hypotheses=None):
    """The same contrasts from statsmodels' OLS with HC2 errors, to check the numpy code."""
    out = {}
    hypotheses = hypotheses or rc.HYPOTHESES
    for outcome in rc.OUTCOMES:
        if not any(rc.HYPOTHESES[h]["outcome"] == outcome for h in hypotheses):
            continue
        fit = smf.ols(f"{outcome} ~ C(group, Treatment('{rc.REFERENCE_GROUP}')) + C(country){formula_extra}",
                      data=data).fit(cov_type=rc.ROBUST_ERRORS, use_t=True)
        names = list(fit.params.index)
        for h in hypotheses:
            spec = rc.HYPOTHESES[h]
            if spec["outcome"] != outcome:
                continue
            c = np.zeros(len(names))
            for group, sign in ((spec["minuend"], 1.0), (spec["subtrahend"], -1.0)):
                if group != rc.REFERENCE_GROUP:
                    c[[i for i, name in enumerate(names) if name.startswith("C(group") and
                       name.endswith(f"[T.{group}]")][0]] += sign
            out[h] = {"estimate": float(c @ fit.params.to_numpy()),
                      "se": float(np.sqrt(c @ fit.cov_params().to_numpy() @ c)), "df": float(fit.df_resid)}
    return out


# =================================================================================================================
# 6. Planned secondary analyses (intervals only)
# =================================================================================================================
def random_effects(estimates, ses, level=rc.SECONDARY_INTERVAL_LEVEL):
    """DerSimonian and Laird summary, with the prediction interval of Higgins, Thompson and Spiegelhalter (2009)."""
    y, v = np.asarray(estimates, float), np.asarray(ses, float) ** 2
    k = len(y)
    w = 1 / v
    fixed = (w * y).sum() / w.sum()
    q = (w * (y - fixed) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    w_star = 1 / (v + tau2)
    summary = (w_star * y).sum() / w_star.sum()
    se = np.sqrt(1 / w_star.sum())
    z = stats.norm.ppf(1 - (1 - level) / 2)
    t_pred = stats.t.ppf(1 - (1 - level) / 2, k - 2)
    half_pred = t_pred * np.sqrt(tau2 + se ** 2)
    return {"summary": summary, "se": se, "lower": summary - z * se, "upper": summary + z * se,
            "tau": np.sqrt(tau2), "i_squared": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0,
            "prediction_lower": summary - half_pred, "prediction_upper": summary + half_pred, "countries": k}


def interval(est, se, df, level=rc.SECONDARY_INTERVAL_LEVEL):
    t = stats.t.ppf(1 - (1 - level) / 2, df)
    return float(est - t * se), float(est + t * se)


def wilson(successes, n, level=rc.SECONDARY_INTERVAL_LEVEL):
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def group_means(data, column):
    out = {}
    for group in rc.GROUPS:
        y = data.loc[data["group"] == group, column].to_numpy(float)
        m = float(y.mean())
        lo, hi = interval(m, y.std(ddof=1) / np.sqrt(len(y)), len(y) - 1) if len(y) > 1 else (m, m)
        out[group] = {"n": int(len(y)), "mean": m, "lower": lo, "upper": hi}
    return out


def group_shares(data, column):
    out = {}
    for group in rc.GROUPS:
        y = data.loc[data["group"] == group, column].to_numpy(float)
        lo, hi = wilson(y.sum(), len(y))
        out[group] = {"n": int(len(y)), "share": float(y.mean()), "lower": lo, "upper": hi}
    return out


def pair_contrasts(data, column, model=None):
    """The contrast of each hypothesis's two groups on any column, with its 95 percent interval."""
    model = model or model_for(data)
    fit = model.fit(data[column].to_numpy(float))
    out = {}
    for h, spec in rc.HYPOTHESES.items():
        est, se = fit.contrast(spec["minuend"], spec["subtrahend"])
        lo, hi = interval(est[0], se[0], model.df)
        out[h] = {"contrast": f"{spec['minuend']} minus {spec['subtrahend']}", "estimate": float(est[0]),
                  "lower": lo, "upper": hi}
    return out


def contrast_on(data, outcome, minuend, subtrahend, covariate=None):
    """One contrast with its 95 percent interval, on the participants who have the outcome (and the covariate)."""
    needed = [outcome] + ([covariate] if covariate else [])
    rows = data.dropna(subset=needed)
    model = model_for(rows, rows[covariate].to_numpy(float) if covariate else None)
    est, se = model.fit(rows[outcome].to_numpy(float)).contrast(minuend, subtrahend)
    lo, hi = interval(est[0], se[0], model.df)
    return {"estimate": float(est[0]), "se": float(se[0]), "lower": lo, "upper": hi, "n": int(len(rows))}


def contrasts_with_intervals(data):
    """H1 to H5 on their own outcomes in a subgroup, with 95 percent intervals."""
    return {h: contrast_on(data, spec["outcome"], spec["minuend"], spec["subtrahend"])
            for h, spec in rc.HYPOTHESES.items()}


def lee_bounds(y_minuend, y_subtrahend, retained_minuend, retained_subtrahend):
    """Lee (2009) bounds on the difference in means: trim the group with the higher retention by the excess share."""
    a, b = np.sort(np.asarray(y_minuend, float)), np.sort(np.asarray(y_subtrahend, float))
    if retained_minuend >= retained_subtrahend:
        trim = (retained_minuend - retained_subtrahend) / retained_minuend
        k = int(round(trim * len(a)))
        low, high = a[: len(a) - k].mean() - b.mean(), a[k:].mean() - b.mean()
    else:
        trim = (retained_subtrahend - retained_minuend) / retained_subtrahend
        k = int(round(trim * len(b)))
        low, high = a.mean() - b[k:].mean(), a.mean() - b[: len(b) - k].mean()
    return {"lower": float(low), "upper": float(high), "trimmed_share": float(trim), "trimmed": int(k)}


def position_between_breaks(data, draws=rc.POSITION_BOOTSTRAP_DRAWS, seed=rc.POSITION_BOOTSTRAP_SEED, batch=250):
    """(AI-saved minus planned) / (found minus planned) on the share of fun, with a 95 percent percentile bootstrap
    interval (participants resampled within each group and country, so the regression's design stays the same).
    Reported only when found minus planned is significant at the registered alpha."""
    rows = data.dropna(subset=["fun_share"])
    model = model_for(rows)
    y = rows["fun_share"].to_numpy(float)
    fit = model.fit(y)
    found, found_se = fit.contrast("found_break", rc.REFERENCE_GROUP)
    ai, _ = fit.contrast("ai_saved", rc.REFERENCE_GROUP)
    p = float(two_sided_p(found, found_se, model.df)[0])
    out = {"found_minus_planned": float(found[0]), "ai_saved_minus_planned": float(ai[0]), "p_found_minus_planned": p,
           "reported": bool(p < rc.ALPHA)}
    if not out["reported"]:
        out["reason"] = "found minus planned is not significant at the registered alpha, so the ratio is not reported"
        return out
    cells = [np.asarray(v) for v in rows.groupby(["group", "country"]).indices.values()]
    rng = np.random.default_rng(seed)
    ratios = []
    for start in range(0, draws, batch):
        b = min(batch, draws - start)
        index = np.empty((len(y), b), dtype=int)
        for positions in cells:
            index[positions] = rng.choice(positions, size=(len(positions), b), replace=True)
        boot = model.fit(y[index])
        ratios.append(boot.contrast("ai_saved", rc.REFERENCE_GROUP)[0] / boot.contrast("found_break",
                                                                                        rc.REFERENCE_GROUP)[0])
    ratios = np.concatenate(ratios)
    out.update({"ratio": float(ai[0] / found[0]), "lower": float(np.percentile(ratios, 2.5)),
                "upper": float(np.percentile(ratios, 97.5)), "bootstrap_draws": int(draws), "seed": int(seed)})
    return out


def optional_descriptives(data):
    """Planned secondary measures with no hypothesis, each from an optional column of the export."""
    out = {}

    def absent(column):
        return column not in data

    def shares_by(column, rows, codes=None):
        known = rows[rows[column].notna()]
        if codes is None:
            values = {"share": known[column].astype(float)}
        else:
            values = {code: (known[column] == code).astype(float) for code in codes}
        result = {}
        for g in rc.GROUPS:
            in_group = known["group"] == g
            result[g] = {"n": int(in_group.sum()), "missing": int(((rows["group"] == g) & rows[column].isna()).sum())}
            for name, v in values.items():
                n = int(in_group.sum())
                share = float(v[in_group].mean()) if n else float("nan")
                lo, hi = wilson(v[in_group].sum(), n) if n else (float("nan"), float("nan"))
                result[g][name] = {"share": share, "lower": lo, "upper": hi}
        return result

    def by_week(frame, column, codes=None):
        if absent(rc.STARTED_AT_COLUMN):
            return "no started_at column in the export"
        started = pd.to_datetime(frame[rc.STARTED_AT_COLUMN], errors="coerce")
        week = ((started - started.min()).dt.days // 7 + 1).astype("Int64")
        return {int(w): shares_by(column, frame[week == w], codes) for w in sorted(week.dropna().unique())}

    def as_share(column):  # True/False with empty kept empty
        return data.assign(**{column: data[column].map(
            lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() in ("true", "1", "1.0", "yes")))})

    column = rc.AUTOMATED_FLAG_COLUMN
    out["flagged_possibly_automated"] = "not in the export" if absent(column) else shares_by(column, as_share(column))
    column = rc.PURPOSE_CODED_COLUMN
    if absent(column):
        out["purpose_names_reason_or_source"] = "not coded yet (no column in the export)"
    else:
        coded = as_share(column)
        out["purpose_names_reason_or_source"] = {"by_group": shares_by(column, coded),
                                                 "by_week": by_week(coded, column)}
    column = rc.HEARD_BEFORE_COLUMN
    if absent(column):
        out["heard_of_study_before"] = "not in the export"
    else:
        heard = data.assign(**{column: to_codes(data[column], rc.HEARD_BEFORE_ANSWERS)})
        out["heard_of_study_before"] = {"by_group": shares_by(column, heard, rc.HEARD_BEFORE_ANSWERS),
                                        "by_week": by_week(heard, column, rc.HEARD_BEFORE_ANSWERS)}
    for column in rc.TIMING_COLUMNS + rc.REVIEW_EDIT_COLUMNS:
        if absent(column):
            out[column] = "not in the export"
            continue
        rows = data[data[column].notna()]
        out[column] = {g: v | {"missing": int(((data["group"] == g) & data[column].isna()).sum())}
                       for g, v in group_means(rows, column).items()}
    return out


def hidden_page_dropped(data):
    """The robustness rule: the page was hidden for more than half of the minutes given to videos and stayed for."""
    return data["hidden_share_of_video_minutes"] > rc.HIDDEN_SHARE_OF_VIDEO_MINUTES_MAX


def secondary_analyses(data, checks_passed, bound_form, attrition):
    model = model_for(data)
    out = {}

    uses = {"fun_videos_watched": "fun_videos_watched", "self_improvement_videos_watched":
            "self_improvement_videos_watched", "paid_work": "paid_work", "left_early": "left_early_minutes"}
    out["uses_of_time_minutes"] = {
        use: {"by_group": group_means(data, column), "contrasts": pair_contrasts(data, column, model)}
        for use, column in uses.items()}

    data = data.assign(left_early_share=data["left_early"].astype(float))
    out["leaving_early"] = {"by_group": group_shares(data, "left_early_share"),
                            "contrasts": pair_contrasts(data, "left_early_share", model)}

    out["labeled_minus_unlabeled_paid_work_minutes"] = contrast_on(data, "paid_work_minutes", "ai_saved_labeled",
                                                                   "ai_saved")

    covariate = contrast_on(data, "paid_work_minutes", "ai_saved", "found_break", covariate="fun_share")
    reference = statsmodels_contrasts(data.dropna(subset=["fun_share"]), " + fun_share", ["H2"])["H2"]
    if abs(reference["estimate"] - covariate["estimate"]) > 1e-8 or abs(reference["se"] - covariate["se"]) > 1e-8:
        raise RuntimeError("statsmodels and the numpy regression differ on H2 with the fun-share covariate")
    out["h2_with_fun_share_covariate"] = covariate
    # H2 among participants who stayed: a comparison of groups selected after assignment (leaving is itself an outcome).
    out["h2_among_participants_who_stayed"] = contrast_on(data[~data["left_early"]], "paid_work_minutes", "ai_saved",
                                                          "found_break") | {
        "label": "a comparison of groups selected after assignment"}

    by_country = {}
    for h, spec in rc.HYPOTHESES.items():
        rows = []
        for country in sorted(data["country"].unique()):
            a = data.loc[(data["country"] == country) & (data["group"] == spec["minuend"]), spec["outcome"]].dropna()
            b = data.loc[(data["country"] == country) & (data["group"] == spec["subtrahend"]), spec["outcome"]].dropna()
            se_c = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
            row = {"country": country, "estimate": float(a.mean() - b.mean()), "se": se_c, "n": int(len(a) + len(b))}
            if spec["outcome"] == "paid_work_minutes" and country in rc.PPP_TABLE:  # reported only
                row["value_of_one_item"] = rc.ppp_value_of_item(country)
            rows.append(row)
        usable = [r for r in rows if r["se"] > 0]
        summary = random_effects([r["estimate"] for r in usable], [r["se"] for r in usable])
        by_country[h] = {"countries": rows, "random_effects": {k: float(v) for k, v in summary.items()},
                         "countries_left_out_for_zero_variance": [r["country"] for r in rows if r["se"] == 0]}
    out["by_country"] = by_country

    out["weekly_ai_users"] = contrasts_with_intervals(data[data["ai_use"].isin(rc.WEEKLY_AI_USE_ANSWERS)])
    employed = data[data["employment_status"].isin(rc.PAID_EMPLOYMENT_ANSWERS)]
    out["in_paid_employment"] = {"tests": contrasts_with_intervals(employed),
                                 "employment_unknown": int((~data["employment_status"].isin(rc.EMPLOYMENT_ANSWERS)).sum())}

    def robustness(rows, hypotheses):
        tests = confirmatory_on(rows, bound_form)
        gated = apply_check_gates({h: t["reading"] for h, t in tests.items()}, checks_passed)
        out_ = {}
        for h in hypotheses:
            t = tests[h]
            lo, hi = interval(t["estimate"][0], t["se"][0], t["df"], rc.DIFFERENCE_INTERVAL_LEVEL)
            out_[h] = {"estimate": float(t["estimate"][0]), "lower": lo, "upper": hi, "n": int(t["n"]),
                       "reading": str(gated[h][0])}
        return out_

    dropped = hidden_page_dropped(data)
    kept_tests = robustness(data[~dropped], list(rc.HYPOTHESES))
    out["without_hidden_page"] = {"dropped": int(dropped.sum()), "n": int((~dropped).sum()), "tests": kept_tests,
                                  "pair_h3_h5": str(pair_reading(kept_tests["H1"]["reading"], kept_tests["H3"]["reading"],
                                                                 kept_tests["H5"]["reading"]))}
    watched = data[data["video_minutes"] > 0]
    out["fun_share_tests_among_participants_who_watched_a_video"] = {
        h: contrast_on(watched, spec["outcome"], spec["minuend"], spec["subtrahend"])
        for h, spec in rc.HYPOTHESES.items() if spec["outcome"] == "fun_share"}
    out["position_between_breaks"] = position_between_breaks(data)
    out["descriptives"] = optional_descriptives(data)
    five = data["made_all_five_picks"]
    out["among_participants_who_made_all_five_picks"] = {
        "dropped": int((~five).sum()), "n": int(five.sum()),
        "tests": robustness(data[five], list(rc.FIVE_PICKS_ROBUSTNESS_HYPOTHESES))}

    # Criterion d by country, reported with no gate: the pay and the videos may not mean
    # the same in every country.
    planned = data[data["group"] == rc.REFERENCE_GROUP]
    out["planned_group_floors_by_country"] = {
        country: {"n": int(len(rows)), "mean_fun_share": float(rows["fun_share"].mean()),
                  "share_of_free_minutes_to_paid_work": float(rows["paid_work_minutes"].sum()
                                                              / (len(rows) * rc.FREE_MINUTES)),
                  "value_of_one_item": rc.ppp_value_of_item(country) if country in rc.PPP_TABLE else None}
        for country, rows in planned.groupby("country")}

    data = data.assign(pick_not_finished=(data["picks_made"] < rc.VIDEO_PICKS).astype(float),
                       no_pick_made=(data["picks_made"] == 0).astype(float))
    out["choosing_not_finished_at_c_plus_4"] = {
        "pick_not_finished (some picks filled at random)": group_shares(data, "pick_not_finished"),
        "no_pick_made (no share of fun)": group_shares(data, "no_pick_made")}

    if attrition["threshold_crossed"]:
        bounds = {}
        for h, spec in rc.HYPOTHESES.items():
            ra = attrition["by_group"][spec["minuend"]]
            rb = attrition["by_group"][spec["subtrahend"]]
            bounds[h] = lee_bounds(data.loc[data["group"] == spec["minuend"], spec["outcome"]].dropna(),
                                   data.loc[data["group"] == spec["subtrahend"], spec["outcome"]].dropna(),
                                   ra["analyzed"] / ra["assigned"], rb["analyzed"] / rb["assigned"])
        out["lee_bounds"] = bounds
    return out


def answer_tables(data):
    """The full table of answers to the attribution and recognition questions, by group (shares)."""
    attribution = {g: {code: float((data.loc[data["group"] == g, "how_review_ended"] == code).mean())
                       for code in rc.ATTRIBUTION_ANSWERS} for g in rc.GROUPS}
    recognition = {g: {column: float(data.loc[data["group"] == g, column].mean()) for column in RECOGNITION_COLUMNS}
                   for g in rc.GROUPS}
    return attribution, recognition


# =================================================================================================================
# 7. One data set, start to finish
# =================================================================================================================
def analyze(raw, bound_form=None):
    bound_form = bound_form or rc.BOUND_FORM
    data, exclusions = apply_exclusions(raw)
    attrition = attrition_table(raw, data)
    model = model_for(data)

    checks = outcome_neutral_checks(model, data["fun_share"].to_numpy(), data["paid_work_minutes"].to_numpy(),
                                    data["surprise"].to_numpy(), data["says_ai_finished"].to_numpy(),
                                    data["says_not_reviewed"].to_numpy(), data["saw_label"].to_numpy())
    tests = confirmatory_on(data, bound_form)
    gated = apply_check_gates({h: t["reading"] for h, t in tests.items()}, checks["passed"])

    reference = statsmodels_contrasts(data)
    agreement = max(max(abs(reference[h]["estimate"] - float(tests[h]["estimate"][0])),
                        abs(reference[h]["se"] - float(tests[h]["se"][0]))) for h in rc.HYPOTHESES)
    if agreement > 1e-8:
        raise RuntimeError(f"statsmodels and the numpy regression differ by {agreement}")

    less_surprised = bool(checks["surprise_ai_saved_minus_found"]["ai_saved_less_surprised"][0])
    hypotheses = {}
    for h, spec in rc.HYPOTHESES.items():
        t = tests[h]
        est, se, df = float(t["estimate"][0]), float(t["se"][0]), t["df"]
        d_lo, d_hi = interval(est, se, df, rc.DIFFERENCE_INTERVAL_LEVEL)
        e_lo, e_hi = interval(est, se, df, rc.EQUIVALENCE_INTERVAL_LEVEL)
        failed = [c for c, hs in rc.CHECK_GATES.items() if h in hs and not bool(checks["passed"][c][0])]
        reading = str(gated[h][0])
        notes = []
        significant = float(t["p_two_sided"][0]) < rc.ALPHA
        equivalent = float(t["p_equivalence"][0]) < rc.ALPHA
        if significant and equivalent:
            notes.append("the difference is significant and also smaller than the bound d = 0.15")
        windfall_limit = (less_surprised and h in rc.SURPRISE_WINDFALL_LIMIT_HYPOTHESES
                          and str(t["reading"][0]) in rc.AGAINST_WINDFALL_READINGS[h])
        if windfall_limit:
            notes.append("the AI-saved group was less surprised than the found-break group, so this result is not "
                         "read as evidence against the windfall account")
        hypotheses[h] = {
            "question": spec["question"], "outcome": spec["outcome"], "unit": rc.OUTCOME_UNITS[spec["outcome"]],
            "contrast": f"{spec['minuend']} minus {spec['subtrahend']}",
            "estimate": est, "se": se, "df": float(df), "n": int(t["n"]),
            "difference_interval": [d_lo, d_hi], "difference_interval_level": rc.DIFFERENCE_INTERVAL_LEVEL,
            "p_two_sided": float(t["p_two_sided"][0]),
            "bound": float(t["bound"][0]), "bound_form": bound_form,
            "equivalence_interval": [e_lo, e_hi], "equivalence_interval_level": rc.EQUIVALENCE_INTERVAL_LEVEL,
            "p_equivalence": float(t["p_equivalence"][0]),
            "result": str(t["result"][0]),
            "reading_before_criteria": str(t["reading"][0]),
            "reading": reading,
            "account": spec["accounts"][reading],
            "not_interpretable_because": failed,
            "not_read_against_windfall": bool(windfall_limit),
            "notes": notes,
            "statsmodels_estimate": reference[h]["estimate"], "statsmodels_se": reference[h]["se"],
        }

    attribution_table, recognition_table = answer_tables(data)
    one = lambda v: float(np.asarray(v).reshape(-1)[0])  # noqa: E731
    passed = {c: bool(np.asarray(v).reshape(-1)[0]) for c, v in checks["passed"].items()}
    check_report = {
        "surprise": {grp: {k: (bool(np.asarray(v).reshape(-1)[0]) if k == "passed" else one(v)) for k, v in r.items()}
                     for grp, r in checks["surprise"].items()},
        "surprise_ai_saved_minus_found": {
            "difference": one(checks["surprise_ai_saved_minus_found"]["difference"]),
            "lower": one(checks["surprise_ai_saved_minus_found"]["lower"]),
            "upper": one(checks["surprise_ai_saved_minus_found"]["upper"]),
            "limit": one(checks["surprise_ai_saved_minus_found"]["limit"]),
            "ai_saved_less_surprised": less_surprised},
        "attribution": {"share_choosing_ai": {g: one(v) for g, v in checks["attribution"]["share_choosing_ai"].items()},
                        "share_choosing_not_reviewed": {g: one(v) for g, v in
                                                      checks["attribution"]["share_choosing_not_reviewed"].items()},
                        "table_by_group": attribution_table},
        "label": {k: one(v) for k, v in checks["label"].items()} | {"recognition_table_by_group": recognition_table},
        "fun_share_floor_ceiling": {"planned_break_mean": one(checks["fun_share_floor_ceiling"]["planned_break_mean"])},
        "paid_work_floor": {"planned_break_share_of_free_minutes":
                            one(checks["paid_work_floor"]["planned_break_share_of_free_minutes"])},
        "passed": passed,
        "watching_rule": {"participants_with_page_hidden_over_half_their_video_minutes":
                          int(hidden_page_dropped(data).sum())},
    }
    passed_arrays = {c: np.asarray(v) for c, v in checks["passed"].items()}
    three = (hypotheses["H1"]["reading"], hypotheses["H3"]["reading"], hypotheses["H5"]["reading"])
    pair = str(pair_reading(*three))
    pair_rule = int(pair_reading(*three, rule_numbers=True))
    return {
        "bound_form": bound_form,
        "alpha": rc.ALPHA,
        "exclusions": exclusions,
        "attrition": attrition,
        "checks": check_report,
        "hypotheses": hypotheses,
        "pair_h3_h5": {"reading": pair, "rule": pair_rule, "account": rc.PAIR_ACCOUNTS[pair],
                       "h1": hypotheses["H1"]["reading"],
                       "h3": hypotheses["H3"]["reading"], "h5": hypotheses["H5"]["reading"]},
        "statsmodels_agreement_max_difference": agreement,
        "secondary": secondary_analyses(data, passed_arrays, bound_form, attrition),
    }


# =================================================================================================================
# 8. Outputs
# =================================================================================================================
def _f(x, digits=2):
    return f"{x:.{digits}f}"


def _pct(level):
    return f"{100 * level:g}%"


def _status(ok, failing):
    return "passed" if ok else f"FAILED ({failing} not interpretable)"


def report_text(results, source):
    r, lines = results, []
    add = lines.append
    add(f"Registered analysis of {source}")
    add(f"Alpha for each test: {r['alpha']:.4f}; intervals {_pct(rc.DIFFERENCE_INTERVAL_LEVEL)} for a difference and "
        f"{_pct(rc.EQUIVALENCE_INTERVAL_LEVEL)} for equivalence. Equivalence bounds: {r['bound_form']} (d = "
        f"{rc.SMALLEST_EFFECT_D} times the pooled SD of the two groups compared).")
    add("")
    ex = r["exclusions"]
    add("1. Exclusions")
    add(f"   Rows read: {ex['rows_read']}. Failed the attention check before assignment: {ex['failed_attention_check']}."
        f" Assigned: {ex['assigned']}. Did not finish: {ex['did_not_finish']}. Withdrew at the debriefing: "
        f"{ex['withdrew']}. Analyzed: {ex['analyzed']}.")
    add("   Analyzed by group: " + ", ".join(f"{g} {n}" for g, n in ex["analyzed_by_group"].items())
        + f" (registered: at least {ex['registered_per_group']} a group).")
    add("   No one was excluded on any answer given after assignment.")
    fun_tests = [h for h, spec in rc.HYPOTHESES.items() if spec["outcome"] == "fun_share"]
    other_tests = [h for h in rc.HYPOTHESES if h not in fun_tests]
    add(f"   Made no pick, so no share of fun videos (left out of {', '.join(fun_tests[:-1])} and {fun_tests[-1]} only; "
        f"kept in {', '.join(other_tests)} and every other outcome): {ex['no_pick_made']} ("
        + ", ".join(f"{g} {n}" for g, n in ex["no_pick_made_by_group"].items()) + "). Some picks filled at random at "
        f"c + 4:00 (the share of fun counts only the picks made): {ex['some_picks_filled_at_random']}.")
    at = r["attrition"]
    add("   Did not finish or withdrew, share of those assigned: " + ", ".join(
        f"{g} {_f(100 * v['share_did_not_finish_or_withdrew'], 1)}%" for g, v in at["by_group"].items())
        + f"; largest gap {_f(100 * at['largest_gap_between_groups'], 1)} points (threshold "
        f"{_f(100 * at['threshold'], 0)}): " + ("CROSSED, Lee bounds below." if at["threshold_crossed"]
                                                 else "not crossed."))
    add("")
    ch = r["checks"]
    passed = ch["passed"]
    add("2. Outcome-neutral criteria")
    gate_names = {c: ", ".join(hs) for c, hs in rc.CHECK_GATES.items()}
    for grp, s in ch["surprise"].items():
        add(f"   (a) Surprise, {grp} minus planned_break: {_f(s['difference'])} scale points "
            f"[{_f(s['lower'])}, {_f(s['upper'])}], needs a lower end above {_f(s['minimum'])} (0.15 pooled SD): "
            f"{_status(s['passed'], gate_names['surprise_' + grp])}.")
    sm = ch["surprise_ai_saved_minus_found"]
    add(f"       AI-saved minus found break: {_f(sm['difference'])} [{_f(sm['lower'])}, {_f(sm['upper'])}]"
        + (f"; the interval lies below {_f(sm['limit'])} (minus 0.15 pooled SD): the AI-saved group was LESS surprised, "
           "so H3 and H5 are not read as evidence against the windfall account." if sm["ai_saved_less_surprised"]
           else f"; the interval does not lie below {_f(sm['limit'])} (minus 0.15 pooled SD)."))
    at_ = ch["attribution"]
    add("   (b) Attribution, share choosing \"" + rc.ATTRIBUTION_ANSWERS[rc.ATTRIBUTION_ANSWER_AI] + "\": "
        + ", ".join(f"{g} {_f(100 * v, 1)}%" for g, v in at_["share_choosing_ai"].items()) + ".")
    add(f"       AI-saved part: {_status(passed['attribution_ai_saved'], gate_names['attribution_ai_saved'])}. "
        f"Found-break part ({_f(100 * at_['share_choosing_not_reviewed']['found_break'], 1)}% of the found-break group "
        f"chose \"{rc.ATTRIBUTION_ANSWERS[rc.ATTRIBUTION_ANSWER_NOT_REVIEWED]}\"): "
        f"{_status(passed['attribution_found_break'], gate_names['attribution_found_break'])}.")
    add("       All answers by group: " + "; ".join(
        f"{g}: " + ", ".join(f"{code} {_f(100 * s, 1)}%" for code, s in row.items())
        for g, row in at_["table_by_group"].items()) + ".")
    lab = ch["label"]
    add(f"   (c) Label, \"That saved you 10 minutes.\" recognized, labeled minus unlabeled: "
        f"{_f(100 * lab['difference'], 1)} points [{_f(100 * lab['lower'], 1)}, {_f(100 * lab['upper'], 1)}], needs "
        f"at least {_f(100 * rc.LABEL_MIN_DIFFERENCE, 0)} points and a lower end above zero: "
        f"{_status(passed['label'], gate_names['label'])}.")
    add("       Sentences recognized by group: " + "; ".join(
        f"{g}: " + ", ".join(f"{c.replace('saw_', '')} {_f(100 * s, 1)}%" for c, s in row.items())
        for g, row in lab["recognition_table_by_group"].items()) + ".")
    fl = ch["fun_share_floor_ceiling"]
    add(f"   (d) Floor and ceiling, planned-break mean share of fun videos: {_f(fl['planned_break_mean'])} points: "
        f"{_status(passed['fun_share_floor_ceiling'], gate_names['fun_share_floor_ceiling'])}.")
    pw = ch["paid_work_floor"]
    add(f"       Floor, share of the planned-break group's free minutes given to paid extra work: "
        f"{_f(100 * pw['planned_break_share_of_free_minutes'], 1)}%: "
        f"{_status(passed['paid_work_floor'], gate_names['paid_work_floor'])}.")
    add(f"   Watching (robustness rule): {ch['watching_rule']['participants_with_page_hidden_over_half_their_video_minutes']}"
        f" participants had the page hidden for more than half of their video minutes; they are kept here and dropped "
        f"in the robustness check.")
    add("")
    add("3. Confirmatory tests")
    for h, t in r["hypotheses"].items():
        u = t["unit"]
        add(f"   {h}. {t['question']}")
        add(f"      {t['contrast']}, {t['outcome']} (n = {t['n']}): {_f(t['estimate'])} {u} (SE {_f(t['se'])}); "
            f"{_pct(t['difference_interval_level'])} interval [{_f(t['difference_interval'][0])}, "
            f"{_f(t['difference_interval'][1])}], two-sided p = {t['p_two_sided']:.4f}.")
        add(f"      Equivalence against plus or minus {_f(t['bound'])} {u}: {_pct(t['equivalence_interval_level'])} "
            f"interval [{_f(t['equivalence_interval'][0])}, {_f(t['equivalence_interval'][1])}], "
            f"p = {t['p_equivalence']:.4f}.")
        reason = (" (criteria failed: " + ", ".join(t["not_interpretable_because"]) + "; the reading before them was "
                  + t["reading_before_criteria"] + ")" if t["not_interpretable_because"] else "")
        add(f"      Reading: {t['reading'].upper()}{reason}. {t['account']}")
        for note in t["notes"]:
            add(f"      Note: {note}.")
    pr = r["pair_h3_h5"]
    add(f"   H3 and H5 read as a pair (H1 {pr['h1']}, H3 {pr['h3']}, H5 {pr['h5']}): rule {pr['rule']} of "
        f"Supplementary Table S4, {pr['reading'].upper()}. {pr['account']}")
    add(f"   statsmodels and the numpy regression agree to within {r['statsmodels_agreement_max_difference']:.1e}.")
    add("")
    sec = r["secondary"]
    add("4. Planned secondary analyses (95% intervals, no test)")
    add("   Minutes given to each of the four uses of the 10 minutes, by group:")
    for use, block in sec["uses_of_time_minutes"].items():
        add(f"      {use}: " + ", ".join(f"{g} {_f(e['mean'])} [{_f(e['lower'])}, {_f(e['upper'])}]"
                                         for g, e in block["by_group"].items()) + ".")
        add("         contrasts: " + "; ".join(f"{h} {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}]"
                                              for h, e in block["contrasts"].items()) + ".")
    le = sec["leaving_early"]
    add("   Share who left early, by group: " + ", ".join(
        f"{g} {_f(100 * e['share'], 1)}% [{_f(100 * e['lower'], 1)}, {_f(100 * e['upper'], 1)}]"
        for g, e in le["by_group"].items()) + ".")
    add("      contrasts (points): " + "; ".join(
        f"{h} {_f(100 * e['estimate'], 1)} [{_f(100 * e['lower'], 1)}, {_f(100 * e['upper'], 1)}]"
        for h, e in le["contrasts"].items()) + ".")
    m = sec["labeled_minus_unlabeled_paid_work_minutes"]
    add(f"   Minutes given to paid extra work, labeled minus unlabeled AI-saved: {_f(m['estimate'])} "
        f"[{_f(m['lower'])}, {_f(m['upper'])}].")
    m = sec["h2_with_fun_share_covariate"]
    add(f"   H2 with the participant's share of fun videos as a covariate: {_f(m['estimate'])} minutes "
        f"[{_f(m['lower'])}, {_f(m['upper'])}] (n = {m['n']}; the {ex['no_pick_made']} participants who made no pick "
        f"have no share and are left out of this analysis only).")
    m = sec["h2_among_participants_who_stayed"]
    add(f"   H2 among participants who stayed ({m['label']}): {_f(m['estimate'])} minutes "
        f"[{_f(m['lower'])}, {_f(m['upper'])}] (n = {m['n']}).")
    add("   By country, random-effects summary with its 95% interval and 95% prediction interval:")
    for h, bc in sec["by_country"].items():
        re_ = bc["random_effects"]
        add(f"      {h}: {_f(re_['summary'])} [{_f(re_['lower'])}, {_f(re_['upper'])}], prediction "
            f"[{_f(re_['prediction_lower'])}, {_f(re_['prediction_upper'])}], tau {_f(re_['tau'])}, "
            f"{int(re_['countries'])} countries.")
    add(f"   H2 by country, beside what one typing item ({rc.PPP_ITEM_USD:.2f} USD) buys there in international dollars "
        f"(World Bank PPP, {rc.PPP_YEAR}; reported only): " + "; ".join(
            f"{row['country']} {_f(row['estimate'])} minutes, {row['value_of_one_item']['international_dollars']:.3f}"
            for row in sec["by_country"]["H2"]["countries"] if "value_of_one_item" in row) + ".")
    add(f"   Participants who use generative AI weekly (1 to 6 days a week or every day; n = "
        f"{next(iter(sec['weekly_ai_users'].values()))['n']}):")
    for h, e in sec["weekly_ai_users"].items():
        add(f"      {h}: {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}].")
    emp = sec["in_paid_employment"]
    add(f"   Participants in paid employment (Prolific's export, full or part time; n = "
        f"{next(iter(emp['tests'].values()))['n']}; employment unknown for {emp['employment_unknown']}):")
    for h, e in emp["tests"].items():
        add(f"      {h}: {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}].")
    wh = sec["without_hidden_page"]
    add(f"   Robustness check without the {wh['dropped']} participants whose page was hidden for more than half of their "
        f"video minutes (n = {wh['n']}; {_pct(rc.DIFFERENCE_INTERVAL_LEVEL)} intervals):")
    for h, e in wh["tests"].items():
        add(f"      {h}: {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}], reading {e['reading']}.")
    add(f"      H3 and H5 as a pair: {wh['pair_h3_h5']}.")
    fp = sec["among_participants_who_made_all_five_picks"]
    add(f"   Robustness check among the {fp['n']} participants who made all five picks themselves "
        f"({fp['dropped']} left out; {_pct(rc.DIFFERENCE_INTERVAL_LEVEL)} intervals):")
    for h, e in fp["tests"].items():
        add(f"      {h}: {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}], reading {e['reading']}.")
    fc = sec["planned_group_floors_by_country"]
    fun_values = [v["mean_fun_share"] for v in fc.values()]
    work_values = [v["share_of_free_minutes_to_paid_work"] for v in fc.values()]
    add(f"   Criterion d by country, planned-break group (no gate): mean share of fun {_f(min(fun_values))} to "
        f"{_f(max(fun_values))} points; share of free minutes to paid extra work {_f(100 * min(work_values), 1)}% to "
        f"{_f(100 * max(work_values), 1)}%; countries outside the criterion's limits: "
        + (", ".join(c for c, v in fc.items() if not (rc.FUN_SHARE_FLOOR <= v["mean_fun_share"] <= rc.FUN_SHARE_CEILING)
                     or v["share_of_free_minutes_to_paid_work"] < rc.PAID_WORK_MIN_SHARE_OF_FREE_MINUTES) or "none")
        + ".")
    add("   Choosing not finished at c + 4:00, share by group: " + "; ".join(
        f"{step}: " + ", ".join(f"{g} {_f(100 * e['share'], 1)}%" for g, e in block.items())
        for step, block in sec["choosing_not_finished_at_c_plus_4"].items()) + ".")
    wv = sec["fun_share_tests_among_participants_who_watched_a_video"]
    add(f"   Fun-share contrasts among participants who watched at least one video (n = {next(iter(wv.values()))['n']}): "
        + "; ".join(f"{h} {_f(e['estimate'])} [{_f(e['lower'])}, {_f(e['upper'])}]" for h, e in wv.items()) + ".")
    pos = sec["position_between_breaks"]
    if pos["reported"]:
        add(f"   Where the AI-saved group lies between the breaks, (AI-saved minus planned) / (found minus planned): "
            f"{_f(pos['ratio'])} [{_f(pos['lower'])}, {_f(pos['upper'])}] ({pos['bootstrap_draws']} bootstrap draws "
            f"within group and country).")
    else:
        add(f"   Where the AI-saved group lies between the breaks: not reported ({pos['reason']}; found minus planned "
            f"{_f(pos['found_minus_planned'])} points, p = {pos['p_found_minus_planned']:.4f}).")
    ds = sec["descriptives"]

    def share_line(block, key="share"):
        if isinstance(block, str):
            return block
        return ", ".join(f"{g} {_f(100 * e[key]['share'], 1)}%" for g, e in block.items())

    add("   Flagged as possibly automated by Prolific's checks: " + share_line(ds["flagged_possibly_automated"]) + ".")
    pc = ds["purpose_names_reason_or_source"]
    add("   Purpose answers that name the reason the review ended or the source of the break: "
        + (pc if isinstance(pc, str) else share_line(pc["by_group"])
           + (f" ({len(pc['by_week'])} weeks reported separately)" if isinstance(pc["by_week"], dict) else "")) + ".")
    hb = ds["heard_of_study_before"]
    add("   Heard of the study before (yes): " + (hb if isinstance(hb, str) else share_line(hb["by_group"], "yes")) + ".")
    for column in rc.TIMING_COLUMNS + rc.REVIEW_EDIT_COLUMNS:
        block = ds[column]
        add(f"   {column}, mean by group: " + (block if isinstance(block, str) else ", ".join(
            f"{g} {_f(e['mean'])} ({e['missing']} with no value)" for g, e in block.items())) + ".")
    if "lee_bounds" in sec:
        add("   Lee bounds (attrition differs by more than 3 points between two groups):")
        for h, b in sec["lee_bounds"].items():
            add(f"      {h}: [{_f(b['lower'])}, {_f(b['upper'])}], {_f(100 * b['trimmed_share'], 1)}% trimmed.")
    return "\n".join(lines) + "\n"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def write_outputs(results, out_prefix, source):
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_name(out_prefix.name + ".json")
    text_path = out_prefix.with_name(out_prefix.name.replace("results_", "report_") + ".txt")
    json_path.write_text(json.dumps(jsonable({"source": str(source), **results}), indent=2), encoding="utf-8")
    text = report_text(results, source)
    text_path.write_text(text, encoding="utf-8")
    return json_path, text_path, text


def read_export(path):
    raw = pd.read_csv(path, keep_default_na=True)
    raw["group"] = raw["group"].fillna("")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description="The registered analysis of the experiment (Table 1).")
    parser.add_argument("data", type=Path, help="a file in the survey's export format (CSV)")
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="default: analysis/output/results_<name of the data file>")
    parser.add_argument("--bound-form", choices=rc.BOUND_FORMS, default=rc.BOUND_FORM)
    args = parser.parse_args(argv)
    results = analyze(read_export(args.data), args.bound_form)
    stem = args.data.stem.replace("simulated_", "")
    prefix = args.out_prefix or OUTPUT / f"results_{stem}"
    json_path, text_path, _ = write_outputs(results, prefix, args.data.as_posix())
    for h, t in results["hypotheses"].items():
        print(f"{h}: {t['reading']} (estimate {t['estimate']:.2f} {t['unit']}, two-sided p {t['p_two_sided']:.4f}, "
              f"equivalence p {t['p_equivalence']:.4f})")
    for h, t in results["secondary"]["without_hidden_page"]["tests"].items():
        print(f"{h}, robustness check without hidden pages: {t['reading']}")
    print(f"written: {json_path} and {text_path}")


if __name__ == "__main__":
    sys.exit(main())
