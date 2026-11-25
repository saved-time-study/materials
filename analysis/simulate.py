"""Draw a simulated data set with the structure that the survey will export: one row per participant who started.

Run from the repository's root:
    python analysis/simulate.py --all
    python analysis/simulate.py --pattern no_effects --seed 20261124 --out analysis/output/simulated_no_effects.csv

The columns follow the session's pages; analysis/README.md, "The data file", lists
them. In short: country; passed_attention_check; ai_use (asked before assignment); group (empty for anyone who failed
the attention check and so was never assigned); finished; withdrew (at the debriefing); employment_status (from
Prolific); the five videos in the order they would play (video_pick_1 to video_pick_5) with a flag for each pick
filled at random at c + 4:00 (pick_1_filled to pick_5_filled); the work choice (slot_1_use to slot_5_use: watch,
paid_work or left_early); left_early; fun_share (0 to 100, over the picks the participant made; empty if none);
paid_work_minutes (0 to 10, 2 a work slot); minutes_video_page_hidden (empty without video minutes); the two surprise
items (1 to 7); how_review_ended; and the six sentences of the recognition item (saw_..., True when chosen).

Each pattern of true effects is a set of steps: a group at step 1 differs from a group at step 0 by d = 0.15 (the
registered smallest effect), measured against the pooled within-country standard deviation of the two groups; a
fractional step gives a fraction of d. Leaving early is one choice that replaces all five slots, so a leaver gives 0
minutes to paid extra work; the calibration is made on the minutes of everyone, leavers included, so one step is
still d = 0.15. Every number that this file assumes is a field of Assumptions, so that a test can change it.
"""
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")  # numpy runs on one thread: nothing heavy on this computer

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import registered_constants as rc

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"
BASE_SEED = 20261124

ZERO = {g: 0 for g in rc.GROUPS}
PATTERNS = {
    "no_effects": {
        "description": "No true difference between any groups on either outcome.",
        "fun_share_steps": dict(ZERO),
        "paid_work_steps": dict(ZERO),
        "expected": {"H1": "equivalent", "H2": "equivalent", "H3": "supported", "H4": "equivalent",
                     "H5": "equivalent"},
        "power_claim": True,
    },
    "hypotheses_true_at_bound": {
        "description": "Each directional hypothesis true at the smallest effect: found break d = 0.15 more fun than "
                       "planned (H1), AI-saved d = 0.15 more minutes to paid work than found (H2), AI-saved as much fun "
                       "as planned (H3), labeled d = 0.15 more fun than unlabeled (H4, the prediction that a named "
                       "saving is treated as a gain), and so AI-saved d = 0.15 less fun than found (H5).",
        "fun_share_steps": {"planned_break": 0, "found_break": 1, "ai_saved": 0, "ai_saved_labeled": 1},
        "paid_work_steps": {"planned_break": 0, "found_break": 0, "ai_saved": 1, "ai_saved_labeled": 1},
        "expected": {"H1": "supported", "H2": "supported", "H3": "supported", "H4": "higher", "H5": "supported"},
        "power_claim": True,
    },
    "windfall_account": {
        "description": "AI-saved time is a windfall like found time: the found-break and both AI-saved groups pick "
                       "d = 0.15 more fun than the planned break, and every group gives the same minutes to paid "
                       "work.",
        "fun_share_steps": {"planned_break": 0, "found_break": 1, "ai_saved": 1, "ai_saved_labeled": 1},
        "paid_work_steps": dict(ZERO),
        "expected": {"H1": "supported", "H2": "equivalent", "H3": "higher than planned", "H4": "equivalent",
                     "H5": "equivalent"},
        "power_claim": True,
    },
    "work_account": {
        "description": "The work account, in which time that an AI saves still belongs to work: the found break picks "
                       "d = 0.15 more fun than the planned break, both AI-saved groups pick as much fun as the planned "
                       "break, and both give d = 0.15 more minutes to paid work than the found break.",
        "fun_share_steps": {"planned_break": 0, "found_break": 1, "ai_saved": 0, "ai_saved_labeled": 0},
        "paid_work_steps": {"planned_break": 0, "found_break": 0, "ai_saved": 1, "ai_saved_labeled": 1},
        "expected": {"H1": "supported", "H2": "supported", "H3": "supported", "H4": "equivalent", "H5": "supported"},
        "power_claim": True,
    },
    "halfway": {
        "description": "AI-saved time lies halfway between the two breaks: the found break picks d = 0.15 more fun "
                       "than the planned break, both AI-saved groups d = 0.075 more, and every group gives the same "
                       "minutes to paid work. Reported for its readings; the study is not powered for it.",
        "fun_share_steps": {"planned_break": 0, "found_break": 1, "ai_saved": 0.5, "ai_saved_labeled": 0.5},
        "paid_work_steps": dict(ZERO),
        "expected": {"H1": "supported", "H2": "equivalent", "H3": "higher than planned", "H4": "equivalent",
                     "H5": "supported"},
        "power_claim": False,
    },
}


# The pair reading of H3 and H5 that each pattern implies (registered_constants, "H3 and H5 read as a pair").
EXPECTED_PAIRS = {
    "no_effects": "cannot be told apart from either break",
    "hypotheses_true_at_bound": "spent like planned time",
    "windfall_account": "spent like found time",
    "work_account": "spent like planned time",
    "halfway": "between the two breaks",
}
for _name, _pair in EXPECTED_PAIRS.items():
    PATTERNS[_name]["expected_pair"] = _pair


@dataclass
class Assumptions:
    """What the simulation assumes about people. None of these numbers is registered; each is a stated guess."""
    effect_d: float = rc.SMALLEST_EFFECT_D  # size of one step between groups
    # Share of fun videos: each person has a leaning toward fun (a beta distribution) and picks five videos.
    fun_share_planned_mean: float = 50.0  # points, planned-break group, average over countries
    fun_share_within_country_sd: float = rc.FUN_SHARE_SD  # Chung et al.'s 29 points
    country_fun_share_offsets: tuple = tuple(np.linspace(-6.0, 6.0, len(rc.COUNTRIES)))  # points
    # Leaving early: one choice that replaces all five slots, with the same chance in every group.
    leave_early_rate: float = 0.15
    # Minutes given to paid extra work, among everyone (leavers give 0): a person gives none, or 1 to 5 slots.
    paid_work_within_country_sd: float = rc.PAID_WORK_MINUTES_SD  # minutes, groups at step 0, leavers included
    paid_work_extra_slot_probability: float = 0.5  # given any work: 1 slot plus Binomial(4, this)
    country_paid_work_offsets: tuple = tuple(np.linspace(-0.04, 0.04, len(rc.COUNTRIES)))  # in the chance of any
    # Criterion a: surprise, on a latent scale rounded to the 1 to 7 answers.
    surprise_mean: dict = field(default_factory=lambda: {
        "planned_break": 2.5, "found_break": 4.5, "ai_saved": 4.5, "ai_saved_labeled": 4.5})
    surprise_person_sd: float = 1.5
    surprise_item_sd: float = 0.7
    # Criterion b: the answers to "How did the review of the plan end?", in the order of rc.ATTRIBUTION_ANSWERS.
    attribution_shares: dict = field(default_factory=lambda: {
        "planned_break": (0.85, 0.04, 0.03, 0.08),
        "found_break": (0.07, 0.80, 0.05, 0.08),
        "ai_saved": (0.02, 0.04, 0.90, 0.04),
        "ai_saved_labeled": (0.02, 0.04, 0.90, 0.04),
    })
    # Criterion c and the recognition table: the chance that each group chooses each sentence.
    recognition_shares: dict = field(default_factory=lambda: {
        "saw_reviewed_three_sections": {"planned_break": 0.85, "found_break": 0.10, "ai_saved": 0.10,
                                        "ai_saved_labeled": 0.10},
        "saw_remaining_sections_not_reviewed": {"planned_break": 0.08, "found_break": 0.80, "ai_saved": 0.10,
                                                "ai_saved_labeled": 0.10},
        "saw_ai_assistant_reviewed_remaining": {"planned_break": 0.05, "found_break": 0.08, "ai_saved": 0.85,
                                                "ai_saved_labeled": 0.85},
        "saw_saved_you_10_minutes": {"planned_break": 0.05, "found_break": 0.08, "ai_saved": 0.20,
                                     "ai_saved_labeled": 0.80},
        "saw_review_saved_automatically": {g: 0.05 for g in rc.GROUPS},
        "saw_session_will_restart": {g: 0.05 for g in rc.GROUPS},
    })
    # The robustness rule: some participants hide the page for part of their video minutes.
    share_with_page_hidden: float = 0.20  # of those with video minutes; hidden time uniform over their video minutes
    # Before assignment, and from Prolific.
    ai_use_shares: dict = field(default_factory=lambda: {
        "never": 0.20, "less_than_weekly": 0.25, "one_to_six_days_a_week": 0.30, "every_day": 0.25})
    employment_shares: dict = field(default_factory=lambda: {
        "full_time": 0.40, "part_time": 0.15, "starting_new_job_within_a_month": 0.02, "unemployed_job_seeking": 0.10,
        "not_in_paid_work": 0.20, "other": 0.10, "": 0.03})  # "": expired in Prolific's export
    # Picks still missing at c + 4:00: some participants made one or two picks fewer than five, a few made none. They
    # never saw the slots, so all five slots play a video. Not used by the power analysis, which assumes five picks.
    picks_filled_rate: float = 0.02
    no_pick_rate: float = 0.003
    # Planned secondary measures with no hypothesis (optional columns of the export).
    automated_flag_rate: float = 0.01  # flagged as possibly automated by Prolific's checks
    automated_flag_missing_rate: float = 0.02  # no value in Prolific's export
    purpose_names_reason: dict = field(default_factory=lambda: {
        "planned_break": 0.10, "found_break": 0.20, "ai_saved": 0.20, "ai_saved_labeled": 0.20})
    heard_before_shares: dict = field(default_factory=lambda: {"yes": 0.02, "no": 0.93, "not_sure": 0.05})
    first_day: str = "2027-03-01"  # a stand-in date for the start of data collection
    collection_days: int = 42
    seconds_to_fifth_pick: tuple = (25.0, 8.0)  # mean and SD, among those who made five picks
    seconds_to_last_slot_change: tuple = (40.0, 9.0)  # among those who changed a slot (work or leave)
    review_edits_mean: float = 2.0
    characters_per_edit_mean: float = 8.0
    # Exclusions, replaced until each group has its registered number. The rates are of assigned participants.
    attention_fail_rate: float = 0.05
    not_finished_rate: dict = field(default_factory=lambda: {g: 0.03 for g in rc.GROUPS})
    withdraw_rate: dict = field(default_factory=lambda: {g: 0.01 for g in rc.GROUPS})


@dataclass
class Calibration:
    """Parameters solved from Assumptions so that the simulated outcomes have the stated standard deviations."""
    fun_share_concentration: float  # of the beta distribution of each person's leaning toward fun
    fun_share_step: float  # points added to the mean leaning for one step
    paid_work_base_rate: float  # chance of giving any slot to paid work, among everyone, groups at step 0
    paid_work_step: float  # added to that chance for one step


# ---------------------------------------------------------------------------------------------------------------
# Moments of the two outcomes, used to solve the calibration.
# ---------------------------------------------------------------------------------------------------------------
def fun_share_variance(mean_points, concentration):
    """Variance, in points squared, of 100 x k / 5, where k ~ beta-binomial(5) with the given mean and concentration."""
    mu = np.asarray(mean_points, dtype=float) / 100.0
    m = rc.VIDEO_PICKS
    return 1e4 * mu * (1 - mu) * (concentration + m) / (m * (concentration + 1))


def paid_work_moments(any_rate, extra_slot_probability):
    """Mean and variance, in minutes, of 2 x slots, where slots is 0, or 1 + Binomial(4, p) with chance any_rate."""
    q = np.asarray(any_rate, dtype=float)
    extra = rc.VIDEO_PICKS - 1
    slots_mean = 1 + extra * extra_slot_probability
    slots_second = extra * extra_slot_probability * (1 - extra_slot_probability) + slots_mean ** 2
    mean = rc.SLOT_MINUTES * q * slots_mean
    second = rc.SLOT_MINUTES ** 2 * q * slots_second
    return mean, second - mean ** 2


def calibrate(a: Assumptions) -> Calibration:
    offsets = np.asarray(a.country_fun_share_offsets)
    country_means = a.fun_share_planned_mean + offsets

    def sd_gap(concentration):
        return np.sqrt(fun_share_variance(country_means, concentration).mean()) - a.fun_share_within_country_sd

    if sd_gap(1e-6) < 0 or sd_gap(1e6) > 0:
        raise ValueError(f"a within-country SD of {a.fun_share_within_country_sd} points cannot be reached with five "
                         f"picks at a mean of {a.fun_share_planned_mean}")
    concentration = brentq(sd_gap, 1e-6, 1e6)

    def fun_d_gap(step):
        v0 = fun_share_variance(country_means, concentration)
        v1 = fun_share_variance(country_means + step, concentration)
        return step / np.sqrt(((v0 + v1) / 2).mean()) - a.effect_d

    fun_step = brentq(fun_d_gap, 0.0, 45.0) if a.effect_d > 0 else 0.0

    work_offsets = np.asarray(a.country_paid_work_offsets)
    p = a.paid_work_extra_slot_probability
    slots_mean = 1 + (rc.VIDEO_PICKS - 1) * p
    slots_second = (rc.VIDEO_PICKS - 1) * p * (1 - p) + slots_mean ** 2
    peak = slots_second / (2 * slots_mean ** 2)  # the chance of any work at which the variance is largest
    ceiling = 1.0 - a.leave_early_rate  # leavers cannot give slots to work
    low, high = 1e-6 - work_offsets.min(), min(peak, ceiling) - work_offsets.max()

    def work_sd_gap(base):
        return np.sqrt(paid_work_moments(base + work_offsets, p)[1].mean()) - a.paid_work_within_country_sd

    if work_sd_gap(low) > 0 or work_sd_gap(high) < 0:
        raise ValueError(f"a within-country SD of {a.paid_work_within_country_sd} minutes cannot be reached with an "
                         f"extra-slot probability of {p} and {a.leave_early_rate:.0%} leaving early")
    base = brentq(work_sd_gap, low, high)

    def work_d_gap(step):
        m0, v0 = paid_work_moments(base + work_offsets, p)
        m1, v1 = paid_work_moments(base + work_offsets + step, p)
        return (m1 - m0).mean() / np.sqrt(((v0 + v1) / 2).mean()) - a.effect_d

    work_step = brentq(work_d_gap, 0.0, ceiling - base - work_offsets.max() - 1e-6) if a.effect_d > 0 else 0.0
    return Calibration(concentration, fun_step, base, work_step)


# ---------------------------------------------------------------------------------------------------------------
# The registered allocation of analyzable participants.
# ---------------------------------------------------------------------------------------------------------------
def allocation(per_group_by_country=None, scale=1):
    """Group and country index of every analyzable participant: groups filled evenly within each country."""
    per_group_by_country = per_group_by_country or rc.PARTICIPANTS_PER_GROUP_BY_COUNTRY
    group_index, country_index = [], []
    for c, country in enumerate(rc.COUNTRIES):
        n = per_group_by_country[country] * scale
        for g in range(len(rc.GROUPS)):
            group_index.append(np.full(n, g))
            country_index.append(np.full(n, c))
    return np.concatenate(group_index), np.concatenate(country_index)


def _by_group(values, group_index):
    return np.array([values[g] for g in rc.GROUPS])[group_index][:, None]


# ---------------------------------------------------------------------------------------------------------------
# The core draw: many data sets at once, one column per data set.
# ---------------------------------------------------------------------------------------------------------------
def draw_outcomes(rng, group_index, country_index, draws, pattern, a: Assumptions, cal: Calibration):
    """Outcomes and the answers that the criteria use, as arrays of shape (participants, draws)."""
    n = len(group_index)
    fun_steps = np.array([pattern["fun_share_steps"][g] for g in rc.GROUPS], dtype=float)[group_index]
    work_steps = np.array([pattern["paid_work_steps"][g] for g in rc.GROUPS], dtype=float)[group_index]

    mean_points = (a.fun_share_planned_mean + np.asarray(a.country_fun_share_offsets)[country_index]
                   + fun_steps * cal.fun_share_step)
    mu = (mean_points / 100.0)[:, None]
    leaning = rng.beta(mu * cal.fun_share_concentration, (1 - mu) * cal.fun_share_concentration, size=(n, draws))
    fun_count = rng.binomial(rc.VIDEO_PICKS, leaning)
    fun_share = fun_count * (100.0 / rc.VIDEO_PICKS)

    left_early = rng.random((n, draws)) < a.leave_early_rate
    any_rate = (cal.paid_work_base_rate + np.asarray(a.country_paid_work_offsets)[country_index]
                + work_steps * cal.paid_work_step)[:, None]
    gives_any = ~left_early & (rng.random((n, draws)) < any_rate / (1.0 - a.leave_early_rate))
    extra = rng.binomial(rc.VIDEO_PICKS - 1, a.paid_work_extra_slot_probability, size=(n, draws))
    paid_work_slots = np.where(gives_any, 1 + extra, 0)
    paid_work_minutes = paid_work_slots * float(rc.SLOT_MINUTES)

    person = _by_group(a.surprise_mean, group_index) + rng.normal(0.0, a.surprise_person_sd, size=(n, draws))
    lo, hi = rc.SURPRISE_RANGE
    surprise_unexpected = np.clip(np.rint(person + rng.normal(0.0, a.surprise_item_sd, size=(n, draws))), lo, hi)
    surprise_surprising = np.clip(np.rint(person + rng.normal(0.0, a.surprise_item_sd, size=(n, draws))), lo, hi)

    cumulative = np.cumsum(np.array([a.attribution_shares[g] for g in rc.GROUPS]), axis=1)[group_index]
    u = rng.random((n, draws))
    attribution_index = (u[:, :, None] >= cumulative[:, None, :-1]).sum(axis=2).astype(np.int8)
    codes = list(rc.ATTRIBUTION_ANSWERS)
    says_ai = attribution_index == codes.index(rc.ATTRIBUTION_ANSWER_AI)
    says_not_reviewed = attribution_index == codes.index(rc.ATTRIBUTION_ANSWER_NOT_REVIEWED)

    saw_label = rng.random((n, draws)) < _by_group(a.recognition_shares[rc.LABEL_SENTENCE], group_index)

    return {
        "fun_count": fun_count, "fun_share": fun_share, "left_early": left_early,
        "paid_work_slots": paid_work_slots, "paid_work_minutes": paid_work_minutes,
        "surprise_unexpected": surprise_unexpected, "surprise_surprising": surprise_surprising,
        "attribution_index": attribution_index, "says_ai": says_ai, "says_not_reviewed": says_not_reviewed,
        "saw_label": saw_label,
    }


# ---------------------------------------------------------------------------------------------------------------
# One data set in the survey's export format.
# ---------------------------------------------------------------------------------------------------------------
def simulate_dataset(pattern_name, seed, assumptions=None, per_group_by_country=None, pattern=None):
    a = assumptions or Assumptions()
    cal = calibrate(a)
    pattern = pattern or PATTERNS[pattern_name]
    rng = np.random.default_rng(seed)
    group_index, country_index = allocation(per_group_by_country)
    n = len(group_index)
    d = {k: v[:, 0] for k, v in draw_outcomes(rng, group_index, country_index, 1, pattern, a, cal).items()}

    # Picks missing at c + 4:00 (filled at random; they play but do not count). Those participants never saw the
    # slots, so they cannot give slots to work or leave.
    u = rng.random(n)
    n_filled = np.where(u < a.no_pick_rate, rc.VIDEO_PICKS,
                        np.where(u < a.no_pick_rate + a.picks_filled_rate, rng.integers(1, 3, size=n), 0))
    left_early = d["left_early"] & (n_filled == 0)
    paid_work_slots = np.where(n_filled == 0, d["paid_work_slots"], 0)

    # The five picks, in the order they would play; a work slot replaces the pick in its position.
    picks = np.empty((n, rc.VIDEO_PICKS), dtype=object)
    filled = np.zeros((n, rc.VIDEO_PICKS), dtype=bool)
    slot_use = np.empty((n, rc.VIDEO_PICKS), dtype=object)
    for i in range(n):
        k = int(d["fun_count"][i])
        chosen = list(rng.choice(rc.FUN_VIDEOS, size=k, replace=False)) + list(
            rng.choice(rc.SELF_IMPROVEMENT_VIDEOS, size=rc.VIDEO_PICKS - k, replace=False))
        picks[i] = rng.permutation(np.array(chosen, dtype=object))
        filled[i, rng.permutation(rc.VIDEO_PICKS)[: int(n_filled[i])]] = True
        if left_early[i]:
            slot_use[i] = "left_early"
        else:
            slot_use[i] = "watch"
            slot_use[i, rng.permutation(rc.VIDEO_PICKS)[: int(paid_work_slots[i])]] = "paid_work"
    is_fun = np.isin(picks, rc.FUN_VIDEOS)
    made = (~filled).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fun_share = np.where(made > 0, 100.0 * (is_fun & ~filled).sum(axis=1) / np.maximum(made, 1), np.nan)

    video_minutes = (slot_use == "watch").sum(axis=1) * float(rc.SLOT_MINUTES)
    hides = rng.random(n) < a.share_with_page_hidden
    minutes_hidden = np.where(hides, np.floor(rng.random(n) * video_minutes * 10) / 10, 0.0)
    minutes_hidden = np.where(video_minutes > 0, minutes_hidden, np.nan)  # no value without video minutes

    recognition = {}
    for column, shares in a.recognition_shares.items():
        if column == rc.LABEL_SENTENCE:
            recognition[column] = d["saw_label"]
        else:
            recognition[column] = rng.random(n) < np.array([shares[g] for g in rc.GROUPS])[group_index]

    ai_use = rng.choice(list(a.ai_use_shares), size=n, p=list(a.ai_use_shares.values()))
    employment = rng.choice(list(a.employment_shares), size=n, p=list(a.employment_shares.values()))

    automated = np.where(rng.random(n) < a.automated_flag_missing_rate, None,
                         rng.random(n) < a.automated_flag_rate).astype(object)
    purpose = rng.random(n) < np.array([a.purpose_names_reason[g] for g in rc.GROUPS])[group_index]
    heard = rng.choice(list(a.heard_before_shares), size=n, p=list(a.heard_before_shares.values()))
    started = (pd.Timestamp(a.first_day) + pd.to_timedelta(rng.random(n) * a.collection_days, unit="D")).strftime(
        "%Y-%m-%dT%H:%M:%S")
    m, s = a.seconds_to_fifth_pick
    fifth_pick = np.where(n_filled == 0, np.clip(np.round(rng.normal(m, s, n), 1), 3.0, 60.0), np.nan)
    changed = (left_early | (paid_work_slots > 0))
    m, s = a.seconds_to_last_slot_change
    slot_change = np.where(changed, np.clip(np.round(np.maximum(rng.normal(m, s, n), fifth_pick + 1.0), 1), 4.0, 60.0),
                           np.nan)
    edits = rng.poisson(a.review_edits_mean, n)
    characters = np.where(edits > 0, rng.poisson(a.characters_per_edit_mean * np.maximum(edits, 1)), 0)

    analyzable = pd.DataFrame({
        "country": np.array(rc.COUNTRIES, dtype=object)[country_index],
        "passed_attention_check": True,
        "ai_use": ai_use,
        "group": np.array(rc.GROUPS, dtype=object)[group_index],
        "finished": True,
        "withdrew": False,
        "employment_status": np.where(employment == "", None, employment),
        **{f"video_pick_{j + 1}": picks[:, j] for j in range(rc.VIDEO_PICKS)},
        **{column: filled[:, j] for j, column in enumerate(rc.PICK_FILLED_COLUMNS)},
        **{f"slot_{j + 1}_use": slot_use[:, j] for j in range(rc.VIDEO_PICKS)},
        "left_early": left_early,
        "fun_share": fun_share,
        "paid_work_minutes": paid_work_slots * float(rc.SLOT_MINUTES),
        "minutes_video_page_hidden": minutes_hidden,
        "surprise_unexpected": d["surprise_unexpected"].astype(int),
        "surprise_surprising": d["surprise_surprising"].astype(int),
        "how_review_ended": np.array(list(rc.ATTRIBUTION_ANSWERS), dtype=object)[d["attribution_index"]],
        **recognition,
        rc.AUTOMATED_FLAG_COLUMN: automated,
        rc.PURPOSE_CODED_COLUMN: purpose,
        rc.HEARD_BEFORE_COLUMN: heard,
        rc.STARTED_AT_COLUMN: started,
        "seconds_to_fifth_pick": fifth_pick,
        "seconds_to_last_slot_change": slot_change,
        "review_edits": edits,
        "review_characters_changed": characters,
    })

    # People who were replaced: they failed the attention check before assignment (no group), did not finish, or
    # withdrew their answers at the debriefing (their answers are not in the export).
    blank = {c: np.nan for c in analyzable.columns if c not in ("country", "passed_attention_check", "group",
                                                                  "finished", "withdrew")}
    frames = [analyzable]
    n_attention = int(round(a.attention_fail_rate * n))
    if n_attention:
        frames.append(pd.DataFrame({"country": rng.choice(rc.COUNTRIES, size=n_attention),
                                    "passed_attention_check": False, "group": "", "finished": False,
                                    "withdrew": False, **blank}))
    for k, group in enumerate(rc.GROUPS):
        n_group = int((group_index == k).sum())
        for finished, withdrew, rate in ((False, False, a.not_finished_rate[group]),
                                         (True, True, a.withdraw_rate[group])):
            count = int(rng.poisson(rate / (1 - rate) * n_group)) if rate > 0 else 0
            if count:
                frames.append(pd.DataFrame({"country": rng.choice(rc.COUNTRIES, size=count),
                                            "passed_attention_check": True, "group": group, "finished": finished,
                                            "withdrew": withdrew, **blank}))
    data = pd.concat(frames, ignore_index=True)
    data = data.iloc[rng.permutation(len(data))].reset_index(drop=True)
    data.insert(0, "participant_id", [f"P{i + 1:05d}" for i in range(len(data))])
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="no_effects")
    parser.add_argument("--seed", type=int, default=None, help="default: 20261124 plus the pattern's position")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="write one file for each pattern")
    args = parser.parse_args(argv)
    names = list(PATTERNS) if args.all else [args.pattern]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        seed = args.seed if (args.seed is not None and not args.all) else BASE_SEED + list(PATTERNS).index(name)
        out = args.out if (args.out and not args.all) else OUTPUT / f"simulated_{name}.csv"
        data = simulate_dataset(name, seed)
        data.to_csv(out, index=False)
        analyzable = int((data.passed_attention_check & data.finished.astype(bool) & ~data.withdrew.astype(bool)).sum())
        print(f"{name}: seed {seed}, {len(data)} rows ({analyzable} analyzable), written to {out}")


if __name__ == "__main__":
    main()
