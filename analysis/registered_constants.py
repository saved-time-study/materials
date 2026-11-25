"""Every registered number of the experiment, in one place.

Sources: Table 1 of the Stage 1 protocol, the session's pages (materials/SESSION-PAGES.md) and the sample-size
script (power/design_table_power.py). A number that none of them states is marked ASSUMED, with the reason.

Nothing in simulate.py, analyze.py or power_by_simulation.py repeats a number that is defined here.
"""

# ---------------------------------------------------------------------------------------------------------------
# The four groups. They differ only in how the 10 free minutes arrive.
# ---------------------------------------------------------------------------------------------------------------
GROUPS = ("planned_break", "found_break", "ai_saved", "ai_saved_labeled")
GROUP_DESCRIPTIONS = {
    "planned_break": "Planned break: told at the start that a 10-minute break follows a 3-part review.",
    "found_break": "Found break: told the review has 13 sections; after 3 the remaining sections will not be reviewed.",
    "ai_saved": "AI-saved time: told the review has 13 sections; after 3 the AI assistant has reviewed the rest itself.",
    "ai_saved_labeled": 'AI-saved time, labeled: the same, with the line "That saved you 10 minutes."',
}
REFERENCE_GROUP = "planned_break"

# ---------------------------------------------------------------------------------------------------------------
# Sample and countries.
# ---------------------------------------------------------------------------------------------------------------
PARTICIPANTS_PER_GROUP = 1706  # analyzable participants a group; 6,824 in all (design_table_power_output.txt)
COUNTRIES = (
    "United States", "United Kingdom", "Canada", "Australia", "South Africa", "Mexico",
    "Chile", "Germany", "Poland", "Portugal", "Italy", "Spain",
)
# ASSUMED: 1,706 is not a multiple of 12, so the simulation gives 143 a group to the first two countries and 142 a
# group to the other ten (2 x 143 + 10 x 142 = 1,706), about 569 participants a country. The real split will follow
# the Prolific pool of each country; recruitment in a country stops when each of its groups reaches its share.
PARTICIPANTS_PER_GROUP_BY_COUNTRY = {c: (143 if i < 2 else 142) for i, c in enumerate(COUNTRIES)}

# ---------------------------------------------------------------------------------------------------------------
# Tests: five confirmatory tests, Bonferroni at a familywise .05.
# ---------------------------------------------------------------------------------------------------------------
FAMILYWISE_ALPHA = 0.05
NUMBER_OF_TESTS = 5
ALPHA = FAMILYWISE_ALPHA / NUMBER_OF_TESTS  # .01, two-sided for a difference, and for each one-sided test of TOST
DIFFERENCE_INTERVAL_LEVEL = 1 - ALPHA  # 99 percent: the interval that matches the two-sided test at .01
EQUIVALENCE_INTERVAL_LEVEL = 1 - 2 * ALPHA  # 98 percent: the interval that matches two one-sided tests at .01
ROBUST_ERRORS = "HC2"  # Table 1 says "errors robust to unequal variances"; HC2 gives Welch's error for two groups

# The smallest effect of interest and the equivalence bounds.
SMALLEST_EFFECT_D = 0.15
FUN_SHARE_SD = 29.0  # points; Chung et al. (2023, study 5); used only for the bound's expected value in points
FUN_SHARE_BOUND = SMALLEST_EFFECT_D * FUN_SHARE_SD  # 4.35 points, the expected bound at an SD of 29
# ASSUMED: no published study gives the standard deviation of the minutes given to paid extra work. 3.0 minutes is a
# guess (a 0 to 10 count in steps of 2 minutes, most people giving none); it sets the simulation and the expected bound.
PAID_WORK_MINUTES_SD = 3.0
PAID_WORK_MINUTES_BOUND = SMALLEST_EFFECT_D * PAID_WORK_MINUTES_SD  # 0.45 minutes, the expected bound at an SD of 3
FIXED_BOUNDS = {"fun_share": FUN_SHARE_BOUND, "paid_work_minutes": PAID_WORK_MINUTES_BOUND}
# How the equivalence bound is set. "observed_sd" (registered): d = 0.15 times the pooled standard deviation of the
# two groups compared, computed from the raw values of the two groups (so it includes the differences between
# countries). "fixed": the values in FIXED_BOUNDS, reported by power_by_simulation.py for comparison only.
BOUND_FORMS = ("fixed", "observed_sd")
BOUND_FORM = "observed_sd"  # registered

# What Table 1 claims, from power/design_table_power.py (normal theory, 1,706 a group, alpha .01).
CLAIMED_POWER_DIFFERENCE = 0.964  # to detect d = 0.15, two-sided at .01
CLAIMED_POWER_EQUIVALENCE = 0.960  # two one-sided tests at .01 when the true difference is zero
TARGET_POWER = 0.96  # the power that each test is planned to reach
REQUIRED_POWER = 0.95  # Nature Human Behaviour's rule for every confirmatory test

# ---------------------------------------------------------------------------------------------------------------
# The readings. Each test first gives a result from its two-sided test and its two one-sided tests:
#   higher      the two-sided test is significant at .01 and the minuend is above the subtrahend;
#   lower       the two-sided test is significant at .01 and the minuend is below the subtrahend;
#   equivalent  both one-sided tests are significant at .01 (the 98 percent interval lies inside the bounds);
#   unresolved  neither.
# A "difference" hypothesis reads significance first, and the "equivalence" hypothesis (H3) reads equivalence first;
# at the registered sample both cannot happen at once (design_table_power_output.txt, last lines). Each hypothesis
# then names each result with one word from READINGS. A failed outcome-neutral criterion replaces the reading with
# "not interpretable" (the estimate and its intervals are still reported).
# ---------------------------------------------------------------------------------------------------------------
RESULTS = ("higher", "lower", "equivalent", "unresolved")
NOT_INTERPRETABLE = "not interpretable"
READINGS = ("supported", "opposite", "equivalent", "higher", "lower", "higher than planned", "lower than planned",
            "unresolved", NOT_INTERPRETABLE)

HYPOTHESES = {
    "H1": {"outcome": "fun_share", "minuend": "found_break", "subtrahend": "planned_break", "kind": "difference",
           "question": "Do people spend a found break on more fun than a planned break?",
           "readings": {"higher": "supported", "lower": "opposite", "equivalent": "equivalent"},
           "accounts": {
               "supported": "A found break goes to more fun than a planned break: the found-time effect of Chung and "
                            "colleagues, in the pooled sample of 12 countries.",
               "opposite": "A found break goes to less fun than a planned break; no account in the protocol predicts "
                           "this.",
               "equivalent": "A found break and a planned break go to the same share of fun, within the bound.",
           }},
    "H2": {"outcome": "paid_work_minutes", "minuend": "ai_saved", "subtrahend": "found_break", "kind": "difference",
           "question": "Do people give AI-saved time to paid work more, or less, than found time?",
           "readings": {"higher": "supported", "lower": "lower", "equivalent": "equivalent"},
           "accounts": {
               "supported": "The work account: time saved when an assistant completes the work goes back to paid work "
                            "more than time freed because the work was dropped.",
               "lower": "The goal-progress account: a finished goal licenses leisure, so time saved when the work is "
                        "completed goes to paid work less than time freed because the work was dropped.",
               "equivalent": "AI-saved and found time go to paid work alike, within the bound, as the windfall account "
                             "predicts.",
           }},
    "H3": {"outcome": "fun_share", "minuend": "ai_saved", "subtrahend": "planned_break", "kind": "equivalence",
           "question": "Do people spend AI-saved time on fun like a planned break?",
           "readings": {"equivalent": "supported", "higher": "higher than planned", "lower": "lower than planned"},
           "accounts": {
               "supported": "AI-saved time goes to fun like planned time, within the bound, as the work account "
                            "predicts.",
               "higher than planned": "AI-saved time goes to more fun than planned time, as the windfall account "
                                      "predicts; H5 shows whether it goes to as much fun as found time.",
               "lower than planned": "AI-saved time goes to less fun than even a planned break, as a strong form of "
                                     "the work account predicts.",
           }},
    "H4": {"outcome": "fun_share", "minuend": "ai_saved_labeled", "subtrahend": "ai_saved", "kind": "difference",
           "question": "Does labeling the saving change the share of fun videos, and in which direction?",
           "readings": {"higher": "higher", "lower": "lower", "equivalent": "equivalent"},
           "accounts": {
               "higher": "A named saving is treated as a gain: the label raises the share of fun.",
               "lower": "The label reminds people of the work: it lowers the share of fun.",
               "equivalent": "The label does not change the share of fun, within the bound.",
           }},
    "H5": {"outcome": "fun_share", "minuend": "ai_saved", "subtrahend": "found_break", "kind": "difference",
           "question": "Do people spend AI-saved time on less fun than found time?",
           "readings": {"lower": "supported", "higher": "opposite", "equivalent": "equivalent"},
           "accounts": {
               "supported": "The work account: AI-saved time goes to less fun than found time.",
               "opposite": "AI-saved time goes to more fun than found time, as the goal-progress account predicts.",
               "equivalent": "The windfall account: AI-saved time goes to fun like found time, within the bound.",
           }},
}
for _spec in HYPOTHESES.values():
    _spec["readings"]["unresolved"] = "unresolved"
    _spec["accounts"]["unresolved"] = "Neither test is conclusive; the data cannot tell."
    _spec["accounts"][NOT_INTERPRETABLE] = ("An outcome-neutral criterion tied to this test failed, so the test is "
                                            "not read; its estimate and intervals are reported.")
OUTCOMES = ("fun_share", "paid_work_minutes")
OUTCOME_UNITS = {"fun_share": "points", "paid_work_minutes": "minutes"}

# ---------------------------------------------------------------------------------------------------------------
# The session.
# ---------------------------------------------------------------------------------------------------------------
FREE_MINUTES = 10
SLOT_MINUTES = 2
VIDEO_PICKS = 5  # five picks, which also fill the five 2-minute slots of the break
# Picks still missing at c + 4:00 are filled at random so that the slots have videos to play; a filled pick is flagged
# (pick_j_filled) and does not count: the share of fun videos is computed over the picks the participant made, and a
# participant who made no pick has no value for it (and keeps every other outcome).
PICK_FILLED_COLUMNS = tuple(f"pick_{j + 1}_filled" for j in range(VIDEO_PICKS))
FUN_VIDEOS = ("fun_1", "fun_2", "fun_3", "fun_4", "fun_5")
SELF_IMPROVEMENT_VIDEOS = (
    "self_improvement_1", "self_improvement_2", "self_improvement_3", "self_improvement_4", "self_improvement_5",
)
# The work choice: each slot is watched or given to paid extra work, or the participant leaves now, which replaces all
# five slots: a leaver's five slots are recorded as left_early and the minutes given to paid
# extra work are 0.
SLOT_USES = ("watch", "paid_work", "left_early")
# The four uses of the 10 minutes, a planned secondary outcome (minutes each, summing to 10 for every participant).
USES_OF_TIME = ("fun_videos_watched", "self_improvement_videos_watched", "paid_work", "left_early")

# Before assignment: use of generative AI (asked of everyone after the attention check, before the click on Start).
AI_USE_ANSWERS = {  # export code: the option on the screen
    "never": "Never",
    "less_than_weekly": "Less than once a week",
    "one_to_six_days_a_week": "1 to 6 days a week",
    "every_day": "Every day",
}
WEEKLY_AI_USE_ANSWERS = ("one_to_six_days_a_week", "every_day")  # "weekly" users: the two highest options

# Employment, from Prolific's demographic export (not asked in the session). ASSUMED: the export's labels below are
# Prolific's employment-status options as we know them; they must be checked against the real export, and a value
# that Prolific has let expire is read as unknown (outside the subgroup in paid employment, never excluded).
EMPLOYMENT_ANSWERS = {
    "full_time": "Full-Time",
    "part_time": "Part-Time",
    "starting_new_job_within_a_month": "Due to start a new job within the next month",
    "unemployed_job_seeking": "Unemployed (and job seeking)",
    "not_in_paid_work": "Not in paid work (e.g. homemaker, retired or disabled)",
    "other": "Other",
}
PAID_EMPLOYMENT_ANSWERS = ("full_time", "part_time")

# Question 4: "How did the review of the plan end?" (materials/SESSION-PAGES.md)
ATTRIBUTION_ANSWERS = {  # export code: the option on the screen
    "as_planned_after_three_sections": "As planned, after three sections",
    "remaining_sections_would_not_be_reviewed": "The remaining sections would not be reviewed",
    "ai_assistant_reviewed_remaining_sections": "The AI assistant reviewed the remaining sections",
    "do_not_know": "I don't know",
}
ATTRIBUTION_ANSWER_AI = "ai_assistant_reviewed_remaining_sections"
ATTRIBUTION_ANSWER_NOT_REVIEWED = "remaining_sections_would_not_be_reviewed"

# Question 5: "Which of these sentences did you see during the session? Choose all that apply." One column a
# sentence, True when chosen. The last two were seen by no group.
RECOGNITION_SENTENCES = {
    "saw_reviewed_three_sections": "You have reviewed the three sections.",
    "saw_remaining_sections_not_reviewed": "The remaining sections of the plan will not be reviewed in this session.",
    "saw_ai_assistant_reviewed_remaining": "The AI assistant has reviewed the remaining sections of the plan itself.",
    "saw_saved_you_10_minutes": "That saved you 10 minutes.",
    "saw_review_saved_automatically": "Your review was saved automatically.",
    "saw_session_will_restart": "The session will restart in one minute.",
}
LABEL_SENTENCE = "saw_saved_you_10_minutes"
SURPRISE_RANGE = (1, 7)

# ---------------------------------------------------------------------------------------------------------------
# The outcome-neutral criteria and the tests that each makes not
# interpretable when it fails.
# ---------------------------------------------------------------------------------------------------------------
CHECK_INTERVAL_LEVEL = 0.95
# (a) Surprise: the mean of Chung et al.'s two items. Each of these groups rates the break as more unexpected than the
# planned group, with the 95 percent interval of the difference above SURPRISE_MIN_D times the pooled SD of the two
# groups' surprise scores.
SURPRISE_GROUPS = ("found_break", "ai_saved", "ai_saved_labeled")
SURPRISE_MIN_D = 0.15
# (b) Attribution, on the observed shares (no interval).
ATTRIBUTION_MIN_SHARE_AI_GROUPS = 0.80  # at least 80 percent of each AI-saved group choose the assistant
ATTRIBUTION_MAX_SHARE_FOUND_GROUP = 0.20  # at most 20 percent of the found-break group choose the assistant
ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED = 0.50  # at least 50 percent of the found-break group choose "would not be
#                                                  reviewed"
# (c) The label: the labeled group chooses "That saved you 10 minutes." at least 20 percentage points more often than
# the unlabeled AI-saved group, and the 95 percent interval of the difference lies above zero.
LABEL_MIN_DIFFERENCE = 0.20
# (d) Floor and ceiling, in the planned-break group: its mean share of fun videos lies between 15 and 85 percent, and
# at least 5 percent of its free minutes (10 minutes for every participant, leavers included) go to paid extra work.
FUN_SHARE_FLOOR = 15.0
FUN_SHARE_CEILING = 85.0
PAID_WORK_MIN_SHARE_OF_FREE_MINUTES = 0.05
# Watching is a robustness rule, not a criterion: hidden time is computed over the minutes a participant gave to
# videos and stayed for, and the robustness check drops participants whose page was hidden for more than half of them.
HIDDEN_SHARE_OF_VIDEO_MINUTES_MAX = 0.5
# A second robustness check (materials/SESSION-PAGES.md): the four tests of the share of fun videos (H1, H3, H4 and H5) among
# participants who made all five picks themselves (H3 added after the last cold read, finding 4).
FIVE_PICKS_ROBUSTNESS_HYPOTHESES = ("H1", "H3", "H4", "H5")
# A participant who made no pick has no share of fun videos, so is left out of the four tests of that share (H1, H3, H4
# and H5) only, and stays in H2 and every other outcome; the report gives the number by group. A share of fun is empty only for such a participant: an empty share for someone who made a
# pick is an export error and stops the analysis. The one secondary analysis that uses the share as a covariate (H2
# with the share of fun videos as a covariate) has no value for them and leaves them out.

CHECK_GATES = {
    "surprise_found_break": ("H1", "H2", "H5"),
    "surprise_ai_saved": ("H2", "H3", "H5"),
    "surprise_ai_saved_labeled": ("H4",),
    "attribution_ai_saved": ("H2", "H3", "H4", "H5"),
    "attribution_found_break": ("H1", "H2", "H5"),
    "label": ("H4",),
    "fun_share_floor_ceiling": ("H1", "H3", "H4", "H5"),
    "paid_work_floor": ("H2",),
}
CHECK_DESCRIPTIONS = {
    "surprise_found_break": "Criterion a, surprise, found break: the found-break group rates the break as more "
                            "unexpected than the planned group, with the 95 percent interval of the difference above "
                            "0.15 pooled SD.",
    "surprise_ai_saved": "Criterion a, surprise, AI-saved: the same for the AI-saved group.",
    "surprise_ai_saved_labeled": "Criterion a, surprise, labeled: the same for the labeled AI-saved group.",
    "attribution_ai_saved": "Criterion b, attribution, AI-saved part: at least 80 percent of each AI-saved group choose "
                            '"The AI assistant reviewed the remaining sections".',
    "attribution_found_break": "Criterion b, attribution, found-break part: at most 20 percent of the found-break group "
                               "choose that option, and at least 50 percent choose \"The remaining sections would not "
                               "be reviewed\".",
    "label": 'Criterion c, the label: the labeled group chooses "That saved you 10 minutes." at least 20 percentage '
             "points more often than the unlabeled AI-saved group, with the 95 percent interval of the difference above "
             "zero.",
    "fun_share_floor_ceiling": "Criterion d, floor and ceiling: the planned-break group's mean share of fun videos lies "
                               "between 15 and 85 percent.",
    "paid_work_floor": "Criterion d, floor: the planned-break group gives at least 5 percent of its free minutes to paid "
                       "extra work.",
}
# A registered reading, not a criterion: if the AI-saved group is less surprised than the found-break group, with the
# 95 percent interval of AI-saved minus found below minus 0.15 times the pooled SD of the two groups' surprise scores
# (the size of criterion a), H3 and H5 are not read as evidence against the windfall
# account.
SURPRISE_WINDFALL_LIMIT_D = 0.15
SURPRISE_WINDFALL_LIMIT_HYPOTHESES = ("H3", "H5")
# The readings of H3 and H5 that would count against the windfall account (which predicts H3 higher than planned and
# H5 equivalent).
AGAINST_WINDFALL_READINGS = {"H3": ("supported", "lower than planned"), "H5": ("supported",)}

# ---------------------------------------------------------------------------------------------------------------
# H3 and H5 read as a pair: where AI-saved time lies against the two breaks. The rules are those of
# Supplementary Table S4, with its numbers, and analyze.pair_reading applies them in this order: the first rule that
# holds gives the reading.
#   1. H3 or H5 not interpretable (a failed criterion)                          -> not interpretable
#   2. H3 lower than planned; any H1, any H5                                    -> less fun than a planned break
#   3. H5 higher than found (H5 "opposite"); any H1, any H3                     -> more fun than found time
#   4. H3 or H5 unresolved                                                      -> unresolved
#   5. H1 supported, H3 equivalent (supported), H5 lower (supported)            -> spent like planned time
#   6. H1 supported, H3 higher than planned, H5 equivalent                      -> spent like found time
#   7. H1 supported, H3 higher than planned, H5 lower (supported)               -> between the two breaks
#   8. H3 equivalent (supported) and H5 equivalent; any H1                      -> cannot be told apart from either break
#   9. H1 not supported, H3 and H5 as in rule 5, 6 or 7                         -> restricted: H3 and H5 are reported as
#                                                                                  comparisons with each break
# Rules 2 and 3 depend neither on H1 nor on the other test, unless either test is not interpretable (rule 1). Rule 2
# comes before rule 3; both can hold only if the found break goes to less fun than the planned break (H1 opposite).
# After rules 1 to 4, H3 is supported or higher than planned and H5 is supported or equivalent, so rules 5 to 9 cover
# every combination that is left. Every criterion that gates H1 also gates H5, so H1 is never not interpretable past
# rule 1; rule 9's "not supported" still includes that case, so that every combination has a reading.
# ---------------------------------------------------------------------------------------------------------------
PAIR_LESS_THAN_PLANNED = "less fun than a planned break"
PAIR_MORE_THAN_FOUND = "more fun than found time"
PAIR_RESTRICTED = "restricted"
PAIR_ANY = None  # a rule that holds whatever the result of that test
PAIR_RULES = (  # (rule, H1 condition, H3 results, H5 results, how the H3 and H5 conditions join, pair reading)
    (1, PAIR_ANY, (NOT_INTERPRETABLE,), (NOT_INTERPRETABLE,), "or", NOT_INTERPRETABLE),
    (2, PAIR_ANY, ("lower than planned",), PAIR_ANY, "and", PAIR_LESS_THAN_PLANNED),
    (3, PAIR_ANY, PAIR_ANY, ("opposite",), "and", PAIR_MORE_THAN_FOUND),
    (4, PAIR_ANY, ("unresolved",), ("unresolved",), "or", "unresolved"),
    (5, "supported", ("supported",), ("supported",), "and", "spent like planned time"),
    (6, "supported", ("higher than planned",), ("equivalent",), "and", "spent like found time"),
    (7, "supported", ("higher than planned",), ("supported",), "and", "between the two breaks"),
    (8, PAIR_ANY, ("supported",), ("equivalent",), "and", "cannot be told apart from either break"),
    (9, "not supported", ("supported", "higher than planned"), ("supported", "equivalent"), "and", PAIR_RESTRICTED),
)
PAIR_NO_RULE = "no rule applies"  # never a reading: a combination that no rule covers would show this, and tests fail
PAIR_ORDER = ("spent like planned time", "spent like found time", "between the two breaks",
              "cannot be told apart from either break", PAIR_LESS_THAN_PLANNED, PAIR_MORE_THAN_FOUND,
              PAIR_RESTRICTED, "unresolved", NOT_INTERPRETABLE)
PAIR_ACCOUNTS = {
    "spent like planned time": "AI-saved time goes to fun like planned time and to less fun than found time, as the "
                               "work account predicts.",
    "spent like found time": "AI-saved time goes to more fun than planned time and to fun like found time, as the "
                             "windfall account predicts.",
    "between the two breaks": "AI-saved time goes to more fun than planned time and to less fun than found time.",
    "cannot be told apart from either break": "AI-saved time lies within the bound of both breaks; this can happen only "
                                              "when the two breaks are close, so it is read with H1.",
    PAIR_LESS_THAN_PLANNED: "AI-saved time goes to less fun than even a planned break, as a strong form of the work "
                            "account predicts.",
    PAIR_MORE_THAN_FOUND: "AI-saved time goes to more fun than found time, as the goal-progress account predicts.",
    PAIR_RESTRICTED: "H1 is not supported, so the found break is not shown to differ from the planned break; H3 and H5 "
                     "are reported as comparisons with each break, not as a position between them.",
    "unresolved": "At least one of the two tests is not conclusive.",
    NOT_INTERPRETABLE: "A criterion tied to H3 or H5 failed, so the pair is not read.",
}

# ---------------------------------------------------------------------------------------------------------------
# Planned secondary measures with no hypothesis (materials/SESSION-PAGES.md). Each column is
# optional: if the export does not have it yet, the report says so, and a missing value never stops the analysis.
# ---------------------------------------------------------------------------------------------------------------
# Where the AI-saved group lies between the breaks: (AI-saved minus planned) / (found minus planned), share of fun,
# reported only when found minus planned is significant at the registered alpha, with a bootstrap interval
# (participants resampled within each group and country).
POSITION_BOOTSTRAP_DRAWS = 2000
POSITION_BOOTSTRAP_SEED = 20261125
# ASSUMED column names, to be matched to the real exports.
AUTOMATED_FLAG_COLUMN = "flagged_possibly_automated"  # from Prolific's own checks; True, False or empty
PURPOSE_CODED_COLUMN = "purpose_names_reason_or_source"  # the free-text purpose answer, coded blind to group
HEARD_BEFORE_COLUMN = "heard_of_study_before"
HEARD_BEFORE_ANSWERS = {"yes": "Yes", "no": "No", "not_sure": "Not sure"}
STARTED_AT_COLUMN = "started_at"  # the session's start, for the week of data collection (week 1 starts on the first day)
TIMING_COLUMNS = ("seconds_to_fifth_pick", "seconds_to_last_slot_change")  # empty if the step was not reached
REVIEW_EDIT_COLUMNS = ("review_edits", "review_characters_changed")  # on sections 1 to 3

# What 0.10 USD (the pay for one typing item) is worth in each country, reported beside the country estimates of
# minutes given to paid extra work. Reported only.
# Source: World Bank, World Development Indicators, PA.NUS.PPP (PPP conversion factor, GDP, local currency per
# international dollar) and PA.NUS.FCRF (official exchange rate, local currency per US dollar, period average), year
# 2025, data updated 2026-11-11; downloaded Nov 24, 2026 from api.worldbank.org/v2 and kept in analysis/reference/.
# 0.10 USD converted at the exchange rate buys 0.10 x FCRF / PPP international dollars locally.
PPP_YEAR = 2025
PPP_ITEM_USD = 0.10
PPP_TABLE = {  # country: (PPP conversion factor, exchange rate), both in local currency
    "United States": (1.0, 1.0),
    "United Kingdom": (0.677133, 0.759473957338607),
    "Canada": (1.166683, 1.39810565893901),
    "Australia": (1.398943, 1.5519978866485),
    "South Africa": (7.42029658394341, 17.8887489855833),
    "Mexico": (10.328951, 19.2375083333333),
    "Chile": (453.188563, 951.329454191033),
    "Germany": (0.709983, 0.88496895841236),
    "Poland": (1.969559, 3.76054166666666),
    "Portugal": (0.534218, 0.88496895841236),
    "Italy": (0.610275, 0.88496895841236),
    "Spain": (0.570991, 0.88496895841236),
}


def ppp_value_of_item(country):
    """0.10 USD in local currency at the exchange rate, and what it buys there in international dollars."""
    ppp, rate = PPP_TABLE[country]
    return {"local_currency_at_exchange_rate": PPP_ITEM_USD * rate,
            "international_dollars": PPP_ITEM_USD * rate / ppp}


# ---------------------------------------------------------------------------------------------------------------
# Replacement, attrition and withdrawal. Every new participant is randomized in permuted blocks of four within country;
# a country stops when each group reaches its share; those who do not finish or who withdraw at the debriefing are
# replaced by new randomized participants. If any two groups differ by more than 3 percentage points in the share of
# assigned participants who did not finish or withdrew, Lee bounds are reported for each confirmatory estimate.
# ---------------------------------------------------------------------------------------------------------------
ATTRITION_DIFFERENCE_THRESHOLD = 0.03

# ---------------------------------------------------------------------------------------------------------------
# Planned secondary analyses (95 percent intervals, no test).
# ---------------------------------------------------------------------------------------------------------------
SECONDARY_INTERVAL_LEVEL = 0.95
