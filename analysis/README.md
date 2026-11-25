# Analysis code for the experiment

This folder holds four things. The first is the analysis of the experiment as Table 1 of the Stage 1 protocol registers
it. The second is a simulator that writes data in the format that the
survey will export. The third is a power analysis by simulation, and the fourth is the
tests of the code. No participant has taken part yet, so every data file in output/ is simulated. The code uses Python
3 with numpy, pandas, scipy and statsmodels only. It needs no network and runs on one processor thread.

## How to reproduce every output

Run these commands from the repository's root, in this order:

```
python power/design_table_power.py > power/design_table_power_output.txt
python analysis/simulate.py --all
python analysis/analyze.py analysis/output/simulated_no_effects.csv
python analysis/analyze.py analysis/output/simulated_hypotheses_true_at_bound.csv
python analysis/analyze.py analysis/output/simulated_windfall_account.csv
python analysis/analyze.py analysis/output/simulated_work_account.csv
python analysis/analyze.py analysis/output/simulated_halfway.csv
python analysis/power_by_simulation.py
python analysis/tests/run_tests.py
python analysis/tests/mutation_check.py
python analysis/tools/registered_power_table.py
```

The simulator writes five data sets in about 2 seconds, and each analysis takes about 3 seconds. The power analysis
takes about 6.5 minutes, the tests about 25 seconds and the mutation check about 4 minutes. The last command prints the tables of this README. Then run the commands in figures/README.md, so that Figure 3 and its legend show the new numbers. We ran them with Python 3.12.10, numpy 1.26.4,
pandas 3.0.5, scipy 1.17.1 and statsmodels 0.14.6. Every random draw starts from a fixed seed (20261124 and the numbers
after it), so a second run writes the same files.

When the real data arrive, the registered analysis is one command:
`python analysis/analyze.py <the survey's export>.csv`.

## What each file does

| File | What it does |
|---|---|
| registered_constants.py | Every registered number in one place: the four groups, 1,706 participants a group, the 12 countries, alpha .01 for each test, the smallest effect d = 0.15, the five hypotheses with the reading of each result and the account each reading supports, the criteria and the tests that each makes not interpretable, the export's option texts, and the attrition threshold. A number that no source states is marked ASSUMED, with the reason. |
| simulate.py | Writes one data set in the export's format under one of five patterns of true effects, from a fixed seed. |
| analyze.py | Reads such a file and runs the exclusions, the attrition table, the outcome-neutral criteria, H1 to H5 and the planned secondary analyses, in that order. It writes results_<name>.json and report_<name>.txt in output/ and prints the reading of every test. |
| power_by_simulation.py | Runs the functions of analyze.py on many simulated data sets and counts how often each reading occurs under each pattern. |
| tools/registered_power_table.py | Prints the tables of "Results on the simulated data" from output/power_by_simulation.json. |
| tests/test_reading_rules.py | 73 fast tests of the reading rules, the pair reading of H3 and H5 (all 125 combinations of H1, H3 and H5, with the rule of each), the criteria, the gates, the participants who made no pick, the attrition threshold, the Lee bounds, the position ratio, the PPP table and the export's checks, on constructed values and a small simulated export. |
| tests/test_pipeline.py | 31 tests of the whole analysis on simulated data. |
| tests/run_tests.py | Runs all 104 tests and keeps the log in output/tests.txt. |
| reference/ | The World Bank files behind the purchasing-power table in registered_constants.py (PA.NUS.PPP and PA.NUS.FCRF, 2023 to 2025, downloaded Nov 24, 2026). |
| tests/mutation_check.py | Changes one threshold, sign, reading or gate at a time in a copy of the code and confirms that a fast test fails each time. |

The outputs in output/:

| File | What it holds |
|---|---|
| simulated_<pattern>.csv | Five simulated data sets of about 7,450 rows each: 6,824 analyzable participants, about 340 who failed the attention check, and about 210 who did not finish and 70 who withdrew. |
| results_<pattern>.json, report_<pattern>.txt | The registered analysis of each simulated data set. |
| power_by_simulation.json, power_by_simulation.txt | The power analysis by simulation. |
| tests.txt | The log of the 104 tests. |
| mutation_check.json, mutation_check.txt | The table of the mutation check. |

## The analysis, in order

1. Exclusions. We exclude only the three groups of people that the protocol names. The first failed the attention
   check before assignment (in the export they have no group). The second did not finish the session. The third
   withdrew their answers at the debriefing. No one is excluded on any answer given after assignment. If someone who
   finished has no answer to a question that the analysis uses, or has no group, the script stops and says so,
   because dropping that person would be an exclusion after assignment; the survey therefore requires every answer.
   Three columns have no value by design for some participants: the share of fun videos (none for a participant who
   made no pick), hidden time (none without video minutes) and employment (from Prolific's export, which can expire).
   A participant who made no pick is therefore left out of the four tests of the share of fun videos (H1, H3, H4 and
   H5) and of nothing else: the participant stays in H2, in the criteria and in every secondary analysis, except H2
   with the share of fun videos as a covariate, which has no covariate for them. The report gives the number by
   group. An empty share for a participant who made a pick is an export error, and the script stops.
   The script also checks the export against the session's pages: the share of fun videos counts only the picks a
   participant made, not those filled at random at c + 4:00; leaving early replaces all five slots; a participant
   with filled picks never saw the slots; the minutes of paid extra work are 2 for each work slot. A table gives the
   participants assigned, not finishing, withdrawing and leaving early, by group and by group and country.
2. The outcome-neutral criteria. Each uses the analyzed participants; the intervals are 95 percent, from the same
   regression as the tests. A failed criterion makes the tests tied to it not interpretable: they are reported with
   their estimates and intervals, and with the reading they would have had, but they are not read.

   | Criterion | Passes when | If it fails, not interpretable |
   |---|---|---|
   | a, surprise, found break | the found-break group minus the planned group, on the mean of Chung and colleagues' two items, has its 95 percent interval above 0.15 pooled SD | H1, H2, H5 |
   | a, surprise, AI-saved | the same for the AI-saved group | H2, H3, H5 |
   | a, surprise, labeled | the same for the labeled group | H4 |
   | b, attribution, AI-saved part | at least 80 percent of each AI-saved group choose "The AI assistant reviewed the remaining sections" (observed shares) | H2, H3, H4, H5 |
   | b, attribution, found-break part | at most 20 percent of the found-break group choose that answer, and at least 50 percent choose "The remaining sections would not be reviewed" (observed shares) | H1, H2, H5 |
   | c, the label | the labeled group chooses "That saved you 10 minutes." on the recognition item at least 20 points more often than the unlabeled AI-saved group, with the 95 percent interval of the difference above zero | H4 |
   | d, floor and ceiling | the planned-break group's mean share of fun videos lies between 15 and 85 percent | H1, H3, H4, H5 |
   | d, floor | the planned-break group gives at least 5 percent of its free minutes to paid extra work (10 minutes for each participant, leavers included) | H2 |

   Beside criterion a, the report gives the AI-saved group minus the found-break group on surprise, with its 95
   percent interval. If the interval lies below minus 0.15 times the pooled SD of the two groups' scores, H3 and H5
   are not read as evidence against the windfall account:
   a reading of H3 as supported or lower than planned, or of H5 as supported, is given a note and the flag
   not_read_against_windfall. The full table of answers to the attribution question and to the recognition item is
   reported by group. Watching is not a criterion: it is the rule of a robustness check (item 4).
3. H1 to H5. For each outcome (the share of fun videos among the picks a participant made, and the minutes given to
   paid extra work), we fit one linear regression on group, with all four groups, the planned break as the reference
   group and fixed effects for the 12 countries. The standard errors are HC2, which allow the groups to have unequal
   variances (for two groups HC2 gives Welch's standard error). Each hypothesis is one contrast from that regression:

   | Hypothesis | Outcome | Contrast | Prediction |
   |---|---|---|---|
   | H1 | share of fun videos | found break minus planned break | higher |
   | H2 | minutes given to paid extra work | AI-saved minus found break | higher (the work account) or lower (the goal-progress account) |
   | H3 | share of fun videos | AI-saved minus planned break | within the bound |
   | H4 | share of fun videos | AI-saved labeled minus AI-saved | higher (a named saving treated as a gain) or lower (the label reminds people of the work) |
   | H5 | share of fun videos | AI-saved minus found break | lower (the work account); equivalent (the windfall account) |

   Each contrast gets a two-sided t test at .01 and two one-sided tests at .01 against the bounds of plus and minus
   0.15 times the pooled standard deviation of the two groups' raw values (so the SD includes the differences between
   countries). Each test first gives a result: higher or lower when the two-sided test is significant (its 99 percent
   interval excludes zero), equivalent when both one-sided tests are significant (the 98 percent interval lies inside
   the bounds), and unresolved when neither is. Each hypothesis names each result:

   | Result | H1 | H2 | H3 | H4 | H5 |
   |---|---|---|---|---|---|
   | higher | supported | supported (work account) | higher than planned (windfall account) | higher (a gain) | opposite (goal-progress account) |
   | lower | opposite | lower (goal-progress account) | lower than planned | lower (a reminder of the work) | supported (work account) |
   | equivalent | equivalent | equivalent | supported (work account) | equivalent (no change) | equivalent (windfall account) |
   | neither | unresolved | unresolved | unresolved | unresolved | unresolved |

   H3 and H5 are also read as a pair, which says where AI-saved time lies against the two breaks. The rules are
   registered_constants.PAIR_RULES, the rules of Supplementary Table S4 with its numbers, in their registered
   order; analyze.pair_reading tries them in that order, and the
   first that holds gives the reading (the report also gives the rule's number):

   | Rule | H1 | H3 | H5 | Pair reading |
   |---|---|---|---|---|
   | 1 | any | not interpretable, or | not interpretable | not interpretable |
   | 2 | any | lower than planned | any | less fun than a planned break (a strong form of the work account) |
   | 3 | any | any | opposite (higher than found) | more fun than found time (the goal-progress account) |
   | 4 | any | unresolved, or | unresolved | unresolved |
   | 5 | supported | supported (equivalent) | supported (lower than found) | spent like planned time (the work account) |
   | 6 | supported | higher than planned | equivalent | spent like found time (the windfall account) |
   | 7 | supported | higher than planned | supported (lower than found) | between the two breaks |
   | 8 | any | supported (equivalent) | equivalent | cannot be told apart from either break (read with H1) |
   | 9 | not supported (equivalent, opposite or unresolved) | as in rule 5, 6 or 7 | as in rule 5, 6 or 7 | restricted (H3 and H5 reported as comparisons with each break) |

   Rules 2 and 3 depend neither on H1 nor on the other test, unless either test is not interpretable (rule 1). Rule 9
   covers only the three combinations of rules 5 to 7, which name a position; "cannot be told apart from either break"
   (rule 8) does not depend on H1. After rules 1 to 4, H3 is supported or higher than planned and H5 is supported or
   equivalent, so every combination has exactly one first rule (the fast tests check all 125 combinations of H1, H3
   and H5, with the rule that gives each).
   At the registered sample a result cannot be both significant and inside the bounds: a significant difference
   needs more than 0.088 d and equivalence needs less than 0.070 d. The code
   still handles the case: H1, H2, H4 and H5 read significance first, with a note, and H3 reads equivalence first. We
   fit each regression twice, once with statsmodels and once with a few lines of numpy that the power analysis uses on
   many data sets at once, and the script stops if the two differ by more than 0.00000001.
4. The planned secondary analyses, each with a 95 percent interval and no test:
   - the minutes given to each of the four uses of the 10 minutes (fun videos watched, self-improvement videos
     watched, paid extra work, time left early), by group and as a contrast for each hypothesis's two groups;
   - the share who left early, by group (Wilson intervals) and as contrasts;
   - the minutes given to paid extra work, labeled minus unlabeled AI-saved;
   - H2 with the participant's share of fun videos as a covariate;
   - H2 among participants who stayed, labeled as a comparison of groups selected after assignment;
   - the five contrasts within each country (difference of means with Welch's standard error), with a DerSimonian
     and Laird random-effects summary and its 95 percent prediction interval (Higgins, Thompson and Spiegelhalter,
     2009);
   - the five contrasts among participants who use generative AI weekly (1 to 6 days a week or every day, asked
     before assignment), and among participants in paid employment (full or part time, from Prolific's export);
   - robustness check 1: H1 to H5, with their readings, without the participants whose page was hidden for more than
     half of the minutes they gave to videos and stayed for;
   - robustness check 2: the four tests of the share of fun videos (H1, H3, H4 and H5) among participants who made all
     five picks themselves;
   - the share who had not finished the pick at c + 4:00, and the share who made no pick, by group;
   - criterion d by country, in the planned-break group, with no gate: the mean share of fun and the share of free
     minutes given to paid extra work in each country, and, beside them and beside H2 by country, what one typing
     item (0.10 USD) buys in each country in international dollars (World Bank PPP, 2025);
   - where the AI-saved group lies between the two breaks, (AI-saved minus planned) divided by (found minus planned)
     on the share of fun, with a 95 percent bootstrap interval (2,000 draws, participants resampled within each group
     and country), reported only when found minus planned is significant at .01;
   - the fun-share contrasts (H1, H3, H4, H5) among participants who watched at least one video;
   - the share of sessions flagged as possibly automated by Prolific's own checks, by group;
   - the coded answers to the question on the study's purpose and the answers to the question on having heard of the
     study, by group and by week of data collection;
   - the seconds before the fifth pick and before the last change of the slots, and the edits on sections 1 to 3, by
     group;
   - if any two groups differ by more than 3 percentage points in the share of assigned participants who did not
     finish or withdrew, Lee bounds for each confirmatory estimate.

## The data file

One row per participant who started the session, with these columns:

| Column | Meaning |
|---|---|
| participant_id | an identifier |
| country | one of the 12 countries |
| passed_attention_check | True or False; asked before assignment |
| ai_use | never, less_than_weekly, one_to_six_days_a_week or every_day (or the option's text); asked before assignment |
| group | planned_break, found_break, ai_saved or ai_saved_labeled; empty for anyone who failed the attention check |
| finished, withdrew | True or False; withdrew is True for someone who withdrew the answers at the debriefing |
| employment_status | from Prolific: full_time, part_time, starting_new_job_within_a_month, unemployed_job_seeking, not_in_paid_work or other (or Prolific's text); empty if expired |
| video_pick_1 to video_pick_5 | the video of each slot, in the order the slots play, each fun_1 to fun_5 or self_improvement_1 to self_improvement_5 |
| pick_1_filled to pick_5_filled | True for a pick filled at random at c + 4:00 |
| slot_1_use to slot_5_use | watch, paid_work or left_early (a leaver has left_early on all five) |
| left_early | True or False |
| fun_share | the share of fun videos among the picks the participant made, 0 to 100; empty if no pick was made |
| paid_work_minutes | 2 for each work slot, 0 to 10 |
| minutes_video_page_hidden | the minutes of the video slots during which the session's page was not the visible tab; empty without video minutes |
| surprise_unexpected, surprise_surprising | Chung et al.'s two items, 1 to 7 |
| how_review_ended | as_planned_after_three_sections, remaining_sections_would_not_be_reviewed, ai_assistant_reviewed_remaining_sections or do_not_know (or the option's text) |
| saw_reviewed_three_sections, saw_remaining_sections_not_reviewed, saw_ai_assistant_reviewed_remaining, saw_saved_you_10_minutes, saw_review_saved_automatically, saw_session_will_restart | the recognition item, True for each sentence chosen |
| flagged_possibly_automated (optional) | True or False from Prolific's own checks; empty if Prolific gives no value |
| purpose_names_reason_or_source (optional) | the coded purpose answer, True when it names the reason the review ended or the source of the break |
| heard_of_study_before (optional) | yes, no or not_sure (or the option's text) |
| started_at (optional) | the session's start, for the week of data collection |
| seconds_to_fifth_pick, seconds_to_last_slot_change (optional) | seconds on the choosing page; empty if the step was not reached |
| review_edits, review_characters_changed (optional) | edits on sections 1 to 3 |

An optional column that the export does not have is reported as missing, and an empty value in one never stops the
analysis. None of them enters the five registered tests.

## What the simulation assumes

Each pattern of true effects puts each group at a step on each outcome. Two groups one step apart differ by d = 0.15
of their pooled standard deviation within countries, which is 4.34 points of the share of fun or 0.464 minutes of
paid work.

| Pattern | Fun share: planned, found, AI-saved, labeled | Paid work: planned, found, AI-saved, labeled | Expected readings |
|---|---|---|---|
| no_effects | 0, 0, 0, 0 | 0, 0, 0, 0 | H1 equivalent, H2 equivalent, H3 supported, H4 equivalent, H5 equivalent |
| hypotheses_true_at_bound | 0, 1, 0, 1 | 0, 0, 1, 1 | H1, H2, H3 and H5 supported, H4 higher |
| windfall_account | 0, 1, 1, 1 | 0, 0, 0, 0 | H1 supported, H2 equivalent, H3 higher than planned, H4 equivalent, H5 equivalent |
| work_account | 0, 1, 0, 0 | 0, 0, 1, 1 | H1, H2, H3 and H5 supported, H4 equivalent |
| halfway (not powered) | 0, 1, 0.5, 0.5 | 0, 0, 0, 0 | H1 supported, H2 equivalent, H3 higher than planned, H4 equivalent, H5 supported |

The other numbers are guesses, and none of them is registered. Each participant's leaning toward fun is drawn from a
beta distribution, and the five picks follow it; the planned break averages 50 points, the countries differ by up to 6
points either way, and the standard deviation within a country is 29 points, as in Chung et al. Fifteen percent of
every group leave early, which replaces all five slots. Among everyone (leavers at 0), about 31 percent give 1 to 5
slots to paid work in the groups at step 0, so that the minutes average about 1.9 with a standard deviation of 3.0
within a country; the calibration includes the leavers. For the criteria, the planned break averages 2.5 on the
surprise items and the other groups 4.5; 90 percent of each AI-saved group and 5 percent of the found-break group say
the assistant reviewed the remaining sections, and 80 percent of the found-break group say the sections would not be
reviewed; 80 percent of the labeled group and 20 percent of the unlabeled group recognize "That saved you 10 minutes."
In addition, 20 percent of participants hide the page for part of their video minutes, 2 percent leave one or two picks
to be filled and 0.3 percent make no pick (these never see the slots), 5 percent fail the attention check, 3 percent of
those assigned do not finish and 1 percent withdraw. The power analysis assumes that everyone makes all five picks.
The optional columns are filled with stand-in values (1 percent flagged as automated, 10 percent of the planned group
and 20 percent of the others naming the reason in the purpose answer, 2 percent having heard of the study, six weeks of
data collection, about 25 seconds to the fifth pick, about two edits on sections 1 to 3), so that the code that
reports them runs; none of them enters the tests.

## Results on the simulated data

Power for each hypothesis under each powered pattern, from 20,000 simulated data sets a pattern, with the registered
bound (power_by_simulation.txt, part 2; printed by tools/registered_power_table.py). Each share is the proportion of
data sets with the expected reading, with its 95 percent Monte Carlo interval. A failed criterion counts as a miss; no
criterion failed in any of the 100,000 data sets.

| Hypothesis | Pattern | Expected reading | Share [95% interval] | Table 1 claims | Against 0.95 | Against 0.96 |
|---|---|---|---|---|---|---|
| H1 | hypotheses_true_at_bound | supported | 0.965 [0.962, 0.968] | 0.964 | 0.95 or more | 0.96 or more |
| H1 | windfall_account | supported | 0.966 [0.964, 0.969] | 0.964 | 0.95 or more | 0.96 or more |
| H1 | work_account | supported | 0.965 [0.963, 0.968] | 0.964 | 0.95 or more | 0.96 or more |
| H1 | no_effects | equivalent | 0.964 [0.961, 0.966] | 0.960 | 0.95 or more | 0.96 or more |
| H2 | hypotheses_true_at_bound | supported | 0.964 [0.961, 0.966] | 0.964 | 0.95 or more | 0.96 or more |
| H2 | work_account | supported | 0.966 [0.963, 0.968] | 0.964 | 0.95 or more | 0.96 or more |
| H2 | windfall_account | equivalent | 0.961 [0.958, 0.963] | 0.960 | 0.95 or more | cannot be told apart from 0.96 |
| H2 | no_effects | equivalent | 0.959 [0.956, 0.961] | 0.960 | 0.95 or more | cannot be told apart from 0.96 |
| H3 | hypotheses_true_at_bound | supported | 0.963 [0.961, 0.966] | 0.960 | 0.95 or more | 0.96 or more |
| H3 | windfall_account | higher than planned | 0.966 [0.963, 0.968] | 0.964 | 0.95 or more | 0.96 or more |
| H3 | work_account | supported | 0.963 [0.960, 0.965] | 0.960 | 0.95 or more | cannot be told apart from 0.96 |
| H3 | no_effects | supported | 0.966 [0.963, 0.968] | 0.960 | 0.95 or more | 0.96 or more |
| H4 | hypotheses_true_at_bound | higher | 0.963 [0.961, 0.966] | 0.964 | 0.95 or more | 0.96 or more |
| H4 | windfall_account | equivalent | 0.963 [0.960, 0.965] | 0.960 | 0.95 or more | 0.96 or more |
| H4 | work_account | equivalent | 0.963 [0.961, 0.966] | 0.960 | 0.95 or more | 0.96 or more |
| H4 | no_effects | equivalent | 0.962 [0.959, 0.964] | 0.960 | 0.95 or more | cannot be told apart from 0.96 |
| H5 | hypotheses_true_at_bound | supported | 0.962 [0.959, 0.964] | 0.964 | 0.95 or more | cannot be told apart from 0.96 |
| H5 | work_account | supported | 0.965 [0.962, 0.967] | 0.964 | 0.95 or more | 0.96 or more |
| H5 | windfall_account | equivalent | 0.963 [0.961, 0.966] | 0.960 | 0.95 or more | 0.96 or more |
| H5 | no_effects | equivalent | 0.963 [0.960, 0.965] | 0.960 | 0.95 or more | cannot be told apart from 0.96 |

Every cell lies between 0.959 and 0.966, and every interval lies above 0.95. Fourteen intervals lie at or above 0.96,
and the other six include 0.96. The fixed bound of 4.35 points and 0.45 minutes, which is not registered, gives about
the same shares at the assumed standard deviations (the rows marked "Bound form: fixed" in the same file).

The halfway pattern puts both AI-saved groups 0.075 d above the planned break, halfway to the found break. The study is
not powered for it, because each of its differences is half the smallest effect of interest:

| Hypothesis | Expected reading | Share with it | All readings |
|---|---|---|---|
| H1 | supported | 0.966 | supported 0.966, equivalent 0.011, unresolved 0.024 |
| H2 | equivalent | 0.961 | supported 0.005, equivalent 0.961, lower 0.005, unresolved 0.029 |
| H3 | higher than planned | 0.355 | supported 0.461, higher than planned 0.355, unresolved 0.184 |
| H4 | equivalent | 0.965 | equivalent 0.965, higher 0.005, lower 0.005, unresolved 0.025 |
| H5 | supported | 0.348 | supported 0.348, equivalent 0.467, unresolved 0.185 |

H3 and H5 read as a pair (the rules in "The analysis, in order", item 3), share of data sets with each pair reading
(registered bound; printed by tools/registered_power_table.py):

| Pattern | Expected pair reading | Share [95% interval] | like planned | like found | between | cannot be told apart | less than planned | more than found | restricted | unresolved |
|---|---|---|---|---|---|---|---|---|---|---|
| hypotheses_true_at_bound | spent like planned time | 0.908 [0.904, 0.912] | 0.908 | 0.001 | 0.004 | 0.010 | 0.005 | 0.000 | 0.022 | 0.051 |
| windfall_account | spent like found time | 0.912 [0.908, 0.916] | 0.001 | 0.912 | 0.003 | 0.008 | 0.000 | 0.006 | 0.022 | 0.048 |
| work_account | spent like planned time | 0.908 [0.904, 0.912] | 0.908 | 0.001 | 0.003 | 0.009 | 0.005 | 0.000 | 0.023 | 0.051 |
| no_effects | cannot be told apart from either break | 0.934 [0.931, 0.937] | 0.000 | 0.000 | 0.000 | 0.934 | 0.005 | 0.005 | 0.006 | 0.049 |
| halfway | between the two breaks | 0.056 [0.053, 0.059] | 0.236 | 0.241 | 0.056 | 0.136 | 0.000 | 0.000 | 0.003 | 0.328 |

No pair was not interpretable, because no criterion failed. The pair readings that name a position need three tests at
once (H1 supported, and H3 and H5 each conclusive), so they occur in about 0.91 of data sets under the patterns that
imply them, against about 0.96 for each test alone. The study is powered for each test, not for the pair. Under the
halfway pattern, the pair reads "between the two breaks" in only 0.056 of data sets, and reads like the work account
(0.236) or like the windfall account (0.241) about equally often.

Power of each equivalence test (both one-sided tests significant at .01) against the true difference, from 10,000
data sets for each point (part 3):

| True difference (d) | Normal theory | H1 | H2 | H3 | H4 | H5 |
|---|---|---|---|---|---|---|
| 0.00 | 0.960 | 0.967 | 0.960 | 0.965 | 0.965 | 0.965 |
| 0.05 | 0.724 | 0.734 | 0.718 | 0.728 | 0.726 | 0.725 |
| 0.10 | 0.193 | 0.202 | 0.192 | 0.209 | 0.209 | 0.213 |
| 0.15 | 0.010 | 0.010 | 0.010 | 0.011 | 0.010 | 0.013 |

The criteria. Under the simulator's assumptions no criterion failed in any data set, so these values say only that
manipulations as strong as those assumed pass. By normal theory at the registered sample (part 5), each criterion
fails with probability 0.05 at these true values, and with probability 0.50 at the threshold itself:

| Criterion | Fails with probability 0.05 at | 0.50 at |
|---|---|---|
| surprise, each group (difference from planned, in d of the surprise score) | 0.273 | 0.217 |
| attribution, AI-saved part (true share in each AI-saved group) | 0.818 | 0.805 |
| attribution, found-break part (true share choosing the assistant) | 0.185 | 0.200 |
| attribution, found-break part (true share choosing "would not be reviewed") | 0.520 | 0.500 |
| label (true difference in recognition, unlabeled at 0.20) | 0.225 | 0.200 |
| fun-share floor (true planned mean, points; the ceiling mirrors it at 83.8) | 16.2 | 15.0 |
| paid-work floor (true share of the planned group's free minutes) | 0.057 | 0.050 |

When the AI-saved and found-break groups are equally surprised in truth, the interval of AI-saved minus found never
lay below minus 0.15 pooled SD in 100,000 data sets, so the limit on reading H3 and H5 did not apply by chance. With
the earlier rule (the interval below zero) it applied in about 2.5 percent.

Equivalence power when the true standard deviation differs from the one behind each fixed bound (2,000 data sets with
no effect, part 4):

| True SD of fun share | H1, H3, H4, H5 with the fixed bound of 4.35 points | With the bound from the observed SD |
|---|---|---|
| 25 points | 0.985 to 0.994 | 0.957 to 0.971 |
| 29 points | 0.950 to 0.967 | 0.955 to 0.969 |
| 33 points | 0.857 to 0.881 | 0.955 to 0.965 |

| True SD of paid-work minutes | H2 with the fixed bound of 0.45 minutes | With the bound from the observed SD |
|---|---|---|
| 2.0 minutes | 0.991 | 0.960 |
| 3.0 minutes | 0.960 | 0.963 |
| 4.0 minutes | 0.668 | 0.959 |

How H1 to H5 read on the one simulated data set for each pattern (report_<pattern>.txt):

| Pattern | H1 | H2 | H3 | H4 | H5 | H3 and H5 as a pair |
|---|---|---|---|---|---|---|
| no_effects | equivalent | unresolved | supported | equivalent | equivalent | cannot be told apart from either break |
| hypotheses_true_at_bound | supported | supported | supported | higher | supported | spent like planned time |
| windfall_account | supported | equivalent | higher than planned | equivalent | equivalent | spent like found time |
| work_account | supported | supported | supported | equivalent | supported | spent like planned time |
| halfway | supported | equivalent | higher than planned | equivalent | supported | between the two breaks |

Of the twenty readings of the four powered patterns, nineteen are the expected ones. H2 in the data set with no
effect reads unresolved (estimate -0.24 minutes, two-sided p = .020, equivalence p = .020), which happens in about 3
percent of such data sets. The halfway data set happens to read between the two breaks, which occurs in about 6
percent of such data sets.

## Tests

All 104 tests pass (output/tests.txt). The pipeline tests show that planted effects are found (including H2's
goal-progress reading and the pair readings "spent like planned time" and "spent like found time"), that a data set
with no effect reads equivalent, that the people who failed the attention check, did not finish or withdrew are
excluded and no one else is, that a missing answer or a finisher with no group stops the analysis, that each failed
criterion makes exactly its own tests not interpretable while their estimates are still reported, that the windfall
limit applies to H3 and H5 only, and that Lee bounds appear only when attrition differs by more than 3 points. The fast
tests write out all 125 combinations of H1, H3 and H5 with the reading and the rule of Supplementary Table S4 that
gives each; they check that a participant who made no pick is left out of H1, H3, H4 and H5 only, stays in H2 and the
other outcomes, and is counted by group, and that the robustness check among participants who made all five picks
covers H1, H3, H4 and H5; and they check the PPP table against the World Bank files. The mutation check made 70
changes to the code, one at a time: thresholds (alpha, the bound, the 80, 20 and 50 percent of criterion b, the 20
points of criterion c, the floor and the ceiling, the surprise limit, the attrition threshold, the robustness rule),
signs (the result's direction, either one-sided test), readings (each hypothesis's names, the nine pair rules, their
order, the H1 condition of each and the code that applies them), the gates of each criterion, the position ratio's
condition, the PPP formula, the export's checks (filled picks, partial leaving, hidden time), the rule for
participants who made no pick and the tests of the five-picks check. A fast test failed for all 70
(output/mutation_check.txt).

## Choices made in the code

The registered design does not settle the following points. Each choice is made in one place in the code.

- the reading words: H2's lower result is "lower" (the goal-progress account), H4's results are "higher" and
  "lower", and H5's are "supported" (lower than found) and "opposite";
- the bootstrap of the position ratio resamples participants within each group and country, so the regression's
  design is the same in every draw; it uses a fixed seed and 2,000 draws;
- the PPP value of one item uses the GDP-based PPP conversion factor and the official exchange rate for 2025, the
  latest year with both values for all 12 countries;
- the names of the optional columns (Prolific's automated flag, the coded purpose answer, heard-before, the start
  time, the choosing times, the review edits) are guesses to be matched to the real exports;
- the observed difference of criterion c is the regression contrast with country fixed effects, which equals the
  raw difference when the groups fill evenly within each country;
- Lee bounds compare the raw values of the two groups of each contrast, pooled over countries, trim the group that
  kept more of its assigned participants, and are given without an interval;
- recruitment continues until every group reaches its share, so groups end at or above 1,706; the exclusion check
  asks for at least 1,706, and the power analysis uses exactly 1,706;
- the four uses count every video that played, including picks filled at random; the share of fun counts only the
  picks made;
- a participant with picks filled at random cannot have a work slot or have left, because the slots appear only
  after five picks; an export that shows otherwise stops the analysis, and so does a hidden-time value for someone
  without video minutes;
- Prolific's employment labels in registered_constants.py are those we know and must be checked against the real
  export;
- the split of 1,706 a group across 12 countries (143 in two countries, 142 in ten) stands in for the real split;
- the cost script keeps the first draft's 0.40 USD of paid extra work a person on average; with the typing items at
  0.40 USD a slot, that is one work slot a person, and the simulation's participants give about 0.94 slots each.
