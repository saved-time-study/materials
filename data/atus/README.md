# The American Time Use Survey before and after ChatGPT

This folder holds the analysis of the national time diaries for the manuscript. We compare how U.S.
workers in the occupations most exposed to generative AI spent their day with how workers in the least exposed
occupations spent theirs, from 2015 to 2025, before and after ChatGPT was released on November 30, 2022. The Supplementary
Information of the manuscript reports the findings.

## How to rerun everything from nothing

From the root of the repository, with Python 3.11 and the packages pandas, numpy, scipy, statsmodels, matplotlib,
openpyxl, xlrd and pyarrow:

```
python -I data/atus/scripts/run_all.py
```

It runs these steps in order, and stops at the first one that fails:

| Step | Script | What it does | Time on one computer |
|---|---|---|---|
| 0 | test_fe.py | Tests the estimator against statsmodels with explicit dummies (4 tests, one of them a planted fault) | seconds |
| 1 | 01_download.py | Downloads every input into raw/, extracts the files of each Zip archive into a folder of the archive's name, and writes MANIFEST.csv (URL, bytes and SHA-256 of each file) | a few minutes |
| 2 | 02_build_sample.py | Builds one row per employed respondent aged 18 to 64, 2015 to 2025 without 2020 | about a minute |
| 3 | 03_exposure.py | Puts both Census occupation schemes on one harmonized list and attaches four exposure scores | seconds |
| 4 | 04_analysis.py | Runs every model and writes every estimate to results/ | under a minute |
| 5 | 05_figures_tables.py | Draws the figures and the main table from results/ only | seconds |
| 6 | 06_checks.py | Refuses (exit 1) when the outcomes' estimates do not add up to zero, a table cell has no matching estimate, or a respondent lacks an occupation or a score | seconds |
| 7 | 07_jpxz_replication.py | Reproduces the ChatGPT event study of Jiang, Park, Xiao and Zhang (NBER w33536, Table 2, column 1) on 2022 and 2023, then adds 2024 and 2025 and other reference years; writes results/jpxz_estimates.csv, jpxz_key_numbers.json and figureS4_jpxz_event_study.png | under a minute |

The raw files (about 2.2 GB once extracted) and derived/sample.parquet are not in git; the scripts make them again.
The Bureau of Labor Statistics refuses a download whose User-Agent holds no contact e-mail, so set the environment
variable BLS_CONTACT_EMAIL to your own address before the first run (for example,
`export BLS_CONTACT_EMAIL=you@example.org`, or `set BLS_CONTACT_EMAIL=you@example.org` on Windows).

## Inputs

All are public and need no account (MANIFEST.csv lists each with its URL, size and SHA-256):
- The ATUS 2003 to 2025 multi-year files of the Bureau of Labor Statistics (released June 25, 2026): the activity
  summary, respondent, roster, ATUS-CPS and activity files, the data dictionaries, the user's guide and the activity
  lexicon.
- The Census occupation code lists of 2010 and 2018 with their SOC codes, and the BLS crosswalk from 2010 SOC to 2018
  SOC.
- Exposure to generative AI by occupation from Eloundou, Manning, Mishkin and Rock (2024, Science), the occupation file
  of github.com/openai/GPTs-are-GPTs: the GPT-4 rating and the human rating of beta (the share of an occupation's tasks
  that a language model could make at least 50 percent faster, counting tasks that need extra software at one half).
- The AI occupational exposure of Felten, Raj and Seamans (2021, Strategic Management Journal) and its
  language-modeling version (2023), from github.com/AIOE-Data/AIOE.

## Definitions

- Sample: diary years 2015 to 2025 without 2020 (BLS stopped collecting diaries from March 18 to May 9, 2020);
  employed (TELFS 1 or 2); aged 18 to 64. There are 48,018 respondents. In 2025 no diaries were collected in October,
  during the federal shutdown, so 2025 has 2,986.
- Weights: TUFNWGTP, the final weight that BLS documents for its multi-year files in every year except 2020, rescaled
  to a mean of 1 within each year so that each year counts by its number of diaries.
- Outcomes, in minutes of the diary day, from the six-digit activity codes: personal care including sleep (01);
  household activities (02); paid work and work-related activities (05); leisure (12 and 13); and all other activities,
  so that the five add up to 1,440 minutes. Leisure parts: television (120303, 120304); socializing (1201, 1202,
  120501, 120502); sports and exercise (13); computer leisure and games (120307, 120308); relaxing and thinking
  (120301); other leisure (the rest of 12).
- Worked at home that day: any minutes of work at a main or other job (0501) whose location was the respondent's home
  or yard (TEWHERE 1).
- Harmonized occupation: ATUS uses the 2010 Census codes up to 2019 and the 2018 codes from 2020. We link a 2010 code
  to a 2018 code when their SOC codes overlap after the BLS crosswalk, and treat each connected group of linked codes
  as one occupation (382 in all, 372 present in the sample). Each occupation's score is the mean over its 2018 codes,
  weighted by the 2021 to 2025 diaries in each code. Codes of the "all other" kind, which O*NET does not rate, take the
  mean of their SOC broad group. Every respondent maps to an occupation with all four scores (derived/mapping_coverage.csv).
- Exposure: the score standardized to mean 0 and standard deviation 1 across workers, or thirds of workers ranked by
  the score (the middle third is left out of the comparison of thirds).

## Models

Weighted least squares with harmonized-occupation fixed effects, year fixed effects, and controls for age, age
squared, sex, education (four groups), day of the week, holiday, month of the diary and whether the person worked at
home that day. Standard errors are clustered by harmonized occupation, with intervals from a t distribution with one
fewer degree of freedom than the number of occupations (fe.py). The smallest detectable effect is 2.8 standard errors
(80 percent power for a two-sided test at 5 percent). Holm-adjusted p-values cover the four outcomes that are free to
move (leisure, paid work, household activities, personal care).

| Name in results/estimates.csv | Before | After | Own terms |
|---|---|---|---|
| main | 2019, 2021, 2022 | 2024, 2025 | 2023 |
| prepandemic_baseline | 2015 to 2019 | 2024, 2025 | 2021, 2022, 2023 |
| long_pre | 2015 to 2019, 2021, 2022 | 2024, 2025 | 2023 |
| no_wfh_control | as main, without the worked-at-home control | | |
| subsample | as main, within weekdays, weekends, days worked at home, days worked only away from home, days without work | | |
| placebo_* | dates before ChatGPT: 2015-2016 against 2018-2019; 2015-2018 against 2019; 2021 against 2022 | | |
| event_study_ref2019, event_study_ref2022 | exposure times each year, against 2019 or 2022 | | |

## Outputs (results/)

- estimates.csv: every coefficient (spec, measure as score:z or score:third, outcome, sample, term, estimate,
  standard error, 95 percent interval, p, Holm p, smallest detectable effect, diaries, occupations, mean before).
- pretrend_tests.csv: joint tests that groups of event-study years are zero.
- descriptives.csv and yearly_means_by_third.csv: weighted means by third and period or year.
- key_numbers.json: sample sizes, the cut points of the thirds, the standard deviation of each score, and the adding-up
  check.
- figure1_event_study.png and .pdf (main figure); figureS1_measures.png, figureS2_leisure_parts.png,
  figureS3_yearly_means.png.
- table1.md, table1.csv, table1.png (main table).
- jpxz_estimates.csv, jpxz_key_numbers.json, figureS4_jpxz_event_study.png: the reproduction of Jiang et al. The
  script's docstring lists their specification as their text states it and each place where we had to choose
  (their own exposure labels are not public, so we use the GPT-4 rating of Eloundou et al. with the same rubric; our
  harmonized occupations stand in for occ1990dd). Columns: coef is weekly hours per percentile of exposure;
  iqr_hours_week is 50 times that, the shift from the 25th to the 75th percentile; iqr_minutes_day converts it to
  minutes a day. Standard errors are clustered by occupation and by state.
