# Data and code for "Is time saved by an AI assistant spent on leisure or given back to work? A test across 12 countries"

This repository contains the materials and code underlying a Stage 1 Registered Report, which has two parts. The first
part is an analysis of the American Time Use Survey that we completed before submission. In it, we compare how U.S.
workers in the occupations most exposed to generative AI spent their day with how workers in the least exposed
occupations spent theirs, before and after the release of ChatGPT. The second part is an experiment, which we have not
yet run. In the experiment, participants review a dinner plan that an AI assistant drafted, and they then have 10 free
minutes, which come as a planned break, as a found break or as time that the assistant saved them.

We have collected no data in the experiment. This repository contains every page that each group of the experiment
sees, the registered analysis code with its tests and its output on simulated data, the code of the sample size and the
power analysis with its output, the code that downloads and analyzes the time diaries with its manifest and its
results, and the code that draws Figures 1 to 3. At Stage 2, the raw data of the experiment will be added.

## The folders

| Folder or file | What it holds | Manuscript |
|---|---|---|
| `materials/SESSION-PAGES.md` | Every page that each of the four groups sees, in order, with the time at which it appears and its exact text | Methods, Design |
| `analysis/` | The registered analysis (`analyze.py`), every registered number (`registered_constants.py`), the simulator of data in the format of the survey's export, the power analysis by simulation, the tests and the mutation check, their output on simulated data (`output/`), and the World Bank files behind the value of the pay in each country (`reference/`) | Methods, Sampling plan and Analysis plan; Table 1; Fig. 3 |
| `power/` | The sample size, the power of each test and the participant pay behind Table 1 (`design_table_power.py` and its output) | Methods, Sampling plan; Table 1 |
| `data/atus/` | The time diaries: the scripts that download and analyze them, the manifest of every input with its URL, size and SHA-256, the derived occupation files and the results | Methods, Analysis of national time diaries; Fig. 1; Supplementary Information |
| `figures/` | Figures 1 to 3, the scripts that draw them, their legends and their checks | Figs. 1 to 3 |
| `requirements.txt` | The Python packages, at the versions that we tested | |
| `LICENSE` | The licenses of the code and of the materials | |

The README.md file in each of the folders `analysis/`, `data/atus/` and `figures/` describes every file in that folder.

## Software

The code requires Python 3.12, and the file `requirements.txt` lists the packages that you must install to run it. The
code has been run with Python 3.12.10 (64-bit) on Windows 11, in a new virtual environment made from `requirements.txt`
alone, except the steps of the time diaries that need the raw files, which were run with Python 3.11 (see
`data/atus/README.md`). The figures also require the Arial font, and Figure 2 requires Microsoft Edge or Google Chrome, which prints its
drawing to PDF (see `figures/README.md`). An internet connection is required only to download the time diaries. No
account or key is required, and the code runs on a standard desktop or laptop computer without a GPU.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate   (macOS, Linux)
pip install -r requirements.txt
```

## Running each part

Run every command below from the root directory of the repository.

**The experiment's registered analysis.** To run the analysis on simulated data, execute the following scripts in this
order:

```
python analysis/simulate.py --all
python analysis/analyze.py analysis/output/simulated_no_effects.csv
python analysis/analyze.py analysis/output/simulated_hypotheses_true_at_bound.csv
python analysis/analyze.py analysis/output/simulated_windfall_account.csv
python analysis/analyze.py analysis/output/simulated_work_account.csv
python analysis/analyze.py analysis/output/simulated_halfway.csv
python analysis/power_by_simulation.py
python analysis/tests/run_tests.py
python analysis/tests/mutation_check.py
```

The first command simulates five data sets in the format of the survey's export, each under one pattern of true
differences. Each of the next five commands runs the registered analysis on one of these data sets and saves its
results in `analysis/output/results_<pattern>.json` and `analysis/output/report_<pattern>.txt`.
`power_by_simulation.py` runs the same analysis on many simulated data sets under each pattern and calculates how often
each test gives each result. `run_tests.py` runs the tests of the analysis, and their log is saved in
`analysis/output/tests.txt`. `mutation_check.py` changes one threshold or one sign at a time in a copy of the code and
checks that a test fails after each change. On our computer, computation took about 6 seconds for the simulator, about
3 seconds for each analysis, about 6.5 minutes for the power analysis, about 25 seconds for the tests and about
2 minutes for the mutation check. Every random draw is made from a fixed seed, so a second run writes the same
files. Once the data have been collected, the registered analysis is one command,
`python analysis/analyze.py <the survey's export>.csv`.

**The sample size.** `python power/design_table_power.py` prints the sample size, the power of each test and the
participant pay in about a second, and its output is saved in `power/design_table_power_output.txt`.

**The time diaries.** To download every input and run the whole analysis again, execute:

```
export BLS_CONTACT_EMAIL=you@example.org        (Windows: set BLS_CONTACT_EMAIL=you@example.org)
python -I data/atus/scripts/run_all.py
```

`run_all.py` downloads the 15 public inputs (226.6 MB) into `data/atus/raw/` and writes `data/atus/MANIFEST.csv`, which
records the URL, size and SHA-256 of each file. The script then builds the sample, scores the exposure of each
occupation, runs every model, draws the supplementary figures and the table, and checks the results.
`data/atus/README.md` describes each step and its run time. The Bureau of Labor Statistics refuses a download whose
User-Agent holds no contact e-mail, so the download script reads an address from `BLS_CONTACT_EMAIL`. Two steps can be
run without the download: `python -I data/atus/scripts/test_fe.py` tests the estimator, and
`python -I data/atus/scripts/05_figures_tables.py` draws the supplementary figures and the table again from
`data/atus/results/`.

**The figures.** To draw Figures 1 to 3 and write their legends, execute:

```
python figures/figure1.py
python figures/figure2_build.py
python figures/figure3.py
python figures/make_captions.py
python figures/run_qa.py
```

`figure1.py` takes its inputs from `data/atus/results/`, and `figure3.py` takes its inputs from
`analysis/output/power_by_simulation.json`. `figure2_build.py` checks every line that it quotes as screen text
against `materials/SESSION-PAGES.md`. `make_captions.py` writes the three legends into `figures/captions.md` from
the same files, and `run_qa.py` checks every figure (see `figures/README.md`). Each command takes a few seconds.

## Outputs and the manuscript

| Output | Script | Manuscript |
|---|---|---|
| `power/design_table_power_output.txt` | `power/design_table_power.py` | Table 1 and Methods, Sampling plan: the sample size, the power of each test and the participant pay |
| `analysis/output/power_by_simulation.json`, `analysis/output/power_by_simulation.txt` | `analysis/power_by_simulation.py` | Methods, Sampling plan: the simulation study; Fig. 3 |
| `analysis/output/results_<pattern>.json`, `analysis/output/report_<pattern>.txt` | `analysis/analyze.py` | Methods, Analysis plan: the registered analysis, run on simulated data |
| `analysis/output/tests.txt`, `analysis/output/mutation_check.txt` | `analysis/tests/run_tests.py`, `analysis/tests/mutation_check.py` | the tests of the analysis code |
| `data/atus/results/estimates.csv`, `jpxz_estimates.csv`, `jpxz_key_numbers.json` | `data/atus/scripts/04_analysis.py`, `07_jpxz_replication.py` | Methods, Analysis of national time diaries; Fig. 1; Supplementary Information |
| `data/atus/results/table1.csv`, `figureS1_measures.png` to `figureS3_yearly_means.png` | `data/atus/scripts/05_figures_tables.py` | Supplementary Table S1 and Supplementary Figs. S1 to S3 |
| `data/atus/results/figureS4_jpxz_event_study.png` | `data/atus/scripts/07_jpxz_replication.py` | Supplementary Fig. S4 |
| `figures/figure1.pdf`, `figures/figure2.pdf`, `figures/figure3.pdf` | `figures/figure1.py`, `figures/figure2_build.py`, `figures/figure3.py` | Figs. 1 to 3 |
| `figures/captions.md` | `figures/make_captions.py` | the figure legends |

## What the repository does not hold

- The raw files of the American Time Use Survey and the other public inputs, which `data/atus/scripts/01_download.py`
  downloads, are not included, and neither is `data/atus/derived/sample.parquet`, which
  `data/atus/scripts/02_build_sample.py` builds from them. `data/atus/MANIFEST.csv` gives the URL, size and SHA-256 of
  each input.
- The data of the experiment are not included, because we have not collected them.
- The dinner plan that participants review, the typing items of the paid extra work, the ten videos of Chung and
  colleagues' fifth study and the translations of the materials are not included. `materials/SESSION-PAGES.md`
  describes each of them, and we prepare them before data collection.

## Tested from a fresh copy

Before deposit, we copied the repository to an empty folder and made a new virtual environment there from
`requirements.txt` alone. From the copy, we ran every command in "Running each part" except `run_all.py`, which needs
the download, and we compared every file that the commands wrote with the file in this repository. The simulated data
sets, the results and reports of the analysis, the output of the sample size and the table of the time diaries
(`data/atus/results/table1.csv`) are identical to those in this repository. Three files differ only in the run times
that they print (`analysis/output/power_by_simulation.json`, `analysis/output/power_by_simulation.txt` and
`analysis/output/tests.txt`), the PDF files of Figures 1 to 3 differ only in their creation dates, and the SVG files
differ only in their dates and in the random names of their clipping paths and markers. The supplementary figures that
`05_figures_tables.py` drew again show the same values with small shifts of their layout, because the files in this
repository were drawn with matplotlib 3.10.8 and not with the 3.11.1 of `requirements.txt`. All tests passed, a test
failed for every change that the mutation check made, and every figure passed its checks.

## The review copy

This copy is for anonymous review, so the download script of the time diaries holds no contact address. Instead, it
reads the address from the environment variable `BLS_CONTACT_EMAIL`.

## License

All code is licensed under the MIT License, and the materials and results are licensed under CC BY 4.0 (`LICENSE`). The
five check scripts in `figures/vendor/` keep their Apache License 2.0.
