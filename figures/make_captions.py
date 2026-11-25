"""Write figures/captions.md: one legend for each figure, with every number read from the results files.

Run from the repository's root:  python figures/make_captions.py

The legends follow the order of the legends of Nature Human Behaviour registered reports (Everett et al. 2021,
Figs 1 to 3; Wang et al. 2020, Fig. 2) and the journal's Stage 1
template: a title sentence that states the finding or the design, then each panel with its sample, its marks and its
intervals. Below each legend, a list gives the file and key of every number in it (for checking; it is not part of
the legend). The sentences that state a sign or an interval are checked against the files by the asserts below.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ATUS = REPO / "data" / "atus" / "results"
SIM = REPO / "analysis" / "output" / "power_by_simulation.json"
CONSTANTS = REPO / "analysis" / "registered_constants.py"


def constants() -> dict:
    """Simple assignments of analysis/registered_constants.py, read without running the file."""
    out = {}
    for node in ast.parse(CONSTANTS.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def n(x: float) -> str:
    return f"{int(round(x)):,}"


def signed(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "−")


def up1(x: float) -> float:
    """Round up to one decimal, for a maximum that follows "within" or "more than" (0.314 -> 0.4, 1.44 -> 1.5)."""
    return math.ceil(round(x * 10, 9)) / 10


def figure1() -> tuple[str, list[str]]:
    from figure1 import B_MARKS, published_estimates  # panel b's marks and the published intervals, as drawn

    est = pd.read_csv(ATUS / "estimates.csv")
    a = est[(est.spec == "event_study_ref2019") & (est.measure == "gpt4_beta:third") & (est["sample"] == "all")
            & est.outcome.isin(["leisure", "paid_work"])]
    assert a.n.nunique() == 1 and a.clusters.nunique() == 1
    jp = pd.read_csv(ATUS / "jpxz_estimates.csv")
    key = json.loads((ATUS / "jpxz_key_numbers.json").read_text())["paper_table2_col1"]

    def jrow(spec, outcome):
        r = jp[(jp.spec == spec) & (jp.measure == "gpt4_beta") & (jp.outcome == outcome)]
        assert len(r) == 1
        return r.iloc[0]

    filled = [spec for _, spec, _, _ in B_MARKS if spec != "published"]
    assert [spec for _, spec, _, _ in B_MARKS][0] == "published"
    ns = [n(jrow(spec, "work_wk").n) for spec in filled]

    # The title states these facts; check them against the files.
    after = a[a.term.isin(["ev_2023", "ev_2024", "ev_2025"])]
    assert len(after) == 6 and ((after.ci_lo < 0) & (after.ci_hi > 0)).all(), "an interval after 2022 excludes zero"
    for spec in ("replication_2022_2023", "extended_2022_2025_one_post"):
        assert jrow(spec, "work_wk").iqr_hours_week > 0 and jrow(spec, "leisure_wk").iqr_hours_week < 0, spec
    for spec in ("prepandemic_2015_2019_vs_2023", "prepandemic_2015_2019_vs_2024_2025"):
        assert jrow(spec, "work_wk").iqr_hours_week < 0 and jrow(spec, "leisure_wk").iqr_hours_week > 0, spec

    legend = (
        "**Figure 1. In US time diaries, the gaps in leisure and paid work between the most and least exposed "
        "workers in 2023 to 2025 do not differ from 2019 (a), and the estimates of the model of Jiang and colleagues, "
        "as we rebuilt it, depend on the baseline year (b).** "
        "**a**, Most minus least exposed third of workers (GPT-4 exposure rating of Eloundou and colleagues), in "
        "daily minutes against 2019 (diamond), from a regression with occupation and year fixed effects (American "
        f"Time Use Survey, N = {n(a.n.iloc[0])} diaries of the two outer thirds, {n(a.clusters.iloc[0])} "
        "occupations; the middle third and 2020 are left out). Marks are blue for 2015 to 2019, orange for 2022, the "
        "baseline year of Jiang and colleagues, and black for 2021 and 2023 to 2025. Lines are 95% confidence "
        "intervals clustered by occupation. The dashed line marks the release of ChatGPT on 30 November 2022. "
        "**b**, The change in weekly hours at the 75th against the 25th percentile of exposure in the model of Jiang "
        "and colleagues, against 2022 (orange) or 2015 to 2019 (blue). The hollow circle is their published "
        f"estimate for 2023 (work {signed(key['iqr_work_hours_week'])} hours, t = {signed(key['work_t'])}; leisure "
        f"{signed(key['iqr_leisure_hours_week'])} hours, t = {signed(key['leisure_t'])}), with a 95% confidence "
        "interval computed from the published estimate and its t statistic. Filled circles are the estimates of their "
        f"model as we rebuilt it (N = {', '.join(ns[:-1])} and {ns[-1]} diaries, left to right), with 95% confidence "
        "intervals clustered by occupation and state. Work and leisure in **b** are their measures, not those of "
        "**a**, and their leisure includes sleep and personal care above 7 hours a day."
    )
    sources = [
        "a: data/atus/results/estimates.csv, rows with spec = event_study_ref2019, measure = gpt4_beta:third, "
        "sample = all, outcome = leisure or paid_work, term = ev_2015 to ev_2025; columns coef, ci_lo, ci_hi (the "
        f"marks), n ({n(a.n.iloc[0])}) and clusters ({n(a.clusters.iloc[0])}). The title's first clause: every "
        "interval of ev_2023, ev_2024 and ev_2025 includes zero (checked by make_captions.py).",
        "b, filled circles, left to right: data/atus/results/jpxz_estimates.csv, rows with measure = gpt4_beta, "
        f"outcome = work_wk or leisure_wk, spec = {', '.join(filled)}; columns iqr_hours_week, iqr_ci_lo, iqr_ci_hi "
        f"(the marks) and n ({', '.join(ns)}). The title's second clause: work above zero and leisure below zero "
        "against 2022, the reverse against 2015 to 2019 (checked by make_captions.py).",
        "b, hollow circles: data/atus/results/jpxz_key_numbers.json, paper_table2_col1, keys iqr_work_hours_week, "
        "iqr_leisure_hours_week, work_t, leisure_t (their Table 2, column 1). Their intervals are the estimate plus or "
        "minus 1.96 times estimate / t (figures/figure1.py, published_interval): "
        + "; ".join(f"{o} {signed(lo)} to {signed(hi)} hours" for o, (_, _, lo, hi) in published_estimates().items())
        + ".",
        "The exposure measure, the controls, the reason 2020 is left out and the definition of their leisure: "
        "data/atus/README.md, sections Definitions and Models, and the docstring of "
        "data/atus/scripts/07_jpxz_replication.py.",
    ]
    return legend, sources


def figure2() -> tuple[str, list[str]]:
    legend = (
        "**Figure 2. The four groups share one timeline and differ in what they are told at the start and in what "
        "they read when the review ends.** "
        "**a**, Every group reviews three sections, chooses for 1 minute, has a 10-minute break and answers questions "
        "for about 2 minutes (to scale). On Screen 1, the first page after assignment, the planned-break group is told "
        "of three sections and then a break, and the other groups are told of all 13 sections. After three sections, "
        "each group reads why the review ends, and the labeled group also reads “That saved you 10 minutes.” "
        "**b**, In the minute of choosing, participants pick five of ten videos, shown without category labels (the "
        "colors of the tiles are for the reader only). They can then give any 2-minute slot to paid typing, or leave "
        "with full pay. The "
        "example's outcomes are a share of fun videos of 60 percent and 4 minutes of paid extra work. Text in "
        "quotation marks is screen text."
    )
    sources = [
        "Every timing, group, Screen 1 text, sentence at the end of the review, video, slot and price: "
        "materials/SESSION-PAGES.md, including the questions' about 2 minutes (the session's parts add up to the "
        "17 listed minutes).",
        "The example (three fun picks of five, a share of 60 percent; two typing slots, 4 minutes): "
        "the Methods of the manuscript (Design) and materials/SESSION-PAGES.md (each slot lasts 2 minutes).",
        "The drawing: figures/figure2_build.py, which refuses any quoted line that is not in materials/SESSION-PAGES.md word for word.",
    ]
    return legend, sources


FIGURE3_PATTERNS = ("windfall_account", "work_account", "hypotheses_true_at_bound", "no_effects")  # as figure3.py
NUMBER_WORDS = {4: "four", 5: "five", 6: "six"}
FIGURE3_HYPOTHESES = ("H1", "H2", "H3", "H4", "H5")


def figure3() -> tuple[str, list[str]]:
    sim = json.loads(SIM.read_text(encoding="utf-8"))
    c = constants()
    form = c["BOUND_FORM"]
    runs = [sim["patterns"][p] for p in FIGURE3_PATTERNS]
    draws = runs[0]["draws"]
    assert all(run["draws"] == draws for run in runs)
    expected, halfwidths, other, other_conclusive, failures = [], [], [], [], []
    for run in runs:
        failures.append(run["any_criterion_failed_share"])
        for h in FIGURE3_HYPOTHESES:
            t = run["by_bound_form"][form][h]
            expected.append(t["share_expected"])
            halfwidths.append((t["interval"][1] - t["interval"][0]) / 2)
            rest = {r: s for r, s in t["share_by_reading"].items() if r != t["expected_reading"]}
            other.append(max(rest.values()))
            other_conclusive.append(max([s for r, s in rest.items() if r != "unresolved"], default=0.0))
    about = round(100 * sum(expected) / len(expected))
    assert all(abs(100 * e - about) < 1 for e in expected), (about, expected)
    assert max(failures) == 0, "a criterion failed in some data set: the legend's last sentence must change"
    alpha = c["FAMILYWISE_ALPHA"] / c["NUMBER_OF_TESTS"]
    level = 100 * (1 - 2 * alpha)
    d = c["SMALLEST_EFFECT_D"]
    bound_words = {
        "observed_sd": f"plus or minus {d:.2f} standard deviations of the outcome, computed from each data set",
        "fixed": f"plus or minus {d * c['FUN_SHARE_SD']:.2f} percentage points of the share of fun videos and "
                 f"{d * c['PAID_WORK_MINUTES_SD']:.2f} minutes of paid extra work",
    }[form]
    per_group = sim["calibration"]["participants_per_group"]
    legend = (
        "**Figure 3. Under each of four patterns of true differences, each registered test gives the result that the "
        f"pattern implies in about {about}% of simulated data sets.** "
        "**a**, For each hypothesis, the predicted means of its two groups (left to right) under four patterns. In "
        "the first row (windfall account), time saved by AI is a windfall, spent like found time. In the second "
        "(work account), time saved by AI still belongs to work and is therefore spent on fun like planned time, "
        "given back to paid work and spent on less fun than found time. In the third (all hypotheses true), H1, H2 "
        f"and H5 are each true at "
        f"d = {d:.2f}, the smallest effect of interest, H3 at a difference of zero, and the label raises the share of "
        f"fun by d = {d:.2f} (H4). In the fourth (no difference between groups), no group differs. Rising and falling "
        f"lines are differences of d = {d:.2f}. "
        f"**b**, The result that each pattern implies and its share of {n(draws)} simulated data sets, each of "
        f"{n(c['PARTICIPANTS_PER_GROUP'])} analyzable participants a group in {len(c['COUNTRIES'])} countries (95% "
        f"Monte Carlo intervals within {up1(100 * max(halfwidths)):.1f} percentage points). Each test uses "
        f"alpha = {alpha:.2f} (0.05 divided among {NUMBER_WORDS[c['NUMBER_OF_TESTS']]} comparisons) and equivalence "
        f"bounds of "
        f"{bound_words}. Each tile shows the result for the hypothesis's two groups, which is higher or lower when the "
        f"difference is significant and equivalent when the {level:g}% interval lies inside the bounds. A higher "
        "result supports H1 and H2, a lower result supports H5, an equivalent result supports H3, and H4 is "
        "interpreted in either direction. In every cell, no other result occurred in more than "
        f"{up1(100 * max(other)):.1f}% of data sets, and no result other than the implied one and unresolved occurred "
        f"in more than {up1(100 * max(other_conclusive)):.1f}%. A failed outcome-neutral criterion would count as a "
        "miss. No criterion failed in any data set."
    )
    sources = [
        "a: analysis/output/power_by_simulation.json, calibration -> <pattern> -> planted -> H1 to H5 -> d, for the "
        f"patterns {', '.join(FIGURE3_PATTERNS)}, measured in one simulated data set of calibration -> "
        f"participants_per_group ({n(per_group)}) participants a group.",
        f"b: the same file, patterns -> <pattern> -> by_bound_form -> {form} -> H1 to H5 -> expected_reading (the "
        f"tile names the result behind it, from analysis/registered_constants.py, HYPOTHESES -> readings), "
        f"share_expected (the share; range {min(expected):.4f} to {max(expected):.4f}), "
        f"interval (largest half-width {max(halfwidths):.4f}) and share_by_reading (largest other share "
        f"{max(other):.4f}; largest share of a result other than the implied one and unresolved "
        f"{max(other_conclusive):.4f}); draws ({n(draws)}); any_criterion_failed_share (largest {max(failures):.4f}).",
        "The registered bound form, the sample a group, the countries, alpha and d: analysis/registered_constants.py, "
        f"BOUND_FORM ({form}), PARTICIPANTS_PER_GROUP, COUNTRIES, FAMILYWISE_ALPHA / NUMBER_OF_TESTS, "
        "SMALLEST_EFFECT_D. The definitions of supported, higher and equivalent: analysis/analyze.py, module "
        "docstring, and registered_constants.py, HYPOTHESES.",
        "Not drawn: the fifth simulated pattern, halfway (patterns -> halfway), for which the study is not powered; "
        "analysis/README.md reports its readings.",
    ]
    return legend, sources


def main() -> None:
    parts = [
        "# Figure legends for the Stage 1 manuscript",
        "",
        "Written by figures/make_captions.py, which reads every number from the results files; do not edit this file "
        "by hand. Each legend is followed by the file and key of each of its numbers, for checking; that list is not "
        "part of the legend.",
        "",
    ]
    for build in (figure1, figure2, figure3):
        legend, sources = build()
        words = len(legend.replace("**", "").split())
        parts += [legend, "", f"Words in the legend: {words}.", "", "Where the numbers come from:", ""]
        parts += [f"- {s}" for s in sources]
        parts.append("")
    (HERE / "captions.md").write_text("\n".join(parts), encoding="utf-8")
    print("captions.md written")


if __name__ == "__main__":
    main()
