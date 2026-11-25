"""A short mutation check on the reading rules, the outcome-neutral criteria and their gates.

Each mutant changes one registered threshold, one sign, one reading or one gate in a copy of analyze.py or
registered_constants.py. The fast
tests (test_reading_rules.py) run against the copy, and the mutant counts as killed only when a test assertion fails.
An import error, a syntax error or any other crash does not count. The unchanged copy runs first and must pass, and
every run checks that the modules it imported came from the copy.

Run from the repository's root: python analysis/tests/mutation_check.py
Writes analysis/output/mutation_check.txt and analysis/output/mutation_check.json.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
OUTPUT = ANALYSIS / "output"
SOURCES = ("analyze.py", "registered_constants.py", "simulate.py")
TEST_MODULE = "test_reading_rules"

# (what the mutant changes, file, original text, mutated text). Each original text occurs exactly once in its file.
RC, AN = "registered_constants.py", "analyze.py"
MUTANTS = [
    # Registered numbers.
    ("alpha split four ways instead of five (.0125)", RC, "NUMBER_OF_TESTS = 5", "NUMBER_OF_TESTS = 4"),
    ("smallest effect d = 0.20 instead of 0.15", RC, "SMALLEST_EFFECT_D = 0.15", "SMALLEST_EFFECT_D = 0.20"),
    ("expected fun-share bound from an SD of 33 instead of 29", RC, "FUN_SHARE_SD = 29.0", "FUN_SHARE_SD = 33.0"),
    # The tests and the reading rules.
    ("two-sided p computed as one-sided", AN,
     "return 2 * stats.t.sf(np.abs(estimate / se), df)", "return stats.t.sf(np.abs(estimate / se), df)"),
    ("significance read at twice the registered alpha", AN,
     "significant = two_sided_p(estimate, se, df) < alpha", "significant = two_sided_p(estimate, se, df) < 2 * alpha"),
    ("equivalence read at twice the registered alpha", AN,
     "equivalent = equivalence_p(estimate, se, df, bound) < alpha",
     "equivalent = equivalence_p(estimate, se, df, bound) < 2 * alpha"),
    ("sign flipped in the test against the lower bound", AN,
     "stats.t.sf((estimate + bound) / se, df)", "stats.t.sf((estimate - bound) / se, df)"),
    ("sign flipped in the test against the upper bound", AN,
     "stats.t.cdf((estimate - bound) / se, df)", "stats.t.cdf((estimate + bound) / se, df)"),
    ("higher and lower swapped in the result", AN,
     "significant & (estimate > 0), HIGHER", "significant & (estimate < 0), HIGHER"),
    ("H3 reads significance first, the others equivalence first", AN,
     "if equivalence_first:", "if not equivalence_first:"),
    ("H1: predicted direction flipped", RC,
     '"readings": {"higher": "supported", "lower": "opposite", "equivalent": "equivalent"},',
     '"readings": {"higher": "opposite", "lower": "supported", "equivalent": "equivalent"},'),
    ("H2: the goal-progress reading named 'opposite'", RC,
     '"readings": {"higher": "supported", "lower": "lower", "equivalent": "equivalent"},',
     '"readings": {"higher": "supported", "lower": "opposite", "equivalent": "equivalent"},'),
    ("H3: higher and lower than planned swapped", RC,
     '"readings": {"equivalent": "supported", "higher": "higher than planned", "lower": "lower than planned"},',
     '"readings": {"equivalent": "supported", "higher": "lower than planned", "lower": "higher than planned"},'),
    ("H4: made one-sided (a predicted direction) instead of two named predictions", RC,
     '"readings": {"higher": "higher", "lower": "lower", "equivalent": "equivalent"},',
     '"readings": {"higher": "supported", "lower": "opposite", "equivalent": "equivalent"},'),
    ("H5: predicted direction flipped", RC,
     '"readings": {"lower": "supported", "higher": "opposite", "equivalent": "equivalent"},',
     '"readings": {"lower": "opposite", "higher": "supported", "equivalent": "equivalent"},'),
    ("H5: AI-saved compared with the planned break instead of the found break", RC,
     '"H5": {"outcome": "fun_share", "minuend": "ai_saved", "subtrahend": "found_break"',
     '"H5": {"outcome": "fun_share", "minuend": "ai_saved", "subtrahend": "planned_break"'),
    # Criterion a, surprise.
    ("surprise: interval above zero instead of above 0.15 pooled SD", RC,
     "SURPRISE_MIN_D = 0.15", "SURPRISE_MIN_D = 0.0"),
    ("surprise: read from the upper end of the interval", AN,
     '"minimum": minimum, "passed": lower > minimum}', '"minimum": minimum, "passed": est + t_check * se > minimum}'),
    ("surprise: AI-saved less surprised read from the lower end", AN,
     '"ai_saved_less_surprised": est + t_check * se < -limit}', '"ai_saved_less_surprised": est - t_check * se < -limit}'),
    ("surprise: AI-saved less surprised when the interval is merely below zero", RC,
     "SURPRISE_WINDFALL_LIMIT_D = 0.15", "SURPRISE_WINDFALL_LIMIT_D = 0.0"),
    # H3 and H5 read as a pair: the nine rules of Supplementary Table S4 and the code that applies them in order.
    ("pair: rule 3 ('more fun than found time') tried before rule 2 ('less fun than a planned break')", RC,
     '    (2, PAIR_ANY, ("lower than planned",), PAIR_ANY, "and", PAIR_LESS_THAN_PLANNED),\n'
     '    (3, PAIR_ANY, PAIR_ANY, ("opposite",), "and", PAIR_MORE_THAN_FOUND),\n',
     '    (3, PAIR_ANY, PAIR_ANY, ("opposite",), "and", PAIR_MORE_THAN_FOUND),\n'
     '    (2, PAIR_ANY, ("lower than planned",), PAIR_ANY, "and", PAIR_LESS_THAN_PLANNED),\n'),
    ("pair: not interpretable only when both tests are (rule 1)", RC,
     '(1, PAIR_ANY, (NOT_INTERPRETABLE,), (NOT_INTERPRETABLE,), "or", NOT_INTERPRETABLE)',
     '(1, PAIR_ANY, (NOT_INTERPRETABLE,), (NOT_INTERPRETABLE,), "and", NOT_INTERPRETABLE)'),
    ("pair: unresolved only when both tests are (rule 4)", RC,
     '(4, PAIR_ANY, ("unresolved",), ("unresolved",), "or", "unresolved")',
     '(4, PAIR_ANY, ("unresolved",), ("unresolved",), "and", "unresolved")'),
    ("pair: 'spent like planned time' read even when H1 is not supported (rule 5)", RC,
     '(5, "supported",', '(5, PAIR_ANY,'),
    ("pair: 'spent like found time' named 'between the two breaks' (rule 6)", RC,
     '("higher than planned",), ("equivalent",), "and", "spent like found time"),',
     '("higher than planned",), ("equivalent",), "and", "between the two breaks"),'),
    ("pair: 'cannot be told apart from either break' made to need H1 supported (rule 8)", RC,
     '(8, PAIR_ANY,', '(8, "supported",'),
    ("pair: 'restricted' applied when H1 is supported (rule 9)", RC,
     '(9, "not supported",', '(9, "supported",'),
    ("pair: the 'or' between the H3 and H5 conditions read as 'and'", AN,
     'holds = (on_h3 | on_h5) if join == "or" else (on_h3 & on_h5)', "holds = on_h3 & on_h5"),
    ("pair: H1 'supported' read as 'not opposite'", AN,
     "holds = holds & (h1 == SUPPORTED)", "holds = holds & (h1 != OPPOSITE)"),
    ("pair: the last rule that holds wins instead of the first", AN,
     "return np.select(conditions, [rule[-1] for rule in rc.PAIR_RULES], default=rc.PAIR_NO_RULE).astype(object)",
     "return np.select(conditions[::-1], [rule[-1] for rule in rc.PAIR_RULES][::-1], "
     "default=rc.PAIR_NO_RULE).astype(object)"),
    ("pair: 'more fun than found time' read against both accounts again", RC,
     'PAIR_MORE_THAN_FOUND: "AI-saved time goes to more fun than found time, as the goal-progress account predicts.",',
     'PAIR_MORE_THAN_FOUND: "AI-saved time goes to more fun than found time, against both accounts.",'),
    # Participants who made no pick, and the robustness check among those who made all five picks.
    ("no pick: participants who made no pick left out of H2 too", AN,
     "        rows = data[data[o].notna()]\n", '        rows = data[data["fun_share"].notna()]\n'),
    ("no pick: the count by group counts every participant", AN,
     'summary["no_pick_made_by_group"] = {g: int(((made == 0) &',
     'summary["no_pick_made_by_group"] = {g: int(((made >= 0) &'),
    ("five picks: the robustness check leaves out H3", RC,
     'FIVE_PICKS_ROBUSTNESS_HYPOTHESES = ("H1", "H3", "H4", "H5")',
     'FIVE_PICKS_ROBUSTNESS_HYPOTHESES = ("H1", "H4", "H5")'),
    ("purchasing power computed the wrong way round", RC,
     '"international_dollars": PPP_ITEM_USD * rate / ppp}', '"international_dollars": PPP_ITEM_USD * ppp / rate}'),
    ("position between the breaks reported even when found minus planned is not significant", AN,
     '"reported": bool(p < rc.ALPHA)}', '"reported": True}'),
    # Criterion b, attribution.
    ("attribution: 70 percent instead of 80 in each AI-saved group", RC,
     "ATTRIBUTION_MIN_SHARE_AI_GROUPS = 0.80", "ATTRIBUTION_MIN_SHARE_AI_GROUPS = 0.70"),
    ("attribution: 'at least 80 percent' read as 'more than 80 percent'", AN,
     'attribution_ai_saved = ((share_ai["ai_saved"] >= rc.ATTRIBUTION_MIN_SHARE_AI_GROUPS)',
     'attribution_ai_saved = ((share_ai["ai_saved"] > rc.ATTRIBUTION_MIN_SHARE_AI_GROUPS)'),
    ("attribution: 30 percent instead of 20 in the found-break group", RC,
     "ATTRIBUTION_MAX_SHARE_FOUND_GROUP = 0.20", "ATTRIBUTION_MAX_SHARE_FOUND_GROUP = 0.30"),
    ("attribution: 40 percent instead of 50 choosing 'would not be reviewed'", RC,
     "ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED = 0.50", "ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED = 0.40"),
    ("attribution: the found-break part no longer asks for 'would not be reviewed'", AN,
     '& (share_not_reviewed["found_break"] >= rc.ATTRIBUTION_MIN_SHARE_FOUND_NOT_REVIEWED))', ")"),
    # Criterion c, the label.
    ("label: 10 points instead of 20", RC, "LABEL_MIN_DIFFERENCE = 0.20", "LABEL_MIN_DIFFERENCE = 0.10"),
    ("label: 'at least 20 points' read as 'more than 20 points'", AN,
     "(np.round(label_est, ROUND) >= rc.LABEL_MIN_DIFFERENCE)", "(np.round(label_est, ROUND) > rc.LABEL_MIN_DIFFERENCE)"),
    ("label: the interval above zero no longer asked for", AN, " & (label_lower > 0)", ""),
    ("label: direction flipped", AN,
     'model.fit(saw_label).contrast("ai_saved_labeled", "ai_saved")',
     'model.fit(saw_label).contrast("ai_saved", "ai_saved_labeled")'),
    # Criterion d, floor and ceiling.
    ("fun-share floor 10 instead of 15", RC, "FUN_SHARE_FLOOR = 15.0", "FUN_SHARE_FLOOR = 10.0"),
    ("fun-share ceiling 90 instead of 85", RC, "FUN_SHARE_CEILING = 85.0", "FUN_SHARE_CEILING = 90.0"),
    ("fun-share floor measured over all groups", AN,
     "planned_mean = np.nanmean(fun[planned], axis=0)", "planned_mean = np.nanmean(fun, axis=0)"),
    ("paid-work floor 1 percent instead of 5", RC,
     "PAID_WORK_MIN_SHARE_OF_FREE_MINUTES = 0.05", "PAID_WORK_MIN_SHARE_OF_FREE_MINUTES = 0.01"),
    ("paid-work floor measured over all groups", AN,
     "paid_share = work[planned].sum(axis=0) / (planned.sum() * rc.FREE_MINUTES)",
     "paid_share = work.sum(axis=0) / (len(g) * rc.FREE_MINUTES)"),
    # The gates.
    ("surprise of the found break no longer gates H1", RC,
     '"surprise_found_break": ("H1", "H2", "H5"),', '"surprise_found_break": ("H2", "H5"),'),
    ("surprise of the AI-saved group no longer gates H5", RC,
     '"surprise_ai_saved": ("H2", "H3", "H5"),', '"surprise_ai_saved": ("H2", "H3"),'),
    ("surprise of the labeled group gates nothing", RC,
     '"surprise_ai_saved_labeled": ("H4",),', '"surprise_ai_saved_labeled": (),'),
    ("attribution, AI-saved part, no longer gates H5", RC,
     '"attribution_ai_saved": ("H2", "H3", "H4", "H5"),', '"attribution_ai_saved": ("H2", "H3", "H4"),'),
    ("attribution, found-break part, gates H2 only", RC,
     '"attribution_found_break": ("H1", "H2", "H5"),', '"attribution_found_break": ("H2",),'),
    ("the label gates nothing", RC, '"label": ("H4",),', '"label": (),'),
    ("the fun-share floor no longer gates H5", RC,
     '"fun_share_floor_ceiling": ("H1", "H3", "H4", "H5"),', '"fun_share_floor_ceiling": ("H1", "H3", "H4"),'),
    ("the paid-work floor gates nothing", RC, '"paid_work_floor": ("H2",),', '"paid_work_floor": (),'),
    ("a failed criterion reads unresolved instead of not interpretable", AN,
     "gated[h] = np.where(failed, NOT_INTERPRETABLE, gated[h])", "gated[h] = np.where(failed, UNRESOLVED, gated[h])"),
    ("a failed criterion changes no reading", AN,
     "gated[h] = np.where(failed, NOT_INTERPRETABLE, gated[h])", "gated[h] = np.where(failed, gated[h], gated[h])"),
    # Attrition, Lee bounds, the export and the robustness rule.
    ("attrition threshold 5 points instead of 3", RC,
     "ATTRITION_DIFFERENCE_THRESHOLD = 0.03", "ATTRITION_DIFFERENCE_THRESHOLD = 0.05"),
    ("attrition: 'more than 3 points' read as 'at least 3 points'", AN,
     "bool(round(gap, ROUND) > rc.ATTRITION_DIFFERENCE_THRESHOLD)",
     "bool(round(gap, ROUND) >= rc.ATTRITION_DIFFERENCE_THRESHOLD)"),
    ("Lee bounds trim the group that kept fewer", AN,
     "if retained_minuend >= retained_subtrahend:", "if retained_minuend < retained_subtrahend:"),
    ("Lee bounds trim from the wrong end", AN,
     "low, high = a[: len(a) - k].mean() - b.mean(), a[k:].mean() - b.mean()",
     "low, high = a[k:].mean() - b.mean(), a[: len(a) - k].mean() - b.mean()"),
    ("picks filled at random counted in the share of fun", AN,
     "recomputed = 100.0 * (is_fun & ~filled).sum(axis=1) / made",
     "recomputed = 100.0 * is_fun.sum(axis=1) / rc.VIDEO_PICKS"),
    ("a participant with filled picks may have work slots", AN,
     "if ((made < rc.VIDEO_PICKS) & ~(uses == \"watch\").all(axis=1)).any():", "if False:"),
    ("a hidden-time value without video minutes is accepted", AN,
     "if (np.nan_to_num(hidden[video_minutes == 0]) != 0).any():", "if False:"),
    ("leaving early allowed on some slots only", AN,
     "if not np.isin(n_left, (0, rc.VIDEO_PICKS)).all():", "if False:"),
    ("hidden time divided by all 10 minutes instead of the video minutes", AN,
     "hidden / np.maximum(video_minutes, 1e-12)", "hidden / rc.FREE_MINUTES"),
    ("robustness rule drops at more than 80 percent instead of half", RC,
     "HIDDEN_SHARE_OF_VIDEO_MINUTES_MAX = 0.5", "HIDDEN_SHARE_OF_VIDEO_MINUTES_MAX = 0.8"),
]


def run_tests(folder):
    env = {**os.environ, "MUTATION_CHECK_DIR": str(folder), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "-m", "unittest", TEST_MODULE], cwd=folder / "tests", env=env,
                          capture_output=True, text=True, timeout=300)
    return done.returncode, done.stdout + done.stderr


def classify(returncode, output):
    failures = output.count("\nFAIL: ")
    errors = output.count("\nERROR: ")
    if returncode == 0:
        return "survived", failures, errors
    if failures:
        return "killed", failures, errors
    return "crashed (not counted as killed)", failures, errors


def main():
    work = Path(tempfile.mkdtemp(prefix="mutation_check_"))
    rows = []
    try:
        folder = work / "analysis"
        (folder / "tests").mkdir(parents=True)
        for name in SOURCES:
            shutil.copy2(ANALYSIS / name, folder / name)
        shutil.copy2(ANALYSIS / "tests" / f"{TEST_MODULE}.py", folder / "tests" / f"{TEST_MODULE}.py")
        shutil.copytree(ANALYSIS / "reference", folder / "reference")  # the World Bank files that a test reads
        originals = {name: (folder / name).read_text(encoding="utf-8") for name in SOURCES}

        code, output = run_tests(folder)
        if code != 0:
            print(output)
            raise SystemExit("the unchanged code fails its own tests; the mutation check cannot be read")
        baseline = "passed"

        for what, name, before, after in MUTANTS:
            for n, text in originals.items():
                (folder / n).write_text(text, encoding="utf-8")
            count = originals[name].count(before)
            if count != 1:
                raise SystemExit(f"mutant '{what}': the original text occurs {count} times in {name}, not once")
            (folder / name).write_text(originals[name].replace(before, after), encoding="utf-8")
            code, output = run_tests(folder)
            status, failures, errors = classify(code, output)
            rows.append({"mutant": what, "file": name, "status": status, "failing_tests": failures,
                         "tests_with_errors": errors})
            print(f"{status:>9}  {what}  ({failures} failing tests)")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    killed = sum(r["status"] == "killed" for r in rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {"baseline": baseline, "mutants": len(rows), "killed": killed, "rows": rows}
    (OUTPUT / "mutation_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"Mutation check on the reading rules: {killed} of {len(rows)} mutants killed "
             f"(the unchanged code {baseline} its tests first).", ""]
    lines += [f"{r['status']:>9}  {r['mutant']} ({r['file']}; {r['failing_tests']} failing tests)" for r in rows]
    (OUTPUT / "mutation_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[0])
    return 0 if killed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
