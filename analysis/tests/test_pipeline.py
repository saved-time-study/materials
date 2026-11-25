"""Tests of the whole analysis on simulated data in the survey's export format.

Run: python -m unittest discover -s analysis/tests -v
"""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze as an  # noqa: E402
import registered_constants as rc  # noqa: E402
import simulate as sim  # noqa: E402

# The columns that make up one participant's free minutes; copied together so that each row stays consistent.
OUTCOME_COLUMNS = an.PICK_COLUMNS + an.FILLED_COLUMNS + an.SLOT_COLUMNS + ["left_early", "fun_share",
                                                                           "paid_work_minutes",
                                                                           "minutes_video_page_hidden"]


def analyzable_mask(data):
    return (data["passed_attention_check"].astype(bool) & data["finished"].astype(bool)
            & ~data["withdrew"].astype(bool))


def constructed_null(seed=11):
    """A simulated data set in which, within each country, every group has exactly the planned group's free minutes.

    The true and the estimated differences are then exactly zero, so the test does not depend on the seed.
    """
    data = sim.simulate_dataset("no_effects", seed)
    keep = analyzable_mask(data)
    for country in rc.COUNTRIES:
        here = keep & (data["country"] == country)
        planned_rows = data.index[here & (data["group"] == "planned_break")]
        for group in rc.GROUPS[1:]:
            rows = data.index[here & (data["group"] == group)]
            for column in OUTCOME_COLUMNS:
                data.loc[rows, column] = data.loc[planned_rows, column].to_numpy()
    return data


def readings(data):
    return {h: t["reading"] for h, t in an.analyze(data)["hypotheses"].items()}


NULL_READINGS = {"H1": an.EQUIVALENT, "H2": an.EQUIVALENT, "H3": an.SUPPORTED, "H4": an.EQUIVALENT,
                 "H5": an.EQUIVALENT}


class NullAndPlantedEffects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.null = constructed_null()
        cls.null_results = an.analyze(cls.null)

    def test_null_reads_equivalent(self):
        h = self.null_results["hypotheses"]
        self.assertEqual({k: v["reading"] for k, v in h.items()}, NULL_READINGS)
        for t in h.values():
            self.assertAlmostEqual(t["estimate"], 0.0, places=9)
            self.assertEqual(t["not_interpretable_because"], [])

    def test_null_pair_reading(self):
        pair = self.null_results["pair_h3_h5"]
        self.assertEqual(pair["reading"], "cannot be told apart from either break")
        self.assertEqual(pair["account"], rc.PAIR_ACCOUNTS[pair["reading"]])

    def test_planted_pairs(self):
        for name, expected in (("hypotheses_true_at_bound", "spent like planned time"),
                               ("windfall_account", "spent like found time")):
            data = sim.simulate_dataset(name, 15, replace(sim.Assumptions(), effect_d=0.30))
            self.assertEqual(an.analyze(data)["pair_h3_h5"]["reading"], expected, name)

    def test_null_reads_equivalent_with_the_fixed_bound_too(self):
        h = an.analyze(self.null, bound_form="fixed")["hypotheses"]
        self.assertEqual({k: v["reading"] for k, v in h.items()}, NULL_READINGS)

    def test_planted_effects_are_found(self):
        # twice the smallest effect: each test then has power above 0.999
        data = sim.simulate_dataset("hypotheses_true_at_bound", 12, replace(sim.Assumptions(), effect_d=0.30))
        self.assertEqual(readings(data), {"H1": an.SUPPORTED, "H2": an.SUPPORTED, "H3": an.SUPPORTED,
                                          "H4": an.HIGHER, "H5": an.SUPPORTED})

    def test_planted_windfall(self):
        data = sim.simulate_dataset("windfall_account", 13, replace(sim.Assumptions(), effect_d=0.30))
        r = readings(data)
        self.assertEqual(r["H3"], an.HIGHER_THAN_PLANNED)
        self.assertEqual(r["H5"], an.EQUIVALENT)

    def test_planted_goal_progress_reads_lower_for_h2(self):
        pattern = {"fun_share_steps": dict(sim.ZERO),
                   "paid_work_steps": {"planned_break": 1, "found_break": 1, "ai_saved": 0, "ai_saved_labeled": 0}}
        data = sim.simulate_dataset(None, 14, replace(sim.Assumptions(), effect_d=0.30), pattern=pattern)
        self.assertEqual(readings(data)["H2"], an.LOWER)

    def test_numpy_regression_matches_statsmodels_on_unbalanced_data(self):
        data, _ = an.apply_exclusions(sim.simulate_dataset("hypotheses_true_at_bound", 14))
        data = data.drop(index=data.index[::7]).reset_index(drop=True)  # unequal groups and countries
        self.assertTrue(data["fun_share"].isna().any())  # some made no pick: the fun-share rows differ
        tests = an.confirmatory_on(data)
        reference = an.statsmodels_contrasts(data)
        for h in rc.HYPOTHESES:
            self.assertAlmostEqual(float(tests[h]["estimate"][0]), reference[h]["estimate"], places=10)
            self.assertAlmostEqual(float(tests[h]["se"][0]), reference[h]["se"], places=10)
            self.assertEqual(tests[h]["df"], reference[h]["df"])


class Exclusions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sim.simulate_dataset("no_effects", 21)

    def test_failed_attention_unfinished_and_withdrawn_are_excluded(self):
        d = self.data
        failed = d.loc[~d["passed_attention_check"], "participant_id"]
        unfinished = d.loc[d["passed_attention_check"] & ~d["finished"].astype(bool), "participant_id"]
        withdrawn = d.loc[d["finished"].astype(bool) & d["withdrew"].astype(bool), "participant_id"]
        for ids in (failed, unfinished, withdrawn):
            self.assertGreater(len(ids), 0)
        analyzed, summary = an.apply_exclusions(d)
        for ids in (failed, unfinished, withdrawn):
            self.assertFalse(analyzed["participant_id"].isin(ids).any())
        self.assertEqual(summary["failed_attention_check"], len(failed))
        self.assertEqual(summary["did_not_finish"], len(unfinished))
        self.assertEqual(summary["withdrew"], len(withdrawn))
        self.assertEqual(summary["analyzed_by_group"], {g: rc.PARTICIPANTS_PER_GROUP for g in rc.GROUPS})
        self.assertTrue(summary["every_group_at_least_registered_size"])

    def test_no_one_is_excluded_on_answers_after_assignment(self):
        data = self.data.copy()
        target = data.index[analyzable_mask(data) & (data["group"] == "ai_saved")][:30]
        data.loc[target, "how_review_ended"] = "do_not_know"
        data.loc[target, "saw_saved_you_10_minutes"] = True
        data.loc[target, "surprise_unexpected"] = 1
        data.loc[target, "saw_session_will_restart"] = True
        analyzed, summary = an.apply_exclusions(data)
        self.assertTrue(data.loc[target, "participant_id"].isin(analyzed["participant_id"]).all())
        self.assertEqual(summary["analyzed"], len(rc.GROUPS) * rc.PARTICIPANTS_PER_GROUP)

    def stops(self, column, value, rows=None):
        data = self.data.copy()
        row = data.index[analyzable_mask(data)][0] if rows is None else rows(data)
        data.loc[row, column] = value
        with self.assertRaises(an.DataError):
            an.apply_exclusions(data)

    def test_missing_answers_stop_the_analysis(self):
        for column in ("how_review_ended", "saw_session_will_restart", "ai_use", "slot_3_use", "pick_2_filled"):
            self.stops(column, np.nan)
        with_videos = lambda d: d.index[analyzable_mask(d) & d["fun_share"].notna()  # noqa: E731
                                        & (d[an.SLOT_COLUMNS] == "watch").any(axis=1)][0]
        self.stops("fun_share", np.nan, with_videos)
        self.stops("minutes_video_page_hidden", np.nan, with_videos)

    def test_missing_employment_does_not_stop_the_analysis(self):
        data = self.data.copy()
        data.loc[data.index[analyzable_mask(data)][:5], "employment_status"] = np.nan
        _, summary = an.apply_exclusions(data)
        self.assertGreaterEqual(summary["employment_unknown"], 5)

    def test_finisher_without_a_group_stops_the_analysis(self):
        self.stops("group", "unknown_group")

    def test_fun_share_must_match_the_five_picks(self):
        self.stops("fun_share", 120.0, lambda d: d.index[analyzable_mask(d) & (d["fun_share"] < 100)][0])

    def test_paid_minutes_must_match_the_work_slots(self):
        self.stops("paid_work_minutes", 4.0, lambda d: d.index[analyzable_mask(d) & (d["paid_work_minutes"] == 0)
                                                               & ~d["left_early"].astype(bool)][0])

    def test_left_early_must_match_the_slots(self):
        self.stops("left_early", True, lambda d: d.index[analyzable_mask(d) & ~d["left_early"].astype(bool)][0])

    def test_hidden_minutes_cannot_exceed_the_video_minutes(self):
        self.stops("minutes_video_page_hidden", 3.0, lambda d: d.index[analyzable_mask(d)
                                                                       & d["left_early"].astype(bool)][0])

    def test_answers_outside_the_options_stop_the_analysis(self):
        self.stops("how_review_ended", "the AI did it")
        self.stops("surprise_surprising", 8)
        self.stops("ai_use", "Once a week or more")  # an overlapping option of an earlier wording is not an option


class FailedCriteriaMakeTheirTestsNotInterpretable(unittest.TestCase):
    def setUp(self):
        self.data = constructed_null(seed=31)
        self.keep = analyzable_mask(self.data)

    def rows(self, group):
        return self.data.index[self.keep & (self.data["group"] == group)]

    def expect(self, failed_tests, results=None):
        results = results or an.analyze(self.data)
        r = {h: t["reading"] for h, t in results["hypotheses"].items()}
        expected = {h: (an.NOT_INTERPRETABLE if h in failed_tests else NULL_READINGS[h]) for h in rc.HYPOTHESES}
        self.assertEqual(r, expected)
        for h in failed_tests:  # the estimate and its intervals are still reported
            t = results["hypotheses"][h]
            self.assertTrue(np.isfinite(t["estimate"]) and np.isfinite(t["equivalence_interval"]).all())
            self.assertEqual(t["reading_before_criteria"], NULL_READINGS[h])
        return results

    def test_failed_ai_saved_attribution(self):
        rows = self.rows("ai_saved")
        self.data.loc[rows[: len(rows) // 2], "how_review_ended"] = "do_not_know"  # about 45 percent say AI
        results = self.expect({"H2", "H3", "H4", "H5"})
        self.assertEqual(results["pair_h3_h5"]["reading"], an.NOT_INTERPRETABLE)
        self.assertFalse(results["checks"]["passed"]["attribution_ai_saved"])
        self.assertTrue(results["checks"]["passed"]["attribution_found_break"])

    def test_failed_found_break_attribution(self):
        rows = self.rows("found_break")
        self.data.loc[rows[: int(len(rows) * 0.4)], "how_review_ended"] = "as_planned_after_three_sections"
        results = self.expect({"H1", "H2", "H5"})
        self.assertLess(results["checks"]["attribution"]["share_choosing_not_reviewed"]["found_break"], 0.5)

    def test_failed_label_check(self):
        self.data.loc[self.rows("ai_saved_labeled"), rc.LABEL_SENTENCE] = (
            self.data.loc[self.rows("ai_saved"), rc.LABEL_SENTENCE].to_numpy())
        self.expect({"H4"})

    def test_failed_surprise_in_one_group(self):
        for group, failed in (("found_break", {"H1", "H2", "H5"}), ("ai_saved_labeled", {"H4"})):
            data = self.data.copy()
            data.loc[self.rows(group), an.SURPRISE_ITEMS] = data.loc[self.rows("planned_break"),
                                                                     an.SURPRISE_ITEMS].to_numpy()
            self.expect(failed, an.analyze(data))

    def test_failed_paid_work_floor_in_the_planned_group(self):
        rows = self.rows("planned_break")
        slots = self.data.loc[rows, an.SLOT_COLUMNS].replace("paid_work", "watch")
        self.data.loc[rows, an.SLOT_COLUMNS] = slots.to_numpy()
        self.data.loc[rows, "paid_work_minutes"] = 0.0
        watching = rows[(slots == "watch").any(axis=1).to_numpy()]
        self.data.loc[watching, "minutes_video_page_hidden"] = self.data.loc[watching,
                                                                             "minutes_video_page_hidden"].fillna(0.0)
        results = an.analyze(self.data)
        self.assertEqual(results["checks"]["paid_work_floor"]["planned_break_share_of_free_minutes"], 0.0)
        r = {h: t["reading"] for h, t in results["hypotheses"].items()}
        self.assertEqual(r["H2"], an.NOT_INTERPRETABLE)
        self.assertEqual(results["hypotheses"]["H2"]["not_interpretable_because"], ["paid_work_floor"])
        for h in ("H1", "H3", "H4", "H5"):
            self.assertNotEqual(r[h], an.NOT_INTERPRETABLE)

    def test_failed_fun_share_floor(self):
        low = replace(sim.Assumptions(), fun_share_planned_mean=10.0, fun_share_within_country_sd=15.0)
        results = an.analyze(sim.simulate_dataset("no_effects", 32, low))
        self.assertFalse(results["checks"]["passed"]["fun_share_floor_ceiling"])
        for h in ("H1", "H3", "H4", "H5"):
            self.assertEqual(results["hypotheses"][h]["reading"], an.NOT_INTERPRETABLE)
            self.assertIn("fun_share_floor_ceiling", results["hypotheses"][h]["not_interpretable_because"])
        self.assertEqual(results["hypotheses"]["H2"]["not_interpretable_because"], [])

    def test_ai_saved_less_surprised_limits_h3_and_h5(self):
        rows = self.rows("ai_saved")
        self.data.loc[rows, "surprise_unexpected"] = np.clip(self.data.loc[rows, "surprise_unexpected"] - 1, 1, 7)
        self.data.loc[rows, "surprise_surprising"] = np.clip(self.data.loc[rows, "surprise_surprising"] - 1, 1, 7)
        results = an.analyze(self.data)
        self.assertTrue(results["checks"]["surprise_ai_saved_minus_found"]["ai_saved_less_surprised"])
        self.assertTrue(results["checks"]["passed"]["surprise_ai_saved"])
        h = results["hypotheses"]
        self.assertEqual(h["H3"]["reading"], an.SUPPORTED)  # would count against the windfall account
        self.assertTrue(h["H3"]["not_read_against_windfall"])
        self.assertTrue(any("windfall" in note for note in h["H3"]["notes"]))
        self.assertFalse(h["H5"]["not_read_against_windfall"])  # H5 reads equivalent, the windfall prediction
        self.assertFalse(h["H1"]["not_read_against_windfall"])


class AttritionAndSecondary(unittest.TestCase):
    def test_lee_bounds_only_when_attrition_differs_by_more_than_3_points(self):
        plain = an.analyze(sim.simulate_dataset("no_effects", 51))
        self.assertFalse(plain["attrition"]["threshold_crossed"])
        self.assertNotIn("lee_bounds", plain["secondary"])
        rates = {g: (0.10 if g == "ai_saved" else 0.03) for g in rc.GROUPS}
        uneven = an.analyze(sim.simulate_dataset("no_effects", 52, replace(sim.Assumptions(), not_finished_rate=rates)))
        self.assertTrue(uneven["attrition"]["threshold_crossed"])
        self.assertEqual(set(uneven["secondary"]["lee_bounds"]), set(rc.HYPOTHESES))
        for h, b in uneven["secondary"]["lee_bounds"].items():
            est = uneven["hypotheses"][h]["estimate"]
            self.assertLessEqual(b["lower"], b["upper"])
            if h in ("H2", "H3", "H5", "H4"):  # each compares a group with the ai_saved group, which lost more
                self.assertGreater(b["trimmed_share"], 0.0)
                self.assertLess(b["lower"], est + 1.0)
        table = uneven["attrition"]["by_group_and_country"]
        self.assertEqual(set(table), set(rc.GROUPS))
        self.assertEqual(sum(v["analyzed"] for v in table["ai_saved"].values()), rc.PARTICIPANTS_PER_GROUP)

    def test_secondary_outputs(self):
        results = an.analyze(sim.simulate_dataset("work_account", 53))
        sec = results["secondary"]
        for g in rc.GROUPS:
            total = sum(sec["uses_of_time_minutes"][use]["by_group"][g]["mean"] for use in rc.USES_OF_TIME)
            self.assertAlmostEqual(total, rc.FREE_MINUTES)
            share = sec["leaving_early"]["by_group"][g]
            self.assertLessEqual(share["lower"], share["share"])
            self.assertLessEqual(share["share"], share["upper"])
        self.assertEqual(set(sec["leaving_early"]["contrasts"]), set(rc.HYPOTHESES))
        self.assertIn("h2_with_fun_share_covariate", sec)
        stayed = sec["h2_among_participants_who_stayed"]
        self.assertEqual(stayed["label"], "a comparison of groups selected after assignment")
        self.assertEqual(stayed["n"], int((~an.apply_exclusions(sim.simulate_dataset("work_account", 53))[0]
                                           ["left_early"]).sum()))
        self.assertEqual(set(sec["weekly_ai_users"]), set(rc.HYPOTHESES))
        self.assertEqual(set(sec["in_paid_employment"]["tests"]), set(rc.HYPOTHESES))
        self.assertEqual(set(sec["by_country"]), set(rc.HYPOTHESES))
        five = sec["among_participants_who_made_all_five_picks"]
        self.assertEqual(set(five["tests"]), {"H1", "H3", "H4", "H5"})
        self.assertEqual(five["n"] + five["dropped"], len(rc.GROUPS) * rc.PARTICIPANTS_PER_GROUP)
        self.assertGreater(five["dropped"], 0)
        self.assertEqual(len(sec["choosing_not_finished_at_c_plus_4"]), 2)
        watched = sec["fun_share_tests_among_participants_who_watched_a_video"]
        self.assertEqual(set(watched), {"H1", "H3", "H4", "H5"})
        self.assertLess(watched["H1"]["n"], results["hypotheses"]["H1"]["n"])
        position = sec["position_between_breaks"]
        if position["reported"]:
            self.assertLessEqual(position["lower"], position["upper"])
        else:
            self.assertIn("not significant", position["reason"])
        descriptives = sec["descriptives"]
        self.assertEqual(set(descriptives["flagged_possibly_automated"]), set(rc.GROUPS))
        self.assertGreater(len(descriptives["heard_of_study_before"]["by_week"]), 1)
        self.assertIn(results["secondary"]["without_hidden_page"]["pair_h3_h5"], rc.PAIR_ORDER)
        floors = sec["planned_group_floors_by_country"]
        self.assertEqual(set(floors), set(rc.COUNTRIES))
        self.assertEqual(sum(v["n"] for v in floors.values()), rc.PARTICIPANTS_PER_GROUP)
        mean_share = sum(v["share_of_free_minutes_to_paid_work"] * v["n"] for v in floors.values()) / rc.PARTICIPANTS_PER_GROUP
        self.assertAlmostEqual(mean_share, results["checks"]["paid_work_floor"]["planned_break_share_of_free_minutes"])
        self.assertGreater(results["exclusions"]["no_pick_made"], 0)
        self.assertLess(results["hypotheses"]["H1"]["n"], results["hypotheses"]["H2"]["n"])  # no pick, no fun share
        tables = results["checks"]
        self.assertEqual(set(tables["attribution"]["table_by_group"]["found_break"]), set(rc.ATTRIBUTION_ANSWERS))
        self.assertEqual(set(tables["label"]["recognition_table_by_group"]["planned_break"]),
                         set(rc.RECOGNITION_SENTENCES))

    def test_robustness_check_never_drops_leavers(self):
        raw = sim.simulate_dataset("no_effects", 54)
        data, _ = an.apply_exclusions(raw)
        dropped = an.hidden_page_dropped(data)
        self.assertFalse(dropped[data["left_early"]].any())
        self.assertFalse(dropped[data["video_minutes"] == 0].any())
        results = an.analyze(raw)
        self.assertEqual(results["secondary"]["without_hidden_page"]["dropped"], int(dropped.sum()))
        self.assertGreater(int(dropped.sum()), 0)


class OutputsAndHelpers(unittest.TestCase):
    def test_random_effects_with_identical_countries(self):
        r = an.random_effects([2.0] * 12, [1.0] * 12)
        self.assertAlmostEqual(r["summary"], 2.0)
        self.assertAlmostEqual(r["tau"], 0.0)
        self.assertAlmostEqual(r["se"], 1 / np.sqrt(12))

    def test_random_effects_against_hand_calculation(self):
        # three studies, estimates 0, 1, 5 with SE 1: Q = 14, C = 2, tau^2 = (14 - 2) / 2 = 6
        r = an.random_effects([0.0, 1.0, 5.0], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(r["tau"] ** 2, 6.0)
        self.assertAlmostEqual(r["summary"], 2.0)
        self.assertAlmostEqual(r["se"], np.sqrt(7 / 3))

    def test_csv_round_trip_and_outputs(self):
        data = sim.simulate_dataset("work_account", 41)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "simulated_work_account.csv"
            data.to_csv(path, index=False)
            results = an.analyze(an.read_export(path))
            json_path, text_path, text = an.write_outputs(results, Path(folder) / "results_work_account", path)
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(set(saved["hypotheses"]), set(rc.HYPOTHESES))
            self.assertIn("Reading:", text_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["exclusions"]["analyzed"], len(rc.GROUPS) * rc.PARTICIPANTS_PER_GROUP)
            self.assertEqual(set(saved["secondary"]["by_country"]["H5"]["random_effects"]),
                             {"summary", "se", "lower", "upper", "tau", "i_squared", "prediction_lower",
                              "prediction_upper", "countries"})
            for t in saved["hypotheses"].values():
                self.assertIn(t["reading"], rc.READINGS)
                self.assertEqual(t["difference_interval_level"], 0.99)
                self.assertEqual(t["equivalence_interval_level"], 0.98)


if __name__ == "__main__":
    unittest.main()
