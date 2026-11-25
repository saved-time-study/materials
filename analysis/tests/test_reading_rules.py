"""Fast tests of the reading rules, the outcome-neutral criteria, the criterion gates, the attrition threshold, the Lee
bounds and the robustness rule, on constructed values.

Each boundary case sits between the registered threshold and a nearby wrong one, so that changing a threshold or a
sign in analyze.py or registered_constants.py makes at least one test fail (tests/mutation_check.py runs that check).
Run: python -m unittest discover -s analysis/tests -v
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze as an  # noqa: E402
import registered_constants as rc  # noqa: E402
import simulate as sim  # noqa: E402

DF = 6824 - 15  # residual degrees of freedom of the registered design: 3 group terms, 11 country terms, intercept
T_EQUIVALENCE = stats.t.ppf(1 - 0.01, DF)  # the one-sided critical value at the registered alpha


def reading(h, estimate, se, bound=4.35):
    return str(an.read_hypothesis(h, np.array([estimate]), np.array([se]), DF, np.array([bound]))[0][0])


def result(estimate, se, bound=4.35, equivalence_first=False):
    return str(an.test_result(np.array([estimate]), np.array([se]), DF, np.array([bound]), equivalence_first)[0])


def estimate_for_two_sided_p(p, se=1.0):
    return stats.t.isf(p / 2, DF) * se


class RegisteredNumbers(unittest.TestCase):
    def test_alpha_is_05_split_five_ways(self):
        self.assertEqual(rc.NUMBER_OF_TESTS, 5)
        self.assertAlmostEqual(rc.ALPHA, 0.01)

    def test_interval_levels(self):
        self.assertAlmostEqual(rc.DIFFERENCE_INTERVAL_LEVEL, 0.99)
        self.assertAlmostEqual(rc.EQUIVALENCE_INTERVAL_LEVEL, 0.98)

    def test_bounds_in_points_and_minutes(self):
        self.assertAlmostEqual(rc.FUN_SHARE_BOUND, 4.35)
        self.assertAlmostEqual(rc.PAID_WORK_MINUTES_BOUND, 0.45)
        self.assertEqual(rc.BOUND_FORM, "observed_sd")

    def test_sample_is_1706_a_group_in_12_countries(self):
        self.assertEqual(len(rc.COUNTRIES), 12)
        self.assertEqual(sum(rc.PARTICIPANTS_PER_GROUP_BY_COUNTRY.values()), rc.PARTICIPANTS_PER_GROUP)
        self.assertEqual(rc.PARTICIPANTS_PER_GROUP, 1706)

    def test_the_five_contrasts(self):
        pairs = {h: (s["outcome"], s["minuend"], s["subtrahend"], s["kind"]) for h, s in rc.HYPOTHESES.items()}
        self.assertEqual(pairs, {
            "H1": ("fun_share", "found_break", "planned_break", "difference"),
            "H2": ("paid_work_minutes", "ai_saved", "found_break", "difference"),
            "H3": ("fun_share", "ai_saved", "planned_break", "equivalence"),
            "H4": ("fun_share", "ai_saved_labeled", "ai_saved", "difference"),
            "H5": ("fun_share", "ai_saved", "found_break", "difference"),
        })

    def test_every_reading_is_registered_and_has_an_account(self):
        for h, spec in rc.HYPOTHESES.items():
            self.assertEqual(set(spec["readings"]), set(rc.RESULTS), h)
            self.assertTrue(set(spec["readings"].values()) <= set(rc.READINGS), h)
            for r in list(spec["readings"].values()) + [rc.NOT_INTERPRETABLE]:
                self.assertIn(r, spec["accounts"], (h, r))

    def test_five_picks_robustness_covers_every_test_of_the_share_of_fun(self):
        self.assertEqual(rc.FIVE_PICKS_ROBUSTNESS_HYPOTHESES, ("H1", "H3", "H4", "H5"))
        self.assertEqual(rc.FIVE_PICKS_ROBUSTNESS_HYPOTHESES,
                         tuple(h for h, s in rc.HYPOTHESES.items() if s["outcome"] == "fun_share"))

    def test_significant_and_equivalent_cannot_both_happen_at_this_sample(self):
        se = np.sqrt(2 / rc.PARTICIPANTS_PER_GROUP)  # in units of d
        smallest_significant = stats.norm.isf(rc.ALPHA / 2) * se
        largest_equivalent = rc.SMALLEST_EFFECT_D - stats.norm.isf(rc.ALPHA) * se
        self.assertAlmostEqual(smallest_significant, 0.0882, places=4)
        self.assertAlmostEqual(largest_equivalent, 0.0703, places=4)
        self.assertGreater(smallest_significant, largest_equivalent)

    def test_modules_come_from_the_folder_under_test(self):
        expected = os.environ.get("MUTATION_CHECK_DIR")
        if expected:
            for module in (an, rc, sim):
                self.assertEqual(Path(module.__file__).resolve().parent, Path(expected).resolve())


class TestResults(unittest.TestCase):
    """The result of one test: higher, lower, equivalent or unresolved."""

    def test_large_differences_are_higher_or_lower(self):
        self.assertEqual(result(10.0, 1.0), an.HIGHER)
        self.assertEqual(result(-10.0, 1.0), an.LOWER)

    def test_p_of_012_is_not_significant_at_registered_alpha(self):
        # significant at the old .0125 but not at .01; its 98% interval also crosses the bound
        self.assertEqual(result(estimate_for_two_sided_p(0.012), 1.0), an.UNRESOLVED)

    def test_p_of_009_is_significant(self):
        self.assertEqual(result(estimate_for_two_sided_p(0.009), 1.0), an.HIGHER)
        self.assertEqual(result(-estimate_for_two_sided_p(0.009), 1.0), an.LOWER)

    def test_two_sided_p_of_018_is_not_significant(self):
        # the one-sided p is .009, below .01: only a one-sided test would call it significant
        self.assertEqual(result(estimate_for_two_sided_p(0.018), 1.0), an.UNRESOLVED)

    def test_zero_difference_with_small_error_is_equivalent(self):
        self.assertEqual(result(0.0, 1.0), an.EQUIVALENT)

    def test_equivalence_p_of_012_is_not_equivalent(self):
        # both one-sided tests at .0125 would pass; at the registered .01 the upper one does not
        self.assertEqual(result(4.35 - stats.t.isf(0.012, DF), 1.0), an.UNRESOLVED)

    def test_equivalence_p_of_009_is_equivalent(self):
        self.assertEqual(result(4.35 - stats.t.isf(0.009, DF), 1.0), an.EQUIVALENT)
        self.assertEqual(result(-4.35 + stats.t.isf(0.009, DF), 1.0), an.EQUIVALENT)

    def test_interval_reaching_past_either_bound_is_not_equivalent(self):
        # the 98% interval runs to 4.5 points (or -4.5): past 4.35, inside the 5.8 points that d = 0.20 would give
        se = 3.5 / T_EQUIVALENCE
        self.assertEqual(result(1.0, se), an.UNRESOLVED)
        self.assertEqual(result(-1.0, se), an.UNRESOLVED)

    def test_interval_just_inside_bounds_is_equivalent(self):
        se = 3.2 / T_EQUIVALENCE  # 98% interval from -2.2 to 4.2
        self.assertEqual(result(1.0, se), an.EQUIVALENT)
        self.assertEqual(result(-1.0, se), an.EQUIVALENT)

    def test_order_when_significant_and_inside_bounds(self):
        # not possible at the registered sample, but the code handles it: a difference hypothesis reads the
        # difference first, H3 reads equivalence first
        self.assertEqual(result(3.0, 0.5), an.HIGHER)
        self.assertEqual(result(3.0, 0.5, equivalence_first=True), an.EQUIVALENT)

    def test_vectorized_over_many_tests(self):
        results = an.test_result(np.array([10.0, -10.0, 0.0, 2.3]), np.ones(4), DF, np.full(4, 4.35))
        self.assertEqual(list(results), [an.HIGHER, an.LOWER, an.EQUIVALENT, an.UNRESOLVED])


class HypothesisReadings(unittest.TestCase):
    def test_h1(self):
        self.assertEqual([reading("H1", e, 1.0) for e in (10.0, -10.0, 0.0, 2.3)],
                         [an.SUPPORTED, an.OPPOSITE, an.EQUIVALENT, an.UNRESOLVED])

    def test_h2_names_the_goal_progress_account_when_lower(self):
        self.assertEqual([reading("H2", e, 1.0) for e in (10.0, -10.0, 0.0)], [an.SUPPORTED, an.LOWER, an.EQUIVALENT])
        self.assertIn("goal-progress", rc.HYPOTHESES["H2"]["accounts"][an.LOWER])
        self.assertIn("work account", rc.HYPOTHESES["H2"]["accounts"][an.SUPPORTED])

    def test_h3(self):
        self.assertEqual([reading("H3", e, 1.0) for e in (0.0, 10.0, -10.0, 2.3)],
                         [an.SUPPORTED, an.HIGHER_THAN_PLANNED, an.LOWER_THAN_PLANNED, an.UNRESOLVED])
        self.assertEqual(reading("H3", 3.0, 0.5), an.SUPPORTED)  # equivalence first

    def test_h4_is_two_sided_with_two_named_predictions(self):
        self.assertEqual([reading("H4", e, 1.0) for e in (10.0, -10.0, 0.0, 2.3)],
                         [an.HIGHER, an.LOWER, an.EQUIVALENT, an.UNRESOLVED])
        self.assertIn("gain", rc.HYPOTHESES["H4"]["accounts"][an.HIGHER])
        self.assertIn("reminds", rc.HYPOTHESES["H4"]["accounts"][an.LOWER])

    def test_h5_predicts_ai_saved_below_found(self):
        self.assertEqual([reading("H5", e, 1.0) for e in (-10.0, 10.0, 0.0, -2.3)],
                         [an.SUPPORTED, an.OPPOSITE, an.EQUIVALENT, an.UNRESOLVED])
        self.assertIn("windfall", rc.HYPOTHESES["H5"]["accounts"][an.EQUIVALENT])
        self.assertEqual(reading("H5", -3.0, 0.5), an.SUPPORTED)  # difference first

    def test_more_fun_than_found_time_is_the_goal_progress_prediction(self):
        # a higher share than the found-break group's is the goal-progress prediction, not "against both accounts"
        for text in (rc.HYPOTHESES["H5"]["accounts"][an.OPPOSITE], rc.PAIR_ACCOUNTS[rc.PAIR_MORE_THAN_FOUND]):
            self.assertIn("as the goal-progress account predicts", text)
            self.assertNotIn("both accounts", text)
            self.assertNotIn("no account", text)


class PairReadings(unittest.TestCase):
    """H3 and H5 read as a pair: all 125 combinations of H1, H3 and H5, written out by hand from the rules of
    Supplementary Table S4 in their registered order, with the rule that applies."""
    H1 = ("supported", "equivalent", "opposite", "unresolved", "not interpretable")
    H3 = ("supported", "higher than planned", "lower than planned", "unresolved", "not interpretable")
    H5 = ("supported", "opposite", "equivalent", "unresolved", "not interpretable")
    TABLE = {  # rows H3 in the order above; columns H5 in the order above; H1 supported
        "supported": ("spent like planned time", "more fun than found time", "cannot be told apart from either break",
                      "unresolved", "not interpretable"),
        "higher than planned": ("between the two breaks", "more fun than found time", "spent like found time",
                                "unresolved", "not interpretable"),
        "lower than planned": ("less fun than a planned break",) * 4 + ("not interpretable",),
        "unresolved": ("unresolved", "more fun than found time", "unresolved", "unresolved", "not interpretable"),
        "not interpretable": ("not interpretable",) * 5,
    }
    RULE = {  # the rule of Table S4 that gives each reading above, H1 supported
        "supported": (5, 3, 8, 4, 1),
        "higher than planned": (7, 3, 6, 4, 1),
        "lower than planned": (2, 2, 2, 2, 1),
        "unresolved": (4, 3, 4, 4, 1),
        "not interpretable": (1, 1, 1, 1, 1),
    }
    POSITIONS = {"spent like planned time", "spent like found time", "between the two breaks"}

    def expected(self, h1, h3, h5):
        k = self.H5.index(h5)
        reading, rule = self.TABLE[h3][k], self.RULE[h3][k]
        if h1 != "supported" and reading in self.POSITIONS:
            return "restricted", 9
        return reading, rule

    def test_every_combination_when_h1_is_supported(self):
        for h3 in self.H3:
            for h5, expected in zip(self.H5, self.TABLE[h3]):
                self.assertEqual(str(an.pair_reading("supported", h3, h5)), expected, (h3, h5))

    def test_the_three_positions_need_h1_supported(self):
        for h1 in ("equivalent", "opposite", "unresolved"):
            for h3 in self.H3:
                for h5, expected in zip(self.H5, self.TABLE[h3]):
                    expected = "restricted" if expected in self.POSITIONS else expected
                    self.assertEqual(str(an.pair_reading(h1, h3, h5)), expected, (h1, h3, h5))

    def test_every_combination_of_h1_h3_and_h5_with_its_rule(self):
        for h1 in self.H1:
            for h3 in self.H3:
                for h5 in self.H5:
                    reading, rule = self.expected(h1, h3, h5)
                    self.assertEqual(str(an.pair_reading(h1, h3, h5)), reading, (h1, h3, h5))
                    self.assertEqual(int(an.pair_reading(h1, h3, h5, rule_numbers=True)), rule, (h1, h3, h5))

    def test_the_rules_are_numbered_in_the_order_they_are_tried(self):
        self.assertEqual([r[0] for r in rc.PAIR_RULES], list(range(1, 10)))
        self.assertEqual({r[-1] for r in rc.PAIR_RULES}, set(rc.PAIR_ORDER))
        h1_conditions = {r[0]: r[1] for r in rc.PAIR_RULES}
        self.assertEqual({k for k, v in h1_conditions.items() if v == "supported"}, {5, 6, 7})
        self.assertEqual({k for k, v in h1_conditions.items() if v == "not supported"}, {9})

    def test_less_fun_than_planned_and_more_fun_than_found_need_neither_h1_nor_the_other_test(self):
        for h1 in self.H1:
            for h5 in self.H5[:-1]:  # any H5 that is interpretable
                self.assertEqual(str(an.pair_reading(h1, "lower than planned", h5)), rc.PAIR_LESS_THAN_PLANNED)
            for h3 in ("supported", "higher than planned", "unresolved"):
                self.assertEqual(str(an.pair_reading(h1, h3, "opposite")), rc.PAIR_MORE_THAN_FOUND)
            # unless the other test is not interpretable (rule 1 comes first)
            self.assertEqual(str(an.pair_reading(h1, "lower than planned", "not interpretable")), an.NOT_INTERPRETABLE)
            self.assertEqual(str(an.pair_reading(h1, "not interpretable", "opposite")), an.NOT_INTERPRETABLE)

    def test_vectorized(self):
        h1 = np.array(["supported", "supported", "supported", "equivalent"])
        h3 = np.array(["supported", "higher than planned", "higher than planned", "higher than planned"])
        h5 = np.array(["supported", "equivalent", "supported", "supported"])
        self.assertEqual(list(an.pair_reading(h1, h3, h5)), ["spent like planned time", "spent like found time",
                                                            "between the two breaks", "restricted"])

    def test_every_pair_reading_has_an_account(self):
        self.assertEqual(set(rc.PAIR_ACCOUNTS), set(rc.PAIR_ORDER))


class PurchasingPowerTable(unittest.TestCase):
    REFERENCE = Path(__file__).resolve().parents[1] / "reference"
    ISO = {"United States": "USA", "United Kingdom": "GBR", "Canada": "CAN", "Australia": "AUS",
           "South Africa": "ZAF", "Mexico": "MEX", "Chile": "CHL", "Germany": "DEU", "Poland": "POL",
           "Portugal": "PRT", "Italy": "ITA", "Spain": "ESP"}

    def test_table_matches_the_world_bank_files(self):
        import json
        values = {}
        for index, indicator in enumerate(("PA.NUS.PPP", "PA.NUS.FCRF")):
            rows = json.loads((self.REFERENCE / f"world_bank_{indicator}_2023-2025.json").read_text(encoding="utf-8"))[1]
            for r in rows:
                if r["date"] == str(rc.PPP_YEAR):
                    values.setdefault(r["countryiso3code"], [None, None])[index] = r["value"]
        self.assertEqual(set(rc.PPP_TABLE), set(rc.COUNTRIES))
        for country, (ppp, rate) in rc.PPP_TABLE.items():
            self.assertAlmostEqual(ppp, values[self.ISO[country]][0], places=6, msg=country)
            self.assertAlmostEqual(rate, values[self.ISO[country]][1], places=6, msg=country)

    def test_value_of_one_item(self):
        self.assertAlmostEqual(rc.ppp_value_of_item("United States")["international_dollars"], 0.10)
        self.assertAlmostEqual(rc.ppp_value_of_item("South Africa")["international_dollars"],
                               0.10 * 17.8887489855833 / 7.42029658394341)
        self.assertAlmostEqual(rc.ppp_value_of_item("Mexico")["local_currency_at_exchange_rate"], 1.92375083333333)


class EquivalenceBounds(unittest.TestCase):
    def test_fixed_bounds(self):
        y = np.zeros((10, 2))
        self.assertTrue(np.allclose(an.equivalence_bound("fun_share", y, y, "fixed"), 4.35))
        self.assertTrue(np.allclose(an.equivalence_bound("paid_work_minutes", y, y, "fixed"), 0.45))

    def test_observed_sd_bound_pools_the_two_groups_raw_values(self):
        a = np.array([[0.0], [2.0], [4.0], [6.0]])
        b = np.array([[1.0], [3.0], [5.0], [9.0]])
        pooled = np.sqrt((3 * a.var(ddof=1) + 3 * b.var(ddof=1)) / 6)
        self.assertAlmostEqual(float(an.equivalence_bound("fun_share", a, b, "observed_sd")[0]), 0.15 * pooled)


def small_design(per_cell=50):
    """4 groups x 2 countries x per_cell participants: group and country index of each."""
    groups, countries = [], []
    for c in range(2):
        for g in range(4):
            groups += [g] * per_cell
            countries += [c] * per_cell
    return np.array(groups), np.array(countries)


class OutcomeNeutralCriteria(unittest.TestCase):
    def setUp(self):
        self.g, self.c = small_design()
        self.model = an.CountryFixedEffectsOLS(self.g, self.c)
        n = len(self.g)
        self.rank = np.zeros(n, dtype=int)  # position of each participant within the group, 0 to 99
        for k in range(4):
            self.rank[self.g == k] = np.arange(100)
        self.jitter = (self.rank % 5) * 0.1
        self.inputs = {
            "fun_share": np.full(n, 50.0),
            "paid_work_minutes": np.full(n, 2.0),
            "surprise": np.where(self.g == 0, 2.0, 5.0) + self.jitter,
            "says_ai": self.share_saying({0: 0, 1: 5, 2: 90, 3: 90}),
            "says_not_reviewed": self.share_saying({0: 5, 1: 80, 2: 5, 3: 5}),
            "saw_label": self.share_saying({0: 5, 1: 8, 2: 20, 3: 85}),
        }

    def share_saying(self, per_hundred, g=None, rank=None):
        """1 for the first k participants of each group of 100, where k is given for each group."""
        g = self.g if g is None else g
        rank = self.rank if rank is None else rank
        return np.array([1.0 if r < per_hundred[k] else 0.0 for k, r in zip(g, rank)])

    def run_checks(self, model=None, **changes):
        inputs = {**self.inputs, **changes}
        return an.outcome_neutral_checks(model or self.model, inputs["fun_share"], inputs["paid_work_minutes"],
                                         inputs["surprise"], inputs["says_ai"], inputs["says_not_reviewed"],
                                         inputs["saw_label"])

    def passed(self, check, **changes):
        return bool(self.run_checks(**changes)["passed"][check][0])

    def failing(self, **changes):
        return {c for c, v in self.run_checks(**changes)["passed"].items() if not bool(v[0])}

    def test_all_criteria_pass_on_the_base_case(self):
        self.assertEqual(self.failing(), set())
        self.assertEqual(set(self.run_checks()["passed"]), set(rc.CHECK_GATES))

    # (a) surprise
    def test_surprise_needs_the_interval_above_015_pooled_sd_not_only_above_zero(self):
        spread = np.where(self.rank % 2 == 0, -1.0, 1.0) * (1 + (self.rank % 5) * 0.2)  # SD about 1.4
        sd = np.sqrt(((spread[self.g == 0] - spread[self.g == 0].mean()) ** 2).sum() / 99)
        surprise = np.where(self.g == 0, 2.0 + spread, 5.0 + spread)
        surprise = np.where(self.g == 2, 2.0 + spread + 0.35 * sd, surprise)  # AI-saved 0.35 SD above planned
        checks = self.run_checks(surprise=surprise)
        ai = checks["surprise"]["ai_saved"]
        self.assertGreater(float(ai["lower"][0]), 0.0)  # its 95% interval is above zero
        self.assertLess(float(ai["lower"][0]), float(ai["minimum"][0]))  # but not above 0.15 pooled SD
        self.assertEqual({c for c, v in checks["passed"].items() if not bool(v[0])}, {"surprise_ai_saved"})

    def test_surprise_failure_is_reported_for_that_group_only(self):
        for k, check in ((1, "surprise_found_break"), (2, "surprise_ai_saved"), (3, "surprise_ai_saved_labeled")):
            surprise = np.where(np.isin(self.g, [0, k]), 2.0, 5.0) + self.jitter  # group k rated like planned
            self.assertEqual(self.failing(surprise=surprise), {check}, check)

    def test_ai_saved_less_surprised_than_found(self):
        flag = self.run_checks()["surprise_ai_saved_minus_found"]["ai_saved_less_surprised"]
        self.assertFalse(bool(flag[0]))  # equal surprise: the interval includes zero
        surprise = np.where(self.g == 0, 2.0, np.where(self.g == 2, 4.0, 5.0)) + self.jitter
        checks = self.run_checks(surprise=surprise)
        self.assertTrue(bool(checks["surprise_ai_saved_minus_found"]["ai_saved_less_surprised"][0]))
        self.assertEqual({c for c, v in checks["passed"].items() if not bool(v[0])}, set())  # not a criterion
        surprise = np.where(self.g == 0, 2.0, np.where(self.g == 2, 6.0, 5.0)) + self.jitter
        self.assertFalse(bool(self.run_checks(surprise=surprise)["surprise_ai_saved_minus_found"]
                              ["ai_saved_less_surprised"][0]))

    def test_less_surprised_needs_the_interval_below_minus_015_pooled_sd(self):
        spread = np.where(self.rank % 2 == 0, -1.0, 1.0) * (1 + (self.rank % 5) * 0.2)  # SD about 1.4
        sd = np.sqrt(((spread[self.g == 1] - spread[self.g == 1].mean()) ** 2).sum() / 99)
        for shift, flagged in ((-0.35, False), (-0.60, True)):  # AI-saved this many SD below the found break
            surprise = np.where(self.g == 0, 1.0 + spread, 5.0 + spread)
            surprise = np.where(self.g == 2, 5.0 + spread + shift * sd, surprise)
            r = self.run_checks(surprise=surprise)["surprise_ai_saved_minus_found"]
            if not flagged:
                self.assertLess(float(r["upper"][0]), 0.0)  # below zero, but not below minus 0.15 SD
            self.assertEqual(bool(r["ai_saved_less_surprised"][0]), flagged, shift)

    # (b) attribution
    def test_attribution_ai_saved_part_needs_80_percent_in_each_ai_group(self):
        self.assertTrue(self.passed("attribution_ai_saved", says_ai=self.share_saying({0: 0, 1: 5, 2: 80, 3: 80})))
        self.assertFalse(self.passed("attribution_ai_saved", says_ai=self.share_saying({0: 0, 1: 5, 2: 79, 3: 90})))
        self.assertEqual(self.failing(says_ai=self.share_saying({0: 0, 1: 5, 2: 90, 3: 75})), {"attribution_ai_saved"})

    def test_attribution_found_part_allows_at_most_20_percent_choosing_the_assistant(self):
        self.assertTrue(self.passed("attribution_found_break", says_ai=self.share_saying({0: 0, 1: 20, 2: 90, 3: 90})))
        self.assertEqual(self.failing(says_ai=self.share_saying({0: 0, 1: 25, 2: 90, 3: 90})),
                         {"attribution_found_break"})

    def test_attribution_found_part_needs_half_choosing_would_not_be_reviewed(self):
        self.assertTrue(self.passed("attribution_found_break",
                                    says_not_reviewed=self.share_saying({0: 5, 1: 50, 2: 5, 3: 5})))
        self.assertEqual(self.failing(says_not_reviewed=self.share_saying({0: 5, 1: 45, 2: 5, 3: 5})),
                         {"attribution_found_break"})

    # (c) the label
    def test_label_needs_20_points_more(self):
        self.assertTrue(self.passed("label", saw_label=self.share_saying({0: 5, 1: 8, 2: 20, 3: 40})))  # 20 exactly
        self.assertEqual(self.failing(saw_label=self.share_saying({0: 5, 1: 8, 2: 20, 3: 39})), {"label"})  # 19
        self.assertFalse(self.passed("label", saw_label=self.share_saying({0: 5, 1: 8, 2: 85, 3: 20})))

    def test_label_needs_its_interval_above_zero(self):
        g, c = small_design(per_cell=5)  # 10 a group: a difference of 20 points with an interval reaching below zero
        model = an.CountryFixedEffectsOLS(g, c)
        rank = np.zeros(len(g), dtype=int)
        for k in range(4):
            rank[g == k] = np.arange(10)
        n = len(g)
        saw = self.share_saying({0: 0, 1: 0, 2: 0, 3: 2}, g, rank)
        checks = an.outcome_neutral_checks(model, np.full(n, 50.0), np.full(n, 2.0),
                                           np.where(g == 0, 2.0, 5.0) + (rank % 5) * 0.1,
                                           self.share_saying({0: 0, 1: 0, 2: 9, 3: 9}, g, rank),
                                           self.share_saying({0: 0, 1: 8, 2: 0, 3: 0}, g, rank), saw)
        self.assertGreaterEqual(round(float(checks["label"]["difference"][0]), an.ROUND), 0.2)
        self.assertLessEqual(float(checks["label"]["lower"][0]), 0.0)
        self.assertFalse(bool(checks["passed"]["label"][0]))

    # (d) floor and ceiling, in the planned group
    def test_fun_share_floor_and_ceiling(self):
        planned = self.g == 0
        for mean, expected in ((12.0, False), (15.0, True), (85.0, True), (88.0, False)):
            fun = np.where(planned, mean, 50.0)
            self.assertEqual(self.passed("fun_share_floor_ceiling", fun_share=fun), expected, msg=f"mean {mean}")
        self.assertTrue(self.passed("fun_share_floor_ceiling", fun_share=np.where(planned, 50.0, 100.0)))

    def test_paid_work_floor_is_measured_in_the_planned_group(self):
        planned = self.g == 0
        self.assertTrue(self.passed("paid_work_floor", paid_work_minutes=np.where(planned, 0.5, 0.0)))  # 5 percent
        self.assertFalse(self.passed("paid_work_floor", paid_work_minutes=np.where(planned, 0.4, 10.0)))  # 4 percent


class CriterionGates(unittest.TestCase):
    def gate(self, failed_check):
        readings = {h: np.array([an.SUPPORTED, an.SUPPORTED]) for h in rc.HYPOTHESES}
        passed = {c: np.array([True, True]) for c in rc.CHECK_GATES}
        if failed_check:
            passed[failed_check] = np.array([True, False])  # fails in the second data set only
        gated = an.apply_check_gates(readings, passed)
        self.assertTrue(all(gated[h][0] == an.SUPPORTED for h in rc.HYPOTHESES))
        self.assertTrue(all(gated[h][1] in (an.SUPPORTED, an.NOT_INTERPRETABLE) for h in rc.HYPOTHESES))
        return {h for h in rc.HYPOTHESES if gated[h][1] == an.NOT_INTERPRETABLE}

    def test_no_failed_criterion_changes_nothing(self):
        self.assertEqual(self.gate(None), set())

    def test_surprise_gates(self):
        self.assertEqual(self.gate("surprise_found_break"), {"H1", "H2", "H5"})
        self.assertEqual(self.gate("surprise_ai_saved"), {"H2", "H3", "H5"})
        self.assertEqual(self.gate("surprise_ai_saved_labeled"), {"H4"})

    def test_attribution_gates(self):
        self.assertEqual(self.gate("attribution_ai_saved"), {"H2", "H3", "H4", "H5"})
        self.assertEqual(self.gate("attribution_found_break"), {"H1", "H2", "H5"})

    def test_label_gate(self):
        self.assertEqual(self.gate("label"), {"H4"})

    def test_floor_gates(self):
        self.assertEqual(self.gate("fun_share_floor_ceiling"), {"H1", "H3", "H4", "H5"})
        self.assertEqual(self.gate("paid_work_floor"), {"H2"})

    def test_not_interpretable_is_not_unresolved(self):
        self.assertNotEqual(an.NOT_INTERPRETABLE, an.UNRESOLVED)
        self.assertIn(an.NOT_INTERPRETABLE, rc.READINGS)


def small_export(**assumptions):
    """A small simulated export: 3 analyzable participants a group in each of the 12 countries."""
    a = sim.Assumptions(**assumptions) if assumptions else None
    return sim.simulate_dataset("no_effects", 5, a, per_group_by_country={c: 3 for c in rc.COUNTRIES})


class ParticipantsWhoMadeNoPick(unittest.TestCase):
    """A participant who made no pick has no share of fun videos: left out of H1, H3, H4 and H5 only, kept in H2 and
    every other outcome, counted by group, and the empty share does not stop the analysis."""
    PLANTED = {"planned_break": 1, "found_break": 2, "ai_saved": 0, "ai_saved_labeled": 3}

    @classmethod
    def setUpClass(cls):
        raw = small_export()
        keep = (raw["passed_attention_check"].astype(bool) & raw["finished"].astype(bool)
                & ~raw["withdrew"].astype(bool))
        filled = raw[list(rc.PICK_FILLED_COLUMNS)].fillna(False).astype(bool)
        all_watch = (raw[an.SLOT_COLUMNS] == "watch").all(axis=1)
        for group, k in cls.PLANTED.items():  # all five picks filled at random: no pick made, every slot a video
            rows = raw.index[keep & ~filled.any(axis=1) & all_watch & (raw["group"] == group)][:k]
            raw.loc[rows, list(rc.PICK_FILLED_COLUMNS)] = True
            raw.loc[rows, "fun_share"] = np.nan
        no_pick = keep & raw[list(rc.PICK_FILLED_COLUMNS)].fillna(False).astype(bool).all(axis=1)
        cls.expected = {g: int((no_pick & (raw["group"] == g)).sum()) for g in rc.GROUPS}
        cls.raw = raw
        cls.data, cls.summary = an.apply_exclusions(raw)  # an empty share for someone who made no pick does not stop it
        cls.n_all = len(cls.data)
        cls._results = None

    @property
    def results(self):
        if type(self)._results is None:
            type(self)._results = an.analyze(self.raw)
        return type(self)._results

    def test_the_tests_leave_them_out_of_the_share_of_fun_only(self):
        tests = an.confirmatory_on(self.data)
        missing = sum(self.expected.values())
        for h, spec in rc.HYPOTHESES.items():
            self.assertEqual(tests[h]["n"], self.n_all - missing if spec["outcome"] == "fun_share" else self.n_all, h)

    def test_the_number_is_counted_by_group(self):
        self.assertEqual(self.summary["no_pick_made_by_group"], self.expected)
        ex = self.results["exclusions"]
        self.assertEqual(ex["no_pick_made_by_group"], self.expected)
        self.assertEqual(ex["no_pick_made"], sum(self.expected.values()))
        for g, k in self.PLANTED.items():
            self.assertGreaterEqual(self.expected[g], k)
        text = an.report_text(self.results, "small export")
        self.assertIn(", ".join(f"{g} {n}" for g, n in self.expected.items()), text)

    def test_left_out_of_the_four_tests_of_the_share_of_fun_only(self):
        missing = sum(self.expected.values())
        self.assertGreater(missing, 0)
        for h, t in self.results["hypotheses"].items():
            fun = rc.HYPOTHESES[h]["outcome"] == "fun_share"
            self.assertEqual(t["n"], self.n_all - missing if fun else self.n_all, h)
        self.assertEqual(self.results["hypotheses"]["H2"]["n"], self.n_all)

    def test_kept_in_every_other_outcome(self):
        sec = self.results["secondary"]
        for use in rc.USES_OF_TIME:
            self.assertEqual(sum(v["n"] for v in sec["uses_of_time_minutes"][use]["by_group"].values()), self.n_all)
        self.assertEqual(sum(v["n"] for v in sec["leaving_early"]["by_group"].values()), self.n_all)
        self.assertEqual(sec["labeled_minus_unlabeled_paid_work_minutes"]["n"], self.n_all)
        planned = self.data[self.data["group"] == "planned_break"]
        self.assertAlmostEqual(self.results["checks"]["paid_work_floor"]["planned_break_share_of_free_minutes"],
                               planned["paid_work_minutes"].sum() / (len(planned) * rc.FREE_MINUTES))
        # the one analysis that uses the share as a covariate has no value for them
        self.assertEqual(sec["h2_with_fun_share_covariate"]["n"], self.n_all - sum(self.expected.values()))


def attrition_raw(not_finished):
    """100 assigned a group in one country; not_finished gives, for each group, how many did not finish."""
    rows = []
    for g in rc.GROUPS:
        for i in range(100):
            rows.append({"country": "Chile", "passed_attention_check": True, "group": g,
                         "finished": i >= not_finished[g], "withdrew": False})
    raw = pd.DataFrame(rows)
    analyzed = raw[raw["finished"]].assign(left_early=False)
    return raw, analyzed


class AttritionAndLeeBounds(unittest.TestCase):
    def test_threshold_is_more_than_3_points(self):
        gaps = {4: True, 3: False}
        for extra, crossed in gaps.items():
            raw, analyzed = attrition_raw({"planned_break": 2, "found_break": 2, "ai_saved": 2 + extra,
                                           "ai_saved_labeled": 2})
            table = an.attrition_table(raw, analyzed)
            self.assertAlmostEqual(table["largest_gap_between_groups"], extra / 100)
            self.assertEqual(table["threshold_crossed"], crossed, extra)

    def test_lee_bounds_by_hand(self):
        a, b = np.arange(10.0), np.arange(10.0)
        bounds = an.lee_bounds(a, b, 1.0, 0.8)  # the minuend group kept more: trim 20 percent of it
        self.assertAlmostEqual(bounds["lower"], 3.5 - 4.5)
        self.assertAlmostEqual(bounds["upper"], 5.5 - 4.5)
        self.assertEqual(bounds["trimmed"], 2)
        bounds = an.lee_bounds(a, b, 0.8, 1.0)  # the subtrahend group kept more: trim it instead
        self.assertAlmostEqual(bounds["lower"], 4.5 - 5.5)
        self.assertAlmostEqual(bounds["upper"], 4.5 - 3.5)
        shifted = an.lee_bounds(a + 10, b, 1.0, 0.5)  # trim half of the minuend group
        self.assertAlmostEqual(shifted["lower"], 12.0 - 4.5)
        self.assertAlmostEqual(shifted["upper"], 17.0 - 4.5)


def shifted_frame(found_shift, ai_shift, per_cell=20):
    """Groups and countries with a share of fun that is the planned group's values plus a fixed shift."""
    base = np.tile(np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 40.0, 60.0, 20.0, 80.0]), per_cell // 10)
    shift = {"planned_break": 0.0, "found_break": found_shift, "ai_saved": ai_shift, "ai_saved_labeled": 0.0}
    rows = [{"group": g, "country": c, "fun_share": v + shift[g]}
            for c in ("Chile", "Spain") for g in rc.GROUPS for v in base]
    return pd.DataFrame(rows)


class PositionBetweenBreaks(unittest.TestCase):
    def test_ratio_and_bootstrap_interval(self):
        out = an.position_between_breaks(shifted_frame(20.0, 10.0), draws=200)
        self.assertTrue(out["reported"])
        self.assertAlmostEqual(out["ratio"], 0.5)
        self.assertLessEqual(out["lower"], 0.5)
        self.assertGreaterEqual(out["upper"], 0.5)

    def test_not_reported_when_found_minus_planned_is_not_significant(self):
        out = an.position_between_breaks(shifted_frame(0.0, 10.0), draws=200)
        self.assertFalse(out["reported"])
        self.assertNotIn("ratio", out)


class ExportAndRobustnessRule(unittest.TestCase):
    def setUp(self):
        self.raw = small_export()
        self.data, self.summary = an.apply_exclusions(self.raw)
        raw = self.raw
        self.keep = (raw["passed_attention_check"].astype(bool) & raw["finished"].astype(bool)
                     & ~raw["withdrew"].astype(bool))
        all_made = ~raw[list(rc.PICK_FILLED_COLUMNS)].fillna(False).astype(bool).any(axis=1)
        all_watch = (raw[an.SLOT_COLUMNS] == "watch").all(axis=1)
        # rows that made five picks, watch every slot and have a share of fun strictly between 0 and 100
        self.plain = raw.index[self.keep & all_made & all_watch & raw["fun_share"].between(1, 99)]

    def test_filled_picks_do_not_count_in_the_share_of_fun(self):
        raw = self.raw.copy()
        row = self.plain[0]
        raw.loc[row, "pick_1_filled"] = True
        made = [raw.loc[row, f"video_pick_{j + 1}"] for j in range(1, rc.VIDEO_PICKS)]
        with self.assertRaises(an.DataError):  # the exported share still counts the filled pick
            an.apply_exclusions(raw)
        raw.loc[row, "fun_share"] = 100.0 * sum(v in rc.FUN_VIDEOS for v in made) / len(made)
        data, summary = an.apply_exclusions(raw)
        self.assertEqual(summary["some_picks_filled_at_random"], self.summary["some_picks_filled_at_random"] + 1)
        self.assertFalse(bool(data.loc[data["participant_id"] == raw.loc[row, "participant_id"],
                                       "made_all_five_picks"].iloc[0]))

    def test_no_pick_means_no_share_of_fun_and_every_other_outcome_kept(self):
        raw = self.raw.copy()
        row = self.plain[0]
        raw.loc[row, list(rc.PICK_FILLED_COLUMNS)] = True
        with self.assertRaises(an.DataError):  # a share of fun for someone who made no pick
            an.apply_exclusions(raw)
        raw.loc[row, "fun_share"] = np.nan
        data, summary = an.apply_exclusions(raw)
        self.assertEqual(summary["analyzed"], self.summary["analyzed"])
        kept = data[data["participant_id"] == raw.loc[row, "participant_id"]].iloc[0]
        self.assertTrue(np.isnan(kept["fun_share"]))
        self.assertEqual(kept["paid_work_minutes"], 0.0)
        raw.loc[self.plain[1], "fun_share"] = np.nan  # an empty share for someone who made five picks stops it
        with self.assertRaises(an.DataError):
            an.apply_exclusions(raw)

    def test_a_participant_with_filled_picks_cannot_have_a_work_slot(self):
        raw = self.raw.copy()
        row = self.plain[0]
        raw.loc[row, "pick_1_filled"] = True
        made = [raw.loc[row, f"video_pick_{j + 1}"] for j in range(1, rc.VIDEO_PICKS)]
        raw.loc[row, "fun_share"] = 100.0 * sum(v in rc.FUN_VIDEOS for v in made) / len(made)
        raw.loc[row, ["slot_1_use", "paid_work_minutes"]] = ["paid_work", 2.0]
        with self.assertRaisesRegex(an.DataError, "never saw the slots"):
            an.apply_exclusions(raw)

    def test_hidden_time_has_no_value_without_video_minutes(self):
        raw = self.raw.copy()
        leaver = raw.index[self.keep & raw["left_early"].astype(bool)]
        self.assertTrue(raw.loc[leaver, "minutes_video_page_hidden"].isna().all())
        raw.loc[leaver[0], "minutes_video_page_hidden"] = 1.0
        with self.assertRaisesRegex(an.DataError, "must be empty"):
            an.apply_exclusions(raw)
        self.assertTrue(self.data.loc[self.data["video_minutes"] == 0, "hidden_share_of_video_minutes"].isna().all())

    def test_four_uses_add_up_to_10_and_leavers_give_no_minutes_to_work(self):
        uses = self.data[["fun_videos_watched", "self_improvement_videos_watched", "paid_work", "left_early_minutes"]]
        self.assertTrue(np.allclose(uses.sum(axis=1), 10))
        self.assertTrue((self.data.loc[self.data["left_early"], "paid_work_minutes"] == 0).all())

    def test_hidden_time_counts_only_video_minutes(self):
        data = self.data.copy()
        data["video_minutes"] = 4.0
        data["hidden_share_of_video_minutes"] = 0.0
        data.loc[0, "hidden_share_of_video_minutes"] = 2.0 / 4.0  # half: kept
        data.loc[1, "hidden_share_of_video_minutes"] = 2.1 / 4.0  # more than half: dropped
        data.loc[2, "hidden_share_of_video_minutes"] = 0.75  # dropped
        self.assertEqual(list(an.hidden_page_dropped(data).iloc[:4]), [False, True, True, False])

    def test_hidden_share_uses_the_video_minutes_as_denominator(self):
        raw = self.raw.copy()
        row = self.plain[0]
        for j in range(rc.VIDEO_PICKS):
            raw.loc[row, f"slot_{j + 1}_use"] = "paid_work" if j < 3 else "watch"  # 4 video minutes, 6 of work
        raw.loc[row, ["paid_work_minutes", "left_early", "minutes_video_page_hidden"]] = [6.0, False, 2.5]
        data, _ = an.apply_exclusions(raw)
        picked = data[data["participant_id"] == raw.loc[row, "participant_id"]].iloc[0]
        self.assertAlmostEqual(picked["hidden_share_of_video_minutes"], 2.5 / 4.0)
        self.assertTrue(bool(an.hidden_page_dropped(data[data["participant_id"] == picked["participant_id"]]).iloc[0]))

    def test_leaving_on_some_slots_only_stops_the_analysis(self):
        raw = self.raw.copy()
        raw.loc[self.plain[0], "slot_1_use"] = "left_early"
        with self.assertRaisesRegex(an.DataError, "replaces all five slots"):  # the guard that names the fault
            an.apply_exclusions(raw)

    def test_optional_descriptives_present_or_absent(self):
        out = an.optional_descriptives(self.data)
        self.assertEqual(set(out["heard_of_study_before"]["by_group"]["ai_saved"]) - {"n", "missing"},
                         set(rc.HEARD_BEFORE_ANSWERS))
        self.assertIsInstance(out["purpose_names_reason_or_source"]["by_week"], dict)
        self.assertEqual(set(out["seconds_to_fifth_pick"]), set(rc.GROUPS))
        bare = an.optional_descriptives(self.data.drop(columns=[rc.AUTOMATED_FLAG_COLUMN, rc.PURPOSE_CODED_COLUMN,
                                                                 "review_edits"]))
        self.assertEqual(bare["flagged_possibly_automated"], "not in the export")
        self.assertTrue(bare["purpose_names_reason_or_source"].startswith("not coded yet"))
        self.assertEqual(bare["review_edits"], "not in the export")

    def test_screen_text_is_read_as_the_export_code(self):
        raw = self.raw.copy()
        raw["how_review_ended"] = raw["how_review_ended"].map(rc.ATTRIBUTION_ANSWERS)
        raw["ai_use"] = raw["ai_use"].map(rc.AI_USE_ANSWERS)
        data, _ = an.apply_exclusions(raw)
        self.assertTrue(np.array_equal(data["says_ai_finished"], self.data["says_ai_finished"]))
        self.assertTrue(data["ai_use"].isin(rc.AI_USE_ANSWERS).all())


if __name__ == "__main__":
    unittest.main()
