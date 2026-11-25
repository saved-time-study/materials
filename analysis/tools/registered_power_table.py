"""Print the README's tables of the power analysis, from analysis/output/power_by_simulation.json.

Usage: python analysis/tools/registered_power_table.py [--form observed_sd]
Prints, in Markdown: (1) for each hypothesis and each powered pattern, the share of data sets with the expected
reading, its 95% Monte Carlo interval, Table 1's claim and the verdicts against 0.95 and 0.96; (2) the halfway
pattern's readings, and the pair reading of H3 and H5 under every pattern; (3) the power curves of the equivalence tests; (4) the
criteria: the share of data sets in which each failed, and the normal-theory values at which each fails.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
form = sys.argv[sys.argv.index("--form") + 1] if "--form" in sys.argv else "observed_sd"
sim = json.loads((ROOT / "output" / "power_by_simulation.json").read_text(encoding="utf-8"))
patterns = sim["patterns"]
POWERED = ["hypotheses_true_at_bound", "windfall_account", "work_account", "no_effects"]
HYPOTHESES = ["H1", "H2", "H3", "H4", "H5"]

print(f"Power for each hypothesis under each pattern ({form} bound, {patterns['no_effects']['draws']:,} data sets a "
      f"pattern)\n")
print("| Hypothesis | Pattern | Expected reading | Share [95% interval] | Table 1 claims | Against 0.95 | Against 0.96 |")
print("|---|---|---|---|---|---|---|")
shares = []
for h in HYPOTHESES:
    order = sorted(POWERED, key=lambda p: (patterns[p]["by_bound_form"][form][h]["expected_reading"] == "equivalent",
                                           POWERED.index(p)))
    for p in order:
        r = patterns[p]["by_bound_form"][form][h]
        shares.append(r["share_expected"])
        print(f"| {h} | {p} | {r['expected_reading']} | {r['share_expected']:.3f} [{r['interval'][0]:.3f}, "
              f"{r['interval'][1]:.3f}] | {r['claimed_in_table_1']:.3f} | {r['verdict_against_0_95']} | "
              f"{r['verdict_against_0_96']} |")
print(f"\n{len(shares)} rows; shares from {min(shares):.3f} to {max(shares):.3f}")

half = patterns["halfway"]
print(f"\nThe halfway pattern ({half['draws']:,} data sets; not powered)\n")
print("| Hypothesis | Expected reading | Share with it | All readings |")
print("|---|---|---|---|")
for h in HYPOTHESES:
    r = half["by_bound_form"][form][h]
    rest = ", ".join(f"{k} {v:.3f}" for k, v in r["share_by_reading"].items())
    print(f"| {h} | {r['expected_reading']} | {r['share_expected']:.3f} | {rest} |")
print("\nH3 and H5 read as a pair: the share of data sets with each registered pair reading\n")
order = ["spent like planned time", "spent like found time", "between the two breaks",
         "cannot be told apart from either break", "less fun than a planned break", "more fun than found time",
         "restricted", "unresolved", "not interpretable"]
print("| Pattern | Expected pair reading | Share [95% interval] | " + " | ".join(order) + " |")
print("|---|---|---|" + "---|" * len(order))
for p in POWERED + ["halfway"]:
    r = patterns[p]
    cells = " | ".join(f"{r['pair_reading_share'].get(k, 0.0):.3f}" for k in order)
    print(f"| {p} | {r['expected_pair']} | {r['expected_pair_share']:.3f} [{r['expected_pair_interval'][0]:.3f}, "
          f"{r['expected_pair_interval'][1]:.3f}] | {cells} |")

print("\nPower of each equivalence test against the true difference (both one-sided tests at .01)\n")
print("| True difference (d) | Normal theory | " + " | ".join(HYPOTHESES) + " |")
print("|---|---|" + "---|" * len(HYPOTHESES))
for row in sim["power_curves"]:
    cells = " | ".join(f"{row['simulated'][h]['share']:.3f}" for h in HYPOTHESES)
    print(f"| {row['true_d']:.2f} | {row['normal_theory']:.3f} | {cells} |")

worst = {c: max(patterns[p]["criterion_failure_share"][c] for p in patterns)
         for c in patterns["no_effects"]["criterion_failure_share"]}
print("\nShare of data sets in which each criterion failed, largest over the five patterns: "
      + ", ".join(f"{c} {v:.4f}" for c, v in worst.items())
      + f"; any criterion {max(patterns[p]['any_criterion_failed_share'] for p in patterns):.4f}.")
print("\nThe true value at which each criterion fails with probability 0.05 and 0.50 (normal theory, registered sample)\n")
print("| Criterion | Fails with probability 0.05 at | 0.50 at |")
print("|---|---|---|")
for name, v in sim["criterion_thresholds"].items():
    print(f"| {name} | {v['0.05']:.3f} | {v['0.5']:.3f} |")
less = [patterns[p]["ai_saved_less_surprised_than_found_share"] for p in patterns]
print(f"\nAI-saved group less surprised than the found-break group by more than 0.15 pooled SD (a limit on reading H3 "
      f"and H5, not a criterion): {min(less):.4f} to {max(less):.4f} of data sets, by pattern.")
