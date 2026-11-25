"""Sample size, power and participant pay behind Table 1 of the manuscript (the design table).

Run from the repository's root:  python power/design_table_power.py
Its output is kept beside it in power/design_table_power_output.txt.

The constants at the top of the script set the design: the smallest effect of interest in standard deviations,
the power that every test must reach, the number of confirmatory tests (alpha is 0.05 divided by that number,
a Bonferroni correction), the numbers of groups and countries, and the participant pay (the rate in USD a
minute, the platform's fee, the share of participants replaced, the mean pay for paid extra work, the cost of
translation and the budget). The script prints the sample a group that a two-sided test needs to detect the
smallest effect, the sample at which two one-sided tests against bounds of plus and minus that effect reach the
same power when the true difference is zero, the sample that the design uses (the larger of the two), the
power of each test at that sample, and the pay. Each printed line names what it gives.
"""
from math import ceil, sqrt

from scipy.stats import norm

D, POWER, TESTS, GROUPS, COUNTRIES = 0.15, 0.96, 5, 4, 12
ALPHA = 0.05 / TESTS
RATE, FEE, REPLACE, BONUS, TRANSLATION, BUDGET = 0.15, 1.33, 1.08, 0.40, 2500, 32000
SESSION_MINUTES = (15, 17)


def n_difference():
    z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)
    return ceil(2 * (z / D) ** 2)


def tost_power(n, true_d=0.0):
    se = sqrt(2 / n)
    z = norm.ppf(1 - ALPHA)
    return max(0.0, norm.cdf((D - true_d) / se - z) + norm.cdf((D + true_d) / se - z) - 1)


def n_equivalence():
    n = 10
    while tost_power(n) < POWER:
        n += 1
    return n


n_diff, n_eq = n_difference(), n_equivalence()
n = max(n_diff, n_eq)
total = GROUPS * n
se = sqrt(2 / n)
power_diff = 1 - norm.cdf(norm.ppf(1 - ALPHA / 2) - D / se)
print(f"alpha per test {ALPHA:.4f}; a difference of d = {D} needs {n_diff} a group, the equivalence test {n_eq};"
      f" the design uses {n} a group, {total} in all, about {total / COUNTRIES:.0f} in each of {COUNTRIES} countries")
print(f"power for a difference of d = {D}: {power_diff:.3f}; for the equivalence test at a true zero: {tost_power(n):.3f}")
bonus = total * REPLACE * BONUS * FEE
for minutes in SESSION_MINUTES:
    pay = total * REPLACE * minutes * RATE * FEE
    print(f"cost at {minutes} minutes: participant pay {pay:,.0f} USD + paid extra work {bonus:,.0f} USD = "
          f"{pay + bonus:,.0f} USD; with translation {pay + bonus + TRANSLATION:,.0f} USD (budget {BUDGET:,} USD; "
          f"no pilot)")
print(f"d = {D} is {D * 29:.2f} points of fun share at Chung et al.'s SD of 29")
# Can one test be significant and also inside the bounds? Only if the observed difference is both above the smallest
# significant difference and below the largest difference that the equivalence test accepts.
sig_d = norm.ppf(1 - ALPHA / 2) * se
eq_d = D - norm.ppf(1 - ALPHA) * se
print(f"with {n} a group the SE of a difference is {se:.4f} d: a significant difference needs |estimate| > {sig_d:.4f} d "
      f"(two-sided, alpha {ALPHA:.2f}); equivalence needs |estimate| < {D} - {norm.ppf(1 - ALPHA):.4f} x {se:.4f} = "
      f"{eq_d:.4f} d; both at once are impossible, unless the SD behind the bound exceeds the SD behind the SE by "
      f"more than {100 * ((sig_d + norm.ppf(1 - ALPHA) * se) / D - 1):.1f} percent")
print("power of the equivalence test at a true difference of: " + "; ".join(
    f"{t:.2f} d {tost_power(n, t):.3f}" for t in (0.0, 0.05, 0.10, 0.15)))
