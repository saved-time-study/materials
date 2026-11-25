"""Tests of fe.py against statsmodels with explicit dummies.  Run: python -I data/atus/scripts/test_fe.py"""
import pathlib
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fe import fe_wls  # noqa: E402


def synthetic(seed=1, n=3000, groups=60):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    a = rng.normal(0, 2, groups)[g]
    x1 = rng.normal(size=n) + 0.3 * a
    x2 = (rng.random(n) < 0.4).astype(float)
    y = 1.5 * x1 - 2.0 * x2 + a + rng.normal(0, 1 + 0.5 * x2, n) + rng.normal(0, 1, groups)[g]
    w = rng.uniform(0.2, 3, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, "w": w})


def test_matches_statsmodels():
    d = synthetic()
    ours = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster="g", count_fe=True)
    sm = smf.wls("y ~ x1 + x2 + C(g)", data=d, weights=d["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"]})
    for v in ["x1", "x2"]:
        assert abs(ours.loc[v, "coef"] - sm.params[v]) < 1e-8, v
        assert abs(ours.loc[v, "se"] / sm.bse[v] - 1) < 1e-6, (v, ours.loc[v, "se"], sm.bse[v])


def test_recovers_truth():
    d = synthetic(seed=7, n=20000, groups=200)
    r = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster="g")
    assert r.loc["x1", "ci_lo"] < 1.5 < r.loc["x1", "ci_hi"]
    assert r.loc["x2", "ci_lo"] < -2.0 < r.loc["x2", "ci_hi"]


def test_adding_up():
    """Outcomes that add to a constant get coefficients that add to zero (the 1,440 minutes check)."""
    d = synthetic(seed=3)
    rng = np.random.default_rng(0)
    d["a"] = rng.normal(size=len(d)) + d["x1"]
    d["b"] = 1440 - d["a"] - d["y"]
    s = sum(fe_wls(d, o, ["x1", "x2"], fe="g", w="w", cluster="g").loc["x1", "coef"] for o in ["a", "b", "y"])
    assert abs(s) < 1e-8


def test_planted_fault_is_caught():
    """Control: a wrong weight vector must break agreement with statsmodels."""
    d = synthetic()
    ours = fe_wls(d.assign(w=1.0), "y", ["x1", "x2"], fe="g", w="w", cluster="g", count_fe=True)
    sm = smf.wls("y ~ x1 + x2 + C(g)", data=d, weights=d["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["g"]})
    assert abs(ours.loc["x1", "coef"] - sm.params["x1"]) > 1e-4


def test_several_fixed_effects_match_dummies():
    """Three absorbed fixed effects give the coefficients of explicit dummies for all three."""
    d = synthetic(seed=5, n=4000, groups=40)
    rng = np.random.default_rng(2)
    d["h"] = rng.integers(0, 15, len(d))
    d["k"] = rng.integers(0, 7, len(d))
    d["y"] = d["y"] + 0.5 * d["h"] - 0.3 * d["k"]
    ours = fe_wls(d, "y", ["x1", "x2"], fe=["g", "h", "k"], w="w", cluster="g")
    sm = smf.wls("y ~ x1 + x2 + C(g) + C(h) + C(k)", data=d, weights=d["w"]).fit()
    for v in ["x1", "x2"]:
        assert abs(ours.loc[v, "coef"] - sm.params[v]) < 1e-7, (v, ours.loc[v, "coef"], sm.params[v])


def test_two_way_cluster():
    """Two-way clustering: identical cluster variables give the one-way result; a second, coarser cluster
    gives the coefficients unchanged and the variance V(a) + V(b) - V(a and b)."""
    d = synthetic(seed=9, n=3000, groups=50)
    rng = np.random.default_rng(4)
    d["s"] = rng.integers(0, 20, len(d))
    one = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster="g")
    same = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster=["g", "g"])
    assert np.allclose(one["se"], same["se"], rtol=1e-6)
    two = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster=["g", "s"])
    assert np.allclose(one["coef"], two["coef"])
    a = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster="s")
    d["gs"] = d["g"].astype(str) + "|" + d["s"].astype(str)
    ab = fe_wls(d, "y", ["x1", "x2"], fe="g", w="w", cluster="gs")
    expect = one["se"] ** 2 + a["se"] ** 2 - ab["se"] ** 2
    assert np.allclose(two["se"] ** 2, expect, rtol=1e-6), (two["se"] ** 2, expect)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("pass", t.__name__)
    print(f"{len(tests)} of {len(tests)} tests passed")
