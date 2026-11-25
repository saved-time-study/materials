"""Weighted least squares with absorbed fixed effects and cluster-robust standard errors.

Fixed effects are removed by weighted within-group demeaning: exactly for one fixed effect, and by alternating
projections (repeated demeaning over each fixed effect until the largest change is below 1e-10) for several.
The cluster-robust variance uses the small-sample factor G/(G-1) * (N-1)/(N-K). K counts the regressors and,
with count_fe=True and one fixed effect, also the fixed-effect groups (statsmodels' convention with explicit
dummies). With count_fe=False the absorbed groups are not counted, which is the convention when the fixed
effect is nested in the clusters (as in Stata's reghdfe). With two cluster variables the variance is
V(a) + V(b) - V(a and b) (Cameron, Gelbach and Miller 2011), each with its own G/(G-1) factor, and intervals and
p-values use a t distribution with min(G) - 1 degrees of freedom.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _absorb(cols, groups, W, tol=1e-10, max_iter=10000):
    """Weighted demeaning of each column of `cols` over every grouping in `groups`."""
    invs, wsums = [], []
    for g in groups:
        inv = pd.factorize(g)[0]
        invs.append(inv)
        wsums.append(np.bincount(inv, weights=W))
    out = []
    for v in cols:
        v = v.astype(float).copy()
        for _ in range(max_iter):
            change = 0.0
            for inv, ws in zip(invs, wsums):
                m = np.bincount(inv, weights=W * v) / ws
                v -= m[inv]
                change = max(change, float(np.max(np.abs(m))))
            if len(invs) == 1 or change < tol:
                break
        out.append(v)
    return out, [int(inv.max()) + 1 for inv in invs]


def _meat(X, We, labels):
    cl = pd.factorize(labels)[0]
    G = cl.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, cl, X * We[:, None])
    return S.T @ S, G


def fe_wls(df, y, x, fe, w, cluster, count_fe=False):
    fes = [fe] if isinstance(fe, str) else list(fe)
    cls = [cluster] if isinstance(cluster, str) else list(cluster)
    cols = list(dict.fromkeys([y] + list(x) + fes + [w] + cls))
    d = df[cols].dropna()
    W = d[w].to_numpy(float)
    demeaned, ngroups = _absorb([d[y].to_numpy(float)] + [d[c].to_numpy(float) for c in x],
                                [d[f].to_numpy() for f in fes], W)
    Y = demeaned[0]
    X = np.column_stack(demeaned[1:])
    keep = np.sqrt((W[:, None] * X ** 2).sum(0)) > 1e-9
    names = [c for c, k in zip(x, keep) if k]
    X = X[:, keep]
    XtWX = X.T @ (X * W[:, None])
    bread = np.linalg.pinv(XtWX)
    beta = bread @ (X.T @ (W * Y))
    e = Y - X @ beta
    N = len(Y)
    K = X.shape[1] + (ngroups[0] if (count_fe and len(fes) == 1) else 0)
    small = (N - 1) / (N - K)
    We = W * e
    if len(cls) == 1:
        M, G = _meat(X, We, d[cls[0]].to_numpy())
        V = bread @ M @ bread * (G / (G - 1)) * small
        Gs = [G]
    elif len(cls) == 2:
        a, b = d[cls[0]].astype(str), d[cls[1]].astype(str)
        Ma, Ga = _meat(X, We, a.to_numpy())
        Mb, Gb = _meat(X, We, b.to_numpy())
        Mab, Gab = _meat(X, We, (a + "|" + b).to_numpy())
        V = bread @ (Ma * Ga / (Ga - 1) + Mb * Gb / (Gb - 1) - Mab * Gab / (Gab - 1)) @ bread * small
        Gs = [Ga, Gb]
    else:
        raise ValueError("one or two cluster variables")
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    df_t = min(Gs) - 1
    tcrit = stats.t.ppf(0.975, df_t)
    out = pd.DataFrame({"coef": beta, "se": se}, index=names)
    out["t"] = out["coef"] / out["se"]
    out["p"] = 2 * stats.t.sf(np.abs(out["t"]), df_t)
    out["ci_lo"] = out["coef"] - tcrit * out["se"]
    out["ci_hi"] = out["coef"] + tcrit * out["se"]
    out.attrs.update(n=N, clusters=min(Gs), cluster_counts=Gs, vcov=pd.DataFrame(V, index=names, columns=names))
    return out


def wald(res, terms):
    """Joint test that the named coefficients are all zero (F with q and G-1 degrees of freedom)."""
    b = res.loc[terms, "coef"].to_numpy()
    V = res.attrs["vcov"].loc[terms, terms].to_numpy()
    q = len(terms)
    F = float(b @ np.linalg.solve(V, b)) / q
    p = float(stats.f.sf(F, q, res.attrs["clusters"] - 1))
    return F, p
