"""
fastlmm.py
==========
Small helpers for the null and reference checks (scripts 104, 105).

TextGLS   : GLS for y = Xb + u_text + e with the variance components held
            fixed (quasi-demeaning). With the components from a fitted
            statsmodels MixedLM this reproduces its fixed effects exactly.
CrossedML : exact maximum likelihood for
            y = Xb + u_text + u_annotator + e   (crossed random intercepts),
            profiled over the two variance ratios. Annotators are few, so
            the text block is eliminated analytically and only an
            annotator-sized matrix is factorised.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def group_matrix(codes):
    n = len(codes)
    return sp.csr_matrix((np.ones(n), (codes, np.arange(n))),
                         shape=(codes.max() + 1, n))


def nested_design(rhs_base, added, data):
    """Full design with the base columns first. patsy reorders terms
    (categorical before numeric), so the full design cannot be built from
    one formula and sliced."""
    import patsy
    Xb = patsy.dmatrix(rhs_base, data, return_type="dataframe").values
    Xn = patsy.dmatrix("0 + " + added, data, return_type="dataframe").values
    return np.column_stack([Xb, Xn]), Xb.shape[1]


def prune_cols(X, n_base=None):
    keep = []
    for j in range(X.shape[1]):
        cand = keep + [j]
        if np.linalg.matrix_rank(X[:, cand]) == len(cand):
            keep = cand
    if n_base is None:
        return X[:, keep]
    return X[:, keep], sum(1 for j in keep if j < n_base)


class TextGLS:
    def __init__(self, text_codes, var_u, var_e):
        self.codes = text_codes
        self.T = group_matrix(text_codes)
        self.cnt = np.asarray(self.T.sum(axis=1)).ravel()
        theta = 1 - np.sqrt(var_e / (var_e + self.cnt * var_u))
        self.theta_row = theta[text_codes]
        self.var_u, self.var_e = var_u, var_e

    def transform(self, M):
        M = np.asarray(M, dtype=float)
        if M.ndim == 1:
            M = M[:, None]
        means = (self.T @ M) / self.cnt[:, None]
        return M - self.theta_row[:, None] * means[self.codes]

    def var_fixed(self, X, yt):
        b, *_ = np.linalg.lstsq(self.transform(X), yt, rcond=None)
        return float(np.var(X @ b))


def shuffled_profile_null(X, y, text_codes, ann_codes, var_u, var_e, n_perm, rng):
    """R^2_m with the observed profiles and under N whole-profile shuffles
    across annotators (text-intercept model, components held fixed)."""
    first = pd.Series(np.arange(len(y))).groupby(ann_codes).first().values
    A = X[first]
    g = TextGLS(text_codes, var_u, var_e)
    yt = g.transform(y).ravel()

    def r2m(M):
        vf = g.var_fixed(M, yt)
        return vf / (vf + var_u + var_e)

    obs = r2m(X)
    null = np.array([r2m(A[rng.permutation(len(first))][ann_codes]) for _ in range(n_perm)])
    return {"n_annotators": len(first), "params": np.linalg.matrix_rank(A),
            "r2m": obs, "null_mean": null.mean(), "null_p95": np.quantile(null, 0.95),
            "perm_p": (np.sum(null >= obs) + 1) / (n_perm + 1),
            "excess": obs - null.mean()}


class CrossedML:
    def __init__(self, y, text_codes, ann_codes):
        self.y = np.asarray(y, dtype=float)
        self.n = len(self.y)
        self.T = group_matrix(text_codes)
        self.A = group_matrix(ann_codes)
        self.Dt = np.asarray(self.T.sum(axis=1)).ravel()
        self.Da = np.asarray(self.A.sum(axis=1)).ravel()
        self.C = (self.T @ self.A.T).toarray()

    def _setX(self, X):
        Xy = np.column_stack([X, self.y])
        self.p = X.shape[1]
        self.G0 = Xy.T @ Xy
        self.Wt = self.T @ Xy
        self.Wa = self.A @ Xy

    def _profile(self, lg, reml=False):
        gt, ga = np.exp(lg)
        P = self.Dt + 1.0 / gt
        CP = self.C / P[:, None]
        Q = self.C.T @ CP
        Stil = np.eye(len(self.Da)) + ga * (np.diag(self.Da) - Q)
        L = cho_factor(Stil)
        ra = self.Wa - CP.T @ self.Wt
        xa = ga * cho_solve(L, ra)
        xt = (self.Wt - self.C @ xa) / P[:, None]
        G = self.G0 - (self.Wt.T @ xt + self.Wa.T @ xa)
        p = self.p
        beta = np.linalg.solve(G[:p, :p], G[:p, p])
        rss = G[p, p] - G[:p, p] @ beta
        logdet = np.sum(np.log(gt * self.Dt + 1)) + 2 * np.sum(np.log(np.diag(L[0])))
        if reml:
            m = self.n - p
            dev = (m * np.log(2 * np.pi * rss / m) + logdet
                   + np.linalg.slogdet(G[:p, :p])[1] + m)
            return dev, beta, rss / m
        dev = self.n * np.log(2 * np.pi * rss / self.n) + logdet + self.n
        return dev, beta, rss / self.n

    def fit(self, X, start=(np.log(0.3), np.log(0.3)), reml=False):
        self._setX(X)
        best = None
        for s in [start, (np.log(0.5), np.log(0.1)), (np.log(0.1), np.log(0.5))]:
            r = minimize(lambda lg: self._profile(lg, reml)[0], np.array(s),
                         method="L-BFGS-B", bounds=[(-14, 4), (-14, 4)])
            if best is None or r.fun < best.fun:
                best = r
        dev, beta, s2 = self._profile(best.x, reml)
        gt, ga = np.exp(best.x)
        vf = float(np.var(X @ beta))
        return {"llf": -dev / 2, "beta": beta, "var_e": s2,
                "var_text": gt * s2, "var_ann": ga * s2, "var_fixed": vf,
                "r2m": vf / (vf + (gt + ga + 1) * s2), "lg": best.x}
