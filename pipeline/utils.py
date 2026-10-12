"""
Shared utilities for MultiPICo variance analysis.
Implements Hu & Collier's mixed-effects regression framework
with Nakagawa-Schielzeth R-squared for binary outcomes.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM


def compute_r_squared_nakagawa(model):
    """
    Compute marginal and conditional R-squared following
    Nakagawa & Schielzeth (2013), extended by Nakagawa, Johnson
    & Schielzeth (2017) for GLMMs.

    For a linear mixed model with a single random intercept:
      R²_m = σ²_f / (σ²_f + σ²_u + σ²_e)
      R²_c = (σ²_f + σ²_u) / (σ²_f + σ²_u + σ²_e)

    For binary outcomes treated via LPM (linear probability model),
    we also report the logistic-distribution-specific variant where
    σ²_dist = π²/3 replaces σ²_e.
    """
    # Variance of the fixed-effects predictions only (X @ beta)
    fixed_pred = np.dot(model.model.exog, model.fe_params)
    fixed_var = np.var(fixed_pred)

    random_var = float(model.cov_re.iloc[0, 0])
    resid_var = model.scale

    total_var = fixed_var + random_var + resid_var

    r2_marginal = fixed_var / total_var
    r2_conditional = (fixed_var + random_var) / total_var

    # Distribution-specific variant for logistic (binary outcome)
    dist_var = (np.pi ** 2) / 3
    total_var_logistic = fixed_var + random_var + dist_var
    r2_marginal_logistic = fixed_var / total_var_logistic
    r2_conditional_logistic = (fixed_var + random_var) / total_var_logistic

    return {
        "r2_marginal": r2_marginal,
        "r2_conditional": r2_conditional,
        "r2_marginal_logistic": r2_marginal_logistic,
        "r2_conditional_logistic": r2_conditional_logistic,
        "var_fixed": fixed_var,
        "var_random": random_var,
        "var_residual": resid_var,
    }


def fit_lmm(formula, data, groups_col="text_id"):
    """
    Fit a linear mixed model with a random intercept for text_id.
    Returns the fitted model and R-squared statistics.
    """
    model = smf.mixedlm(formula, data, groups=data[groups_col])
    result = model.fit(reml=True, method="lbfgs")
    r2 = compute_r_squared_nakagawa(result)
    return result, r2


def variable_importance(data, outcome, variables, groups_col="text_id"):
    """
    Leave-one-out variable importance: fit the full model and then
    drop each variable to measure the decrease in marginal R-squared.
    """
    full_formula = f"{outcome} ~ {' + '.join(variables)}"
    _, full_r2 = fit_lmm(full_formula, data, groups_col)

    importance = {}
    for var in variables:
        reduced_vars = [v for v in variables if v != var]
        if not reduced_vars:
            reduced_formula = f"{outcome} ~ 1"
        else:
            reduced_formula = f"{outcome} ~ {' + '.join(reduced_vars)}"
        _, reduced_r2 = fit_lmm(reduced_formula, data, groups_col)
        importance[var] = full_r2["r2_marginal"] - reduced_r2["r2_marginal"]

    return importance, full_r2


def interaction_lr_test(data, base_formula, interaction_term, groups_col="text_id"):
    """
    Likelihood ratio test for adding an interaction term.
    Returns the test statistic, degrees of freedom, and p-value.
    """
    from scipy import stats

    base_model = smf.mixedlm(base_formula, data, groups=data[groups_col])
    base_result = base_model.fit(reml=False, method="lbfgs")

    full_formula = f"{base_formula} + {interaction_term}"
    full_model = smf.mixedlm(full_formula, data, groups=data[groups_col])
    full_result = full_model.fit(reml=False, method="lbfgs")

    lr_stat = -2 * (base_result.llf - full_result.llf)
    df_diff = full_result.df_modelwc - base_result.df_modelwc
    p_value = stats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else 1.0

    base_r2 = compute_r_squared_nakagawa(base_result)
    full_r2 = compute_r_squared_nakagawa(full_result)

    return {
        "lr_statistic": lr_stat,
        "df": df_diff,
        "p_value": p_value,
        "r2_base": base_r2["r2_marginal"],
        "r2_full": full_r2["r2_marginal"],
        "r2_change": full_r2["r2_marginal"] - base_r2["r2_marginal"],
    }
