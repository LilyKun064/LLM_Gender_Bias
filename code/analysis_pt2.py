#!/usr/bin/env python
"""
analysis_pt2.py

Hierarchical Bayesian analysis of explanation reasons (Appendix II).

UPDATED:
- Content reasons (fact, tone_reason, style, emotion) are now modeled
  jointly using a hierarchical logistic-normal compositional model
  (Option A).
- Stereotype-related reasons (stereo, avoid_stereo, other, mentions_stereotype)
  remain hierarchical logistic (Bernoulli) models, as before.

Data structure:

- Content reasons (Table 1):
    fact, tone_reason, style, emotion
  These are *weights* in [0,1], and for each row they sum to 1.
  We now model the 4-part composition jointly with a logistic-normal
  regression (multivariate).

- Stereotype-related reasons (Table 2):
    stereo, avoid_stereo, other
  These are mutually exclusive binary indicators (0/1); in each row,
  exactly one of these is 1. We fit separate hierarchical logistic
  (Bernoulli) models for each.

- Optionally, a mentions_stereotype column (0/1) can also be modeled
  with the same Bernoulli structure if present.

The grouping structure matches analysis_pt1:

- model
- scenario
- tone
- factor (occupation / food / hobby_profile)
- model × scenario interaction

Outputs:

Content composition (all four reasons jointly):
    content_reasons/
        trace_content_composition.nc
        summary_global_content.csv
        summary_model_content.csv
        summary_scenario_content.csv
        summary_tone_content.csv
        summary_pairwise_model_content.csv

Binary stereotype-related reasons (each separately, as before):
    stereotype_reasons/{code}/
        trace_{code}.nc
        summary_global_and_sigma_{code}.csv
        summary_model_{code}.csv
        summary_scenario_{code}.csv
        summary_tone_{code}.csv
        summary_pairwise_model_{code}.csv
"""

from pathlib import Path
import argparse

import nutpie  # ensure nutpie is available

# ------------------------------------------------------------
# Configure PyTensor BEFORE importing pymc
# ------------------------------------------------------------
import pytensor
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.exception_verbosity = "high"

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def hdi_interval(samples, hdi_prob=0.95):
    """Return (low, high) of the HDI for a 1D array."""
    hdi = az.hdi(samples, hdi_prob=hdi_prob)
    return float(hdi[0]), float(hdi[1])


def summarize_scalar(samples, name):
    samples = np.asarray(samples)
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    hdi_low, hdi_high = hdi_interval(samples)
    return {
        "Parameter": name,
        "Mean": mean,
        "SD": sd,
        "HDI 2.5%": hdi_low,
        "HDI 97.5%": hdi_high,
    }


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add unified 'factor' variable and categorical indices.

    factor =
        occupation       if scenario == "cover_letter"
        food             if scenario == "potluck"
        hobby_profile    otherwise
    """
    df = df.copy()

    factor = np.where(
        df["scenario"] == "cover_letter",
        df["occupation"],
        np.where(
            df["scenario"] == "potluck",
            df["food"],
            df["hobby_profile"],
        ),
    )
    df["factor"] = factor
    df = df.dropna(subset=["factor"])

    # Categorical encodings
    df["model_cat"] = pd.Categorical(df["model"])
    df["scenario_cat"] = pd.Categorical(df["scenario"])
    df["tone_cat"] = pd.Categorical(df["tone"])
    df["factor_cat"] = pd.Categorical(df["factor"])

    df["model_scenario"] = df["model"] + "::" + df["scenario"]
    df["model_scenario_cat"] = pd.Categorical(df["model_scenario"])

    return df


def build_index_dicts(df: pd.DataFrame):
    """Extract integer indices and level names for all grouping factors."""
    model_levels = df["model_cat"].cat.categories.tolist()
    scenario_levels = df["scenario_cat"].cat.categories.tolist()
    tone_levels = df["tone_cat"].cat.categories.tolist()
    factor_levels = df["factor_cat"].cat.categories.tolist()
    model_scenario_levels = df["model_scenario_cat"].cat.categories.tolist()

    idx = {
        "model_idx": df["model_cat"].cat.codes.to_numpy(),
        "scenario_idx": df["scenario_cat"].cat.codes.to_numpy(),
        "tone_idx": df["tone_cat"].cat.codes.to_numpy(),
        "factor_idx": df["factor_cat"].cat.codes.to_numpy(),
        "model_scenario_idx": df["model_scenario_cat"].cat.codes.to_numpy(),
    }

    levels = {
        "model": model_levels,
        "scenario": scenario_levels,
        "tone": tone_levels,
        "factor": factor_levels,
        "model_scenario": model_scenario_levels,
    }

    return idx, levels


# ------------------------------------------------------------
# Logistic-Normal Compositional Model for Content Reasons
# ------------------------------------------------------------

def alr_transform(Y, eps=1e-9):
    """
    Convert 4-part compositions (fact, tone_reason, style, emotion)
    into 3D additive log-ratio (ALR) coordinates using the 4th
    component (emotion) as reference:

    y_alr = [log(p1/p4), log(p2/p4), log(p3/p4)]
    """
    Y = np.asarray(Y, dtype=float)
    Y = np.clip(Y, eps, 1.0)
    Y = Y / Y.sum(axis=1, keepdims=True)

    p1 = Y[:, 0]
    p2 = Y[:, 1]
    p3 = Y[:, 2]
    p4 = Y[:, 3]

    y1 = np.log(p1 / p4)
    y2 = np.log(p2 / p4)
    y3 = np.log(p3 / p4)

    return np.column_stack([y1, y2, y3])


def alr_to_simplex(z_alr):
    """
    Convert ALR coordinates (length-3) back to 4-simplex
    (fact, tone_reason, style, emotion) by appending 0 as
    the log-reference and applying softmax.
    """
    z_full = np.zeros(4, dtype=float)
    z_full[:3] = z_alr
    p = softmax(z_full, axis=0)
    return p  # length-4 vector


def fit_content_composition_model(
    Y,
    idx,
    levels,
    outdir: Path,
    draws: int = 1500,
    tune: int = 1500,
    target_accept: float = 0.97,
    random_seed: int = 123,
):
    """
    Faster hierarchical logistic-normal regression for the 4-part composition:
    [fact, tone_reason, style, emotion].

    Changes vs previous version:
    - Still uses ALR (logistic-normal) structure.
    - But replaces MVN + LKJ covariance with *independent* Normal likelihoods
      for each ALR dimension (diagonal residual covariance).
    - Greatly reduces computational cost.
    """

    outdir.mkdir(parents=True, exist_ok=True)

    CONTENT_CODES = ["fact", "tone_reason", "style", "emotion"]
    n_reason = len(CONTENT_CODES)
    n_alr = n_reason - 1  # 3

    # Convert to ALR space for observed data
    y_alr = alr_transform(Y)  # shape (n, 3)
    n = y_alr.shape[0]

    n_model = len(levels["model"])
    n_scenario = len(levels["scenario"])
    n_tone = len(levels["tone"])
    n_factor = len(levels["factor"])
    n_model_scenario = len(levels["model_scenario"])

    model_idx = idx["model_idx"]
    scenario_idx = idx["scenario_idx"]
    tone_idx = idx["tone_idx"]
    factor_idx = idx["factor_idx"]
    model_scenario_idx = idx["model_scenario_idx"]

    with pm.Model() as m:
        # Global intercept in ALR space (3 dims)
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5, shape=n_alr)

        # Random-effect scales
        sigma_model = pm.HalfNormal("sigma_model", sigma=0.5)
        sigma_scenario = pm.HalfNormal("sigma_scenario", sigma=0.5)
        sigma_tone = pm.HalfNormal("sigma_tone", sigma=0.5)
        sigma_factor = pm.HalfNormal("sigma_factor", sigma=0.5)
        sigma_model_scenario = pm.HalfNormal("sigma_model_scenario", sigma=0.5)

        # Vector-valued non-centered random effects in ALR space
        L_model = pm.Normal("L_model", 0.0, 1.0, shape=(n_model, n_alr))
        L_scenario = pm.Normal("L_scenario", 0.0, 1.0, shape=(n_scenario, n_alr))
        L_tone = pm.Normal("L_tone", 0.0, 1.0, shape=(n_tone, n_alr))
        L_factor = pm.Normal("L_factor", 0.0, 1.0, shape=(n_factor, n_alr))
        L_model_scenario = pm.Normal("L_model_scenario", 0.0, 1.0, shape=(n_model_scenario, n_alr))

        a_model = pm.Deterministic("a_model", L_model * sigma_model)
        a_scenario = pm.Deterministic("a_scenario", L_scenario * sigma_scenario)
        a_tone = pm.Deterministic("a_tone", L_tone * sigma_tone)
        a_factor = pm.Deterministic("a_factor", L_factor * sigma_factor)
        a_model_scenario = pm.Deterministic(
            "a_model_scenario", L_model_scenario * sigma_model_scenario
        )

        # Linear predictor in ALR space
        eta = (
            alpha
            + a_model[model_idx]
            + a_scenario[scenario_idx]
            + a_tone[tone_idx]
            + a_factor[factor_idx]
            + a_model_scenario[model_scenario_idx]
        )  # shape (n, 3)

        # Residual standard deviations for each ALR dimension (diagonal covariance)
        sigma_resid = pm.HalfNormal("sigma_resid", sigma=1.0, shape=n_alr)

        # Independent Normal likelihood in ALR space (much cheaper than MVN+LKJ)
        pm.Normal("y_obs", mu=eta, sigma=sigma_resid, observed=y_alr)

        print("[INFO] Using standard PyMC NUTS sampler for *content composition* model (diagonal residual).")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            cores=4,        # 4 cores 
            chains=4,       # 4 chains, can reduce to 2 if still heavy
        )

    # Save posterior draws
    az.to_netcdf(idata, outdir / "trace_content_composition.nc")

    # ----------------------------------------
    # Summaries on the original simplex scale
    # ----------------------------------------
    post = idata.posterior
    # Flatten chains and draws: (n_samples, 3)
    alpha_vals = post["alpha"].values.reshape(-1, n_alr)

    a_model_vals = post["a_model"].values.reshape(-1, n_model, n_alr)
    a_scenario_vals = post["a_scenario"].values.reshape(-1, n_scenario, n_alr)
    a_tone_vals = post["a_tone"].values.reshape(-1, n_tone, n_alr)

    sigma_model_vals = post["sigma_model"].values.reshape(-1)
    sigma_scenario_vals = post["sigma_scenario"].values.reshape(-1)
    sigma_tone_vals = post["sigma_tone"].values.reshape(-1)
    sigma_factor_vals = post["sigma_factor"].values.reshape(-1)
    sigma_model_scenario_vals = post["sigma_model_scenario"].values.reshape(-1)

    n_samples = alpha_vals.shape[0]

    def baseline_composition(effect=None):
        """
        Compute posterior samples of composition (n_samples x 4)
        at a given baseline (alpha + effect).
        effect: array (n_samples, 3) or None
        """
        if effect is None:
            z_alr = alpha_vals  # (n_samples, 3)
        else:
            z_alr = alpha_vals + effect  # broadcast

        comp = np.zeros((n_samples, n_reason), dtype=float)
        for s in range(n_samples):
            comp[s, :] = alr_to_simplex(z_alr[s, :])
        return comp  # (n_samples, 4)

    # -------------------------
    # 1) Global baselines
    # -------------------------
    comp_global = baseline_composition(effect=None)

    rows_global = []
    for r, code in enumerate(CONTENT_CODES):
        stats = summarize_scalar(comp_global[:, r], name=f"p_global_{code}")
        rows_global.append(
            {
                "Reason": code,
                "Mean": stats["Mean"],
                "SD": stats["SD"],
                "HDI 2.5%": stats["HDI 2.5%"],
                "HDI 97.5%": stats["HDI 97.5%"],
            }
        )

    # Variance components
    var_rows = []
    for param_name, vals in [
        ("sigma_model", sigma_model_vals),
        ("sigma_scenario", sigma_scenario_vals),
        ("sigma_tone", sigma_tone_vals),
        ("sigma_factor", sigma_factor_vals),
        ("sigma_model_scenario", sigma_model_scenario_vals),
    ]:
        stats = summarize_scalar(vals, name=param_name)
        var_rows.append(
            {
                "Reason": param_name,
                "Mean": stats["Mean"],
                "SD": stats["SD"],
                "HDI 2.5%": stats["HDI 2.5%"],
                "HDI 97.5%": stats["HDI 97.5%"],
            }
        )

    df_global = pd.DataFrame(rows_global + var_rows)
    df_global.to_csv(outdir / "summary_global_content.csv", index=False)

    # -------------------------
    # 2) Model-level baselines
    # -------------------------
    rows_model = []
    for j, model_name in enumerate(levels["model"]):
        effect_j = a_model_vals[:, j, :]  # (n_samples, 3)
        comp_model_j = baseline_composition(effect=effect_j)
        for r, code in enumerate(CONTENT_CODES):
            stats = summarize_scalar(
                comp_model_j[:, r], name=f"p_{code}_model_{model_name}"
            )
            rows_model.append(
                {
                    "Model": model_name,
                    "Reason": code,
                    "Mean": stats["Mean"],
                    "SD": stats["SD"],
                    "HDI 2.5%": stats["HDI 2.5%"],
                    "HDI 97.5%": stats["HDI 97.5%"],
                }
            )
    pd.DataFrame(rows_model).to_csv(
        outdir / "summary_model_content.csv", index=False
    )

    # -------------------------
    # 3) Scenario-level baselines
    # -------------------------
    rows_scen = []
    for j, scen_name in enumerate(levels["scenario"]):
        effect_j = a_scenario_vals[:, j, :]
        comp_scen_j = baseline_composition(effect=effect_j)
        for r, code in enumerate(CONTENT_CODES):
            stats = summarize_scalar(
                comp_scen_j[:, r], name=f"p_{code}_scenario_{scen_name}"
            )
            rows_scen.append(
                {
                    "Scenario": scen_name,
                    "Reason": code,
                    "Mean": stats["Mean"],
                    "SD": stats["SD"],
                    "HDI 2.5%": stats["HDI 2.5%"],
                    "HDI 97.5%": stats["HDI 97.5%"],
                }
            )
    pd.DataFrame(rows_scen).to_csv(
        outdir / "summary_scenario_content.csv", index=False
    )

    # -------------------------
    # 4) Tone-level baselines
    # -------------------------
    rows_tone = []
    for j, tone_name in enumerate(levels["tone"]):
        effect_j = a_tone_vals[:, j, :]
        comp_tone_j = baseline_composition(effect=effect_j)
        for r, code in enumerate(CONTENT_CODES):
            stats = summarize_scalar(
                comp_tone_j[:, r], name=f"p_{code}_tone_{tone_name}"
            )
            rows_tone.append(
                {
                    "Tone": tone_name,
                    "Reason": code,
                    "Mean": stats["Mean"],
                    "SD": stats["SD"],
                    "HDI 2.5%": stats["HDI 2.5%"],
                    "HDI 97.5%": stats["HDI 97.5%"],
                }
            )
    pd.DataFrame(rows_tone).to_csv(
        outdir / "summary_tone_content.csv", index=False
    )

    # -------------------------
    # 5) Pairwise model differences (per reason)
    # -------------------------
    rows_pair = []
    n_model = len(levels["model"])
    for i in range(n_model):
        m1 = levels["model"][i]
        effect_1 = a_model_vals[:, i, :]
        comp_1 = baseline_composition(effect=effect_1)

        for j in range(i + 1, n_model):
            m2 = levels["model"][j]
            effect_2 = a_model_vals[:, j, :]
            comp_2 = baseline_composition(effect=effect_2)

            diff = comp_2 - comp_1  # (n_samples, 4)
            for r, code in enumerate(CONTENT_CODES):
                diff_r = diff[:, r]
                stats = summarize_scalar(
                    diff_r, name=f"p_{code}_{m2} - p_{code}_{m1}"
                )
                rows_pair.append(
                    {
                        "Model 1": m1,
                        "Model 2": m2,
                        "Reason": code,
                        "Mean": stats["Mean"],
                        "SD": stats["SD"],
                        "HDI 2.5%": stats["HDI 2.5%"],
                        "HDI 97.5%": stats["HDI 97.5%"],
                        "Prob > 0": float((diff_r > 0).mean()),
                    }
                )

    pd.DataFrame(rows_pair).to_csv(
        outdir / "summary_pairwise_model_content.csv", index=False
    )

# ------------------------------------------------------------
# Binary stereotype models (unchanged)
# ------------------------------------------------------------

def fit_binary_reason_model(
    y,
    idx,
    levels,
    outdir: Path,
    reason_code: str,
    draws: int = 2000,
    tune: int = 2000,
    target_accept: float = 0.97,
    random_seed: int = 123,
):
    """
    Hierarchical logistic model for a single *binary* reason code
    (stereo, avoid_stereo, other, mentions_stereotype, etc.).

    y: binary array (0/1).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    n_model = len(levels["model"])
    n_scenario = len(levels["scenario"])
    n_tone = len(levels["tone"])
    n_factor = len(levels["factor"])
    n_model_scenario = len(levels["model_scenario"])

    model_idx = idx["model_idx"]
    scenario_idx = idx["scenario_idx"]
    tone_idx = idx["tone_idx"]
    factor_idx = idx["factor_idx"]
    model_scenario_idx = idx["model_scenario_idx"]

    with pm.Model() as m:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)

        sigma_model = pm.HalfNormal("sigma_model", sigma=0.5)
        sigma_scenario = pm.HalfNormal("sigma_scenario", sigma=0.5)
        sigma_tone = pm.HalfNormal("sigma_tone", sigma=0.5)
        sigma_factor = pm.HalfNormal("sigma_factor", sigma=0.5)
        sigma_model_scenario = pm.HalfNormal(
            "sigma_model_scenario", sigma=0.5
        )

        z_model = pm.Normal("z_model", 0.0, 1.0, shape=n_model)
        a_model = pm.Deterministic("a_model", z_model * sigma_model)

        z_scenario = pm.Normal("z_scenario", 0.0, 1.0, shape=n_scenario)
        a_scenario = pm.Deterministic("a_scenario", z_scenario * sigma_scenario)

        z_tone = pm.Normal("z_tone", 0.0, 1.0, shape=n_tone)
        a_tone = pm.Deterministic("a_tone", z_tone * sigma_tone)

        z_factor = pm.Normal("z_factor", 0.0, 1.0, shape=n_factor)
        a_factor = pm.Deterministic("a_factor", z_factor * sigma_factor)

        z_model_scenario = pm.Normal("z_model_scenario", 0.0, 1.0, shape=n_model_scenario)
        a_model_scenario = pm.Deterministic(
            "a_model_scenario", z_model_scenario * sigma_model_scenario
        )

        eta = (
            alpha
            + a_model[model_idx]
            + a_scenario[scenario_idx]
            + a_tone[tone_idx]
            + a_factor[factor_idx]
            + a_model_scenario[model_scenario_idx]
        )
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Bernoulli("y_obs", p=p, observed=y)

        print(f"[INFO] Using nutpie NUTS sampler for *binary* reason '{reason_code}'.")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            cores=4,
            nuts_sampler="nutpie",
        )

    trace_path = outdir / f"trace_{reason_code}.nc"
    az.to_netcdf(idata, trace_path)

    post = idata.posterior
    alpha_vals = post["alpha"].values.reshape(-1)
    a_model_vals = post["a_model"].values.reshape(-1, n_model)
    a_scenario_vals = post["a_scenario"].values.reshape(-1, n_scenario)
    a_tone_vals = post["a_tone"].values.reshape(-1, n_tone)
    a_factor_vals = post["a_factor"].values.reshape(-1, n_factor)

    sigma_model_vals = post["sigma_model"].values.reshape(-1)
    sigma_scenario_vals = post["sigma_scenario"].values.reshape(-1)
    sigma_tone_vals = post["sigma_tone"].values.reshape(-1)
    sigma_factor_vals = post["sigma_factor"].values.reshape(-1)
    sigma_model_scenario_vals = post["sigma_model_scenario"].values.reshape(-1)

    summaries = []
    p_global_samples = logistic(alpha_vals)
    summaries.append(summarize_scalar(p_global_samples, "p_global"))
    summaries.append(summarize_scalar(sigma_model_vals, "sigma_model"))
    summaries.append(summarize_scalar(sigma_scenario_vals, "sigma_scenario"))
    summaries.append(summarize_scalar(sigma_tone_vals, "sigma_tone"))
    summaries.append(summarize_scalar(sigma_factor_vals, "sigma_factor"))
    summaries.append(
        summarize_scalar(sigma_model_scenario_vals, "sigma_model_scenario")
    )
    pd.DataFrame(summaries).to_csv(
        outdir / f"summary_global_and_sigma_{reason_code}.csv",
        index=False,
    )

    # Scenario-level baselines
    scenario_rows = []
    for j, scen in enumerate(levels["scenario"]):
        p_scen = logistic(alpha_vals + a_scenario_vals[:, j])
        scenario_rows.append(
            {
                "Scenario": scen,
                **summarize_scalar(
                    p_scen,
                    name=f"p_{reason_code}_scenario_{scen}",
                ),
            }
        )
    pd.DataFrame(scenario_rows).to_csv(
        outdir / f"summary_scenario_{reason_code}.csv", index=False
    )

    # Tone-level baselines
    tone_rows = []
    for j, tone in enumerate(levels["tone"]):
        p_tone = logistic(alpha_vals + a_tone_vals[:, j])
        tone_rows.append(
            {
                "Tone": tone,
                **summarize_scalar(
                    p_tone,
                    name=f"p_{reason_code}_tone_{tone}",
                ),
            }
        )
    pd.DataFrame(tone_rows).to_csv(
        outdir / f"summary_tone_{reason_code}.csv", index=False
    )

    # Model-level baselines
    model_rows = []
    for j, model_name in enumerate(levels["model"]):
        p_model = logistic(alpha_vals + a_model_vals[:, j])
        model_rows.append(
            {
                "Model": model_name,
                **summarize_scalar(
                    p_model,
                    name=f"p_{reason_code}_model_{model_name}",
                ),
            }
        )
    pd.DataFrame(model_rows).to_csv(
        outdir / f"summary_model_{reason_code}.csv", index=False
    )

    # Pairwise model differences
    pair_rows = []
    for i in range(n_model):
        for j in range(i + 1, n_model):
            m1 = levels["model"][i]
            m2 = levels["model"][j]
            p1 = logistic(alpha_vals + a_model_vals[:, i])
            p2 = logistic(alpha_vals + a_model_vals[:, j])
            diff = p2 - p1
            stats = summarize_scalar(diff, name=f"p_{m2} - p_{m1}")
            stats.update(
                {
                    "Model 1": m1,
                    "Model 2": m2,
                    "Prob > 0": float((diff > 0).mean()),
                }
            )
            pair_rows.append(stats)

    pd.DataFrame(pair_rows)[
        ["Model 1", "Model 2", "Mean", "SD", "HDI 2.5%", "HDI 97.5%", "Prob > 0"]
    ].to_csv(
        outdir / f"summary_pairwise_model_{reason_code}.csv",
        index=False,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 analysis of explanation reasons (Appendix II)."
    )
    parser.add_argument(
        "--infile", required=True,
        help="Path to scored data file (e.g., exp_full_LLM_scored.xlsx)",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for pt2 results.",
    )
    parser.add_argument(
        "--draws", type=int, default=2000,
        help="Number of posterior draws per chain.",
    )
    parser.add_argument(
        "--tune", type=int, default=2000,
        help="Number of tuning iterations per chain.",
    )
    parser.add_argument(
        "--target_accept", type=float, default=0.97,
        help="Target accept rate for NUTS.",
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    infile = Path(args.infile)
    outdir = Path(args.outdir)

    # Load data
    if infile.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(infile)
    elif infile.suffix.lower() == ".csv":
        df = pd.read_csv(infile)
    else:
        raise ValueError(
            f"Unsupported file extension: {infile.suffix}. Use .xlsx, .xls, or .csv."
        )

    df = prepare_data(df)
    idx, levels = build_index_dicts(df)

    # --------------------------------------------------------
    # Content reasons: 4D composition (fact, tone_reason, style, emotion)
    # --------------------------------------------------------
    CONTENT_CODES = ["fact", "tone_reason", "style", "emotion"]

    missing_cols = [c for c in CONTENT_CODES if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing expected content reason columns: {missing_cols}"
        )

    Y = df[CONTENT_CODES].to_numpy().astype(float)
    # Normalize rows to sum to 1 (safety)
    Y = np.clip(Y, 1e-9, 1.0)
    Y = Y / Y.sum(axis=1, keepdims=True)

    content_dir = outdir / "content_reasons"
    print("Fitting logistic-normal composition model for content reasons.")
    fit_content_composition_model(
        Y=Y,
        idx=idx,
        levels=levels,
        outdir=content_dir,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    # --------------------------------------------------------
    # Stereotype reasons: mutually exclusive 0/1 (stereo, avoid_stereo, other)
    # --------------------------------------------------------
    STEREO_CODES = ["stereo", "avoid_stereo", "other"]

    stereo_dir = outdir / "stereotype_reasons"
    for code in STEREO_CODES:
        if code not in df.columns:
            print(f"[WARN] Column '{code}' not found in data; skipping.")
            continue
        y = df[code].astype(int).to_numpy()
        print(f"Fitting Bernoulli model for stereotype reason: {code}, n={y.size}")
        fit_binary_reason_model(
            y=y,
            idx=idx,
            levels=levels,
            outdir=stereo_dir / code,
            reason_code=code,
            draws=args.draws,
            tune=args.tune,
            target_accept=args.target_accept,
            random_seed=args.seed,
        )

    # Optional: mentions_stereotype as an additional binary reason
    if "mentions_stereotype" in df.columns:
        code = "mentions_stereotype"
        y = df[code].astype(int).to_numpy()
        print(f"Fitting Bernoulli model for reason: {code}, n={y.size}")
        fit_binary_reason_model(
            y=y,
            idx=idx,
            levels=levels,
            outdir=stereo_dir / code,
            reason_code=code,
            draws=args.draws,
            tune=args.tune,
            target_accept=args.target_accept,
            random_seed=args.seed,
        )

    print("[INFO] analysis_pt2 completed.")


if __name__ == "__main__":
    main()
