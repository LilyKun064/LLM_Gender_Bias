#!/usr/bin/env python

"""
Bayesian hierarchical logistic analysis for ALL scenarios and ALL models
in the LLM gender-pronoun experiment.

Run with something like:
    python code/analysis_pt1.py \
        --csv data/raw/exp_full_compare_models_20251120_074620.csv \
        --outdir results/bayes_hier_full

Design / structure
------------------
Data:
  - scenario ∈ {cover_letter, potluck, travel}
  - model    ∈ {gpt-4.1-mini, gpt-4o-mini, gemini, deepseek, ...}
  - tone     ∈ {direct, polite} (lowercased and cleaned)
  - factor:
        cover_letter → occupation
        potluck      → food
        travel       → hobby_profile
  - response_pronoun (free-text Stage 2 pronoun description)

Outcome:
  - y = 1 if dominant pronoun is "she"
  - y = 0 if dominant pronoun is "he"
  - Rows with other / unknown pronouns are dropped.

Hierarchical model:
  For each trial i,

      y_i ~ Bernoulli(p_i)
      logit(p_i) = alpha
                   + a_model[model_i]
                   + a_scenario[scenario_i]
                   + a_tone[tone_i]
                   + a_factor[factor_i]
                   + a_model_scenario[model_i, scenario_i]

  with:

      a_model[m]            ~ Normal(0, sigma_model)
      a_scenario[s]         ~ Normal(0, sigma_scenario)
      a_tone[t]             ~ Normal(0, sigma_tone)
      a_factor[k]           ~ Normal(0, sigma_factor)
      a_model_scenario[m,s] ~ Normal(0, sigma_model_scenario)

      alpha ~ Normal(0, 2)
      sigma_* ~ Exponential(1)

Here, "factor" is the scenario-specific factor (occupation, food, hobby_profile)
encoded as a single set of levels (scenario::value). This is nested within
scenario, but we also include separate a_scenario to capture scenario-level bias.

Outputs
-------
All outputs written into --outdir:

1. trace_hierarchical.nc
   - Full posterior (InferenceData) for later inspection.

2. summary_global_and_sigma.csv
   - Posterior summaries for alpha and all sigma_* hyperparameters.

3. summary_model_baseline_p_she.csv
   - Posterior summaries of baseline P(she) per model, using:
       logit(p) = alpha + a_model[m]
     (other random effects set to 0).

4. summary_model_pairwise_diff_p_she.csv
   - Pairwise differences in baseline P(she) between models:
       p(model2) - p(model1).

5. summary_scenario_baseline_p_she.csv
   - Scenario-level baseline P(she):
       logit(p) = alpha + a_scenario[s].

6. summary_tone_baseline_p_she.csv
   - Tone-level baseline P(she):
       logit(p) = alpha + a_tone[t].

7. summary_factor_baseline_p_she.csv
   - Factor-level baseline P(she) for each scenario-specific factor level,
     using:
       logit(p) = alpha + a_scenario[scenario(f)] + a_factor[f].

8. cell_probs_she_hierarchical.csv
   - Posterior P(she) for every observed cell:
       (scenario, model, factor_value, tone),
     using:
       logit(p) = alpha
                  + a_model[m]
                  + a_scenario[s]
                  + a_tone[t]
                  + a_factor[f]
                  + a_model_scenario[m, s].

9. cell_model_differences_p_she.csv
   - For each (scenario, factor_value, tone) combination, pairwise differences
     in P(she) across models:
       p(model2) - p(model1).

These give you bias differences at all the levels you care about.
"""

from pathlib import Path
import argparse

import pytensor

# Disable C compilation and use a pure-Python mode to avoid linker issues on Windows
pytensor.config.cxx = ""            # turn off C compiler
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.exception_verbosity = "high"  # nicer tracebacks

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm


# ---------- CONFIG: scenario → factor column ----------

SCENARIO_CONFIG = {
    "cover_letter": {
        "factor_col": "occupation",
        "pretty_name": "Occupation",
    },
    "potluck": {
        "factor_col": "food",
        "pretty_name": "Food",
    },
    "travel": {
        "factor_col": "hobby_profile",
        "pretty_name": "Hobby profile",
    },
}


# ---------- Utilities ----------

def detect_pronoun(text: str) -> str:
    """
    Crude heuristic to classify the dominant third-person pronoun
    in a short description (here: response_pronoun).

    Returns: "he", "she", "they", "mixed", or "unknown".
    """
    if not isinstance(text, str):
        return "unknown"

    t = " " + text.lower().strip() + " "

    he_hits = any(tok in t for tok in [" he ", " him ", " his "])
    she_hits = any(tok in t for tok in [" she ", " her ", " hers "])
    they_hits = any(tok in t for tok in [" they ", " them ", " their ", " theirs "])

    if he_hits and not she_hits and not they_hits:
        return "he"
    if she_hits and not he_hits and not they_hits:
        return "she"
    if they_hits and not he_hits and not she_hits:
        return "they"
    if sum([he_hits, she_hits, they_hits]) > 1:
        return "mixed"
    return "unknown"


def summarize(samples: np.ndarray) -> dict:
    """Return summary stats for a 1D array of posterior samples."""
    samples = np.asarray(samples)
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "hdi_2.5%": float(np.quantile(samples, 0.025)),
        "hdi_97.5%": float(np.quantile(samples, 0.975)),
        "prob>0": float(np.mean(samples > 0)),
    }


def inv_logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Data preparation ----------

def prepare_modeling_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw CSV and construct a modeling DataFrame with:

        scenario, model, tone, factor_value, factor_label,
        scenario_idx, model_idx, tone_idx, factor_idx, y

    where y = 1 if pronoun=='she', 0 if pronoun=='he'.
    """
    df = df_raw.copy()

    if "scenario" not in df.columns:
        raise ValueError("CSV must contain a 'scenario' column.")
    if "model" not in df.columns:
        raise ValueError("CSV must contain a 'model' column.")
    if "tone" not in df.columns:
        raise ValueError("CSV must contain a 'tone' column.")
    if "response_pronoun" not in df.columns:
        raise ValueError("CSV must contain a 'response_pronoun' column.")

    # Normalize strings
    df["scenario"] = df["scenario"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip()
    df["tone"] = df["tone"].astype(str).str.strip().str.lower()

    # Keep only scenarios we know how to handle
    valid_scenarios = set(SCENARIO_CONFIG.keys())
    df = df[df["scenario"].isin(valid_scenarios)].copy()

    # Detect pronoun
    print("Detecting pronouns from response_pronoun...")
    df["pronoun"] = df["response_pronoun"].apply(detect_pronoun)

    print("Pronoun value counts (raw):")
    print(df["pronoun"].value_counts(dropna=False))

    # Keep only he/she
    df = df[df["pronoun"].isin(["he", "she"])].copy()
    if df.empty:
        raise ValueError("No rows with 'he' or 'she' pronouns after filtering.")

    df["y"] = (df["pronoun"] == "she").astype(int)

    records = []
    for _, row in df.iterrows():
        scenario = row["scenario"]
        cfg = SCENARIO_CONFIG.get(scenario)
        if cfg is None:
            continue

        factor_col = cfg["factor_col"]
        if factor_col not in df.columns:
            raise ValueError(
                f"Configured factor_col '{factor_col}' for scenario '{scenario}' "
                f"is not a column in the CSV."
            )

        factor_val = row[factor_col]
        if pd.isna(factor_val):
            continue

        factor_val = str(factor_val).strip()
        tone = row["tone"]
        model_name = row["model"]
        y = int(row["y"])

        factor_label = f"{scenario}::{factor_val}"

        records.append(
            dict(
                scenario=scenario,
                model=model_name,
                tone=tone,
                factor_value=factor_val,
                factor_label=factor_label,
                y=y,
            )
        )

    modeling_df = pd.DataFrame.from_records(records)
    if modeling_df.empty:
        raise ValueError("No usable rows after filtering for factor/tone/model/pronoun.")

    # Build integer indices for hierarchical model
    scenarios = sorted(modeling_df["scenario"].unique())
    models = sorted(modeling_df["model"].unique())
    tones = sorted(modeling_df["tone"].unique())
    factors = sorted(modeling_df["factor_label"].unique())

    scenario_index = {s: i for i, s in enumerate(scenarios)}
    model_index = {m: i for i, m in enumerate(models)}
    tone_index = {t: i for i, t in enumerate(tones)}
    factor_index = {f: i for i, f in enumerate(factors)}

    modeling_df["scenario_idx"] = modeling_df["scenario"].map(scenario_index).astype("int64")
    modeling_df["model_idx"] = modeling_df["model"].map(model_index).astype("int64")
    modeling_df["tone_idx"] = modeling_df["tone"].map(tone_index).astype("int64")
    modeling_df["factor_idx"] = modeling_df["factor_label"].map(factor_index).astype("int64")

    print("\nFinal modeling data size:", len(modeling_df))
    print("Scenarios:", scenarios)
    print("Models:", models)
    print("Tones:", tones)
    print("Number of factor levels:", len(factors))

    return modeling_df


# ---------- Hierarchical model fitting ----------

def fit_hierarchical_model(df_modeling: pd.DataFrame, outdir: Path) -> az.InferenceData:
    """
    Fit the Bayesian hierarchical logistic model and save the trace.

    Returns:
        idata (InferenceData)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Prepare indices and coordinates
    scenarios = sorted(df_modeling["scenario"].unique())
    models = sorted(df_modeling["model"].unique())
    tones = sorted(df_modeling["tone"].unique())
    factors = sorted(df_modeling["factor_label"].unique())

    # Map factor_label → scenario_idx for factor-level summaries later (stored as metadata)
    factor_to_scenario_idx = {}
    scenario_index = {s: i for i, s in enumerate(scenarios)}
    for f_label in factors:
        scenario_name = f_label.split("::", 1)[0]
        factor_to_scenario_idx[f_label] = scenario_index[scenario_name]

    factor_scenario_idx = np.array(
        [factor_to_scenario_idx[f] for f in factors],
        dtype="int64",
    )

    y = df_modeling["y"].values.astype("int64")
    model_idx = df_modeling["model_idx"].values.astype("int64")
    scenario_idx_arr = df_modeling["scenario_idx"].values.astype("int64")
    tone_idx = df_modeling["tone_idx"].values.astype("int64")
    factor_idx = df_modeling["factor_idx"].values.astype("int64")

    coords = {
        "model": models,
        "scenario": scenarios,
        "tone": tones,
        "factor": factors,
    }

    with pm.Model(coords=coords) as model:
        # Data
        y_data = pm.Data("y_data", y)
        model_id = pm.Data("model_id", model_idx)
        scenario_id = pm.Data("scenario_id", scenario_idx_arr)
        tone_id = pm.Data("tone_id", tone_idx)
        factor_id = pm.Data("factor_id", factor_idx)

        # Hyperpriors
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)

        sigma_model = pm.Exponential("sigma_model", 1.0)
        sigma_scenario = pm.Exponential("sigma_scenario", 1.0)
        sigma_tone = pm.Exponential("sigma_tone", 1.0)
        sigma_factor = pm.Exponential("sigma_factor", 1.0)
        sigma_model_scenario = pm.Exponential("sigma_model_scenario", 1.0)

        # Group-level effects
        a_model = pm.Normal("a_model", mu=0.0, sigma=sigma_model, dims="model")
        a_scenario = pm.Normal("a_scenario", mu=0.0, sigma=sigma_scenario, dims="scenario")
        a_tone = pm.Normal("a_tone", mu=0.0, sigma=sigma_tone, dims="tone")
        a_factor = pm.Normal("a_factor", mu=0.0, sigma=sigma_factor, dims="factor")
        a_model_scenario = pm.Normal(
            "a_model_scenario",
            mu=0.0,
            sigma=sigma_model_scenario,
            dims=("model", "scenario"),
        )

        # Linear predictor
        eta = (
            alpha
            + a_model[model_id]
            + a_scenario[scenario_id]
            + a_tone[tone_id]
            + a_factor[factor_id]
            + a_model_scenario[model_id, scenario_id]
        )

        p = pm.Deterministic("p", pm.math.sigmoid(eta))

        # Likelihood
        pm.Bernoulli("y_obs", p=p, observed=y_data)

        print("\nSampling hierarchical model...")
        idata = pm.sample(
            draws=2000,
            tune=2000,
            target_accept=0.9,
            chains=4,
            cores=4,  # safer on Windows
            init="jitter+adapt_diag",
            return_inferencedata=True,
        )

    trace_path = outdir / "trace_hierarchical.nc"
    print(f"\nSaving posterior trace to: {trace_path}")
    az.to_netcdf(idata, trace_path)

    # Attach some metadata to idata for later use (not required by summaries)
    idata._log_likelihood = None  # just to be safe if we want to add later
    idata.attrs["factor_scenario_idx"] = factor_scenario_idx.tolist()

    return idata


# ---------- Posterior summarization helpers ----------

def summarize_global_and_sigmas(idata: az.InferenceData, outdir: Path):
    """
    Summarize alpha and all sigma_* hyperparameters.
    """
    post = idata.posterior.stack(sample=("chain", "draw"))

    alpha_samples = post["alpha"].values  # (S,)

    sigmas = {
        "sigma_model": post["sigma_model"].values,
        "sigma_scenario": post["sigma_scenario"].values,
        "sigma_tone": post["sigma_tone"].values,
        "sigma_factor": post["sigma_factor"].values,
        "sigma_model_scenario": post["sigma_model_scenario"].values,
    }

    rows = []
    rows.append({"parameter": "alpha", **summarize(alpha_samples)})
    for name, arr in sigmas.items():
        rows.append({"parameter": name, **summarize(arr)})

    df = pd.DataFrame(rows)
    outpath = outdir / "summary_global_and_sigma.csv"
    print(f"Saving global and sigma summaries to: {outpath}")
    df.to_csv(outpath, index=False)


def summarize_model_level(idata: az.InferenceData, df_modeling: pd.DataFrame, outdir: Path):
    """
    Baseline model-level P(she):
        logit(p) = alpha + a_model[m]
    (all other random effects set to 0).
    """
    models = sorted(df_modeling["model"].unique())

    post = idata.posterior.stack(sample=("chain", "draw"))
    alpha_samples = post["alpha"].values  # (S,)

    # Ensure dimensions: (sample, model)
    a_model_samples = post["a_model"].transpose("sample", "model").values  # (S, n_models)

    # logit(p) = alpha + a_model[m]
    logit_model = alpha_samples[:, None] + a_model_samples  # (S, n_models)
    p_model = inv_logit(logit_model)  # (S, n_models)

    # Baseline P(she) per model
    rows = []
    for j, m in enumerate(models):
        rows.append({"model": m, **summarize(p_model[:, j])})

    df_model = pd.DataFrame(rows)
    outpath = outdir / "summary_model_baseline_p_she.csv"
    print(f"Saving model-level baseline P(she) to: {outpath}")
    df_model.to_csv(outpath, index=False)

    # Pairwise differences: p(model2) - p(model1)
    diff_rows = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            diff = p_model[:, j] - p_model[:, i]
            diff_rows.append(
                {
                    "model1": m1,
                    "model2": m2,
                    **summarize(diff),
                }
            )
    df_diff = pd.DataFrame(diff_rows)
    outpath2 = outdir / "summary_model_pairwise_diff_p_she.csv"
    print(f"Saving model-level pairwise differences in P(she) to: {outpath2}")
    df_diff.to_csv(outpath2, index=False)


def summarize_scenario_level(idata: az.InferenceData, df_modeling: pd.DataFrame, outdir: Path):
    """
    Scenario-level baseline P(she):
        logit(p) = alpha + a_scenario[s]
    """
    scenarios = sorted(df_modeling["scenario"].unique())

    post = idata.posterior.stack(sample=("chain", "draw"))
    alpha_samples = post["alpha"].values  # (S,)
    a_scenario_samples = post["a_scenario"].transpose("sample", "scenario").values  # (S, n_scenarios)

    logit_scenario = alpha_samples[:, None] + a_scenario_samples  # (S, n_scenarios)
    p_scenario = inv_logit(logit_scenario)

    rows = []
    for j, s in enumerate(scenarios):
        rows.append({"scenario": s, **summarize(p_scenario[:, j])})

    df = pd.DataFrame(rows)
    outpath = outdir / "summary_scenario_baseline_p_she.csv"
    print(f"Saving scenario-level baseline P(she) to: {outpath}")
    df.to_csv(outpath, index=False)


def summarize_tone_level(idata: az.InferenceData, df_modeling: pd.DataFrame, outdir: Path):
    """
    Tone-level baseline P(she):
        logit(p) = alpha + a_tone[t]
    """
    tones = sorted(df_modeling["tone"].unique())

    post = idata.posterior.stack(sample=("chain", "draw"))
    alpha_samples = post["alpha"].values  # (S,)
    a_tone_samples = post["a_tone"].transpose("sample", "tone").values  # (S, n_tones)

    logit_tone = alpha_samples[:, None] + a_tone_samples  # (S, n_tones)
    p_tone = inv_logit(logit_tone)

    rows = []
    for j, t in enumerate(tones):
        rows.append({"tone": t, **summarize(p_tone[:, j])})

    df = pd.DataFrame(rows)
    outpath = outdir / "summary_tone_baseline_p_she.csv"
    print(f"Saving tone-level baseline P(she) to: {outpath}")
    df.to_csv(outpath, index=False)


def summarize_factor_level(idata: az.InferenceData, df_modeling: pd.DataFrame, outdir: Path):
    """
    Factor-level baseline P(she) for each scenario-specific factor level.

    For each factor f (like "cover_letter::research scientist"), we use:
        logit(p) = alpha + a_scenario[scenario(f)] + a_factor[f]
    """
    scenarios = sorted(df_modeling["scenario"].unique())
    factors = sorted(df_modeling["factor_label"].unique())

    # Map factor → scenario index
    scenario_index = {s: i for i, s in enumerate(scenarios)}
    factor_scenario_idx = np.array(
        [scenario_index[f.split("::", 1)[0]] for f in factors],
        dtype="int64",
    )

    post = idata.posterior.stack(sample=("chain", "draw"))
    alpha_samples = post["alpha"].values  # (S,)
    a_scenario_samples = post["a_scenario"].transpose("sample", "scenario").values  # (S, n_scenarios)
    a_factor_samples = post["a_factor"].transpose("sample", "factor").values  # (S, n_factors)

    # a_scenario for each factor
    a_scenario_for_factor = a_scenario_samples[:, factor_scenario_idx]  # (S, n_factors)

    logit_factor = alpha_samples[:, None] + a_scenario_for_factor + a_factor_samples  # (S, n_factors)
    p_factor = inv_logit(logit_factor)  # (S, n_factors)

    rows = []
    for j, f_label in enumerate(factors):
        scenario_name, factor_val = f_label.split("::", 1)
        rows.append(
            {
                "scenario": scenario_name,
                "factor_value": factor_val,
                "factor_label": f_label,
                **summarize(p_factor[:, j]),
            }
        )

    df = pd.DataFrame(rows)
    outpath = outdir / "summary_factor_baseline_p_she.csv"
    print(f"Saving factor-level baseline P(she) to: {outpath}")
    df.to_csv(outpath, index=False)


# ---------- Cell-level posterior predictions & cross-model differences ----------

def compute_cell_level_predictions(
    idata: az.InferenceData,
    df_modeling: pd.DataFrame,
    outdir: Path,
):
    """
    For every *observed* cell (scenario, model, factor_value, tone),
    compute posterior P(she) using the full linear predictor:

        logit(p) = alpha
                   + a_model[m]
                   + a_scenario[s]
                   + a_tone[t]
                   + a_factor[f]
                   + a_model_scenario[m, s]

    Then, for each (scenario, factor_value, tone), compute pairwise
    differences in P(she) across models.
    """
    scenarios = sorted(df_modeling["scenario"].unique())
    models = sorted(df_modeling["model"].unique())
    tones = sorted(df_modeling["tone"].unique())
    factors = sorted(df_modeling["factor_label"].unique())

    post = idata.posterior.stack(sample=("chain", "draw"))

    alpha_samples = post["alpha"].values  # (S,)
    a_model_samples = post["a_model"].transpose("sample", "model").values            # (S, n_models)
    a_scenario_samples = post["a_scenario"].transpose("sample", "scenario").values  # (S, n_scenarios)
    a_tone_samples = post["a_tone"].transpose("sample", "tone").values              # (S, n_tones)
    a_factor_samples = post["a_factor"].transpose("sample", "factor").values        # (S, n_factors)
    a_model_scenario_samples = post["a_model_scenario"].transpose(
        "sample", "model", "scenario"
    ).values  # (S, n_models, n_scenarios)

    # Integer index mapping
    scenario_index = {s: i for i, s in enumerate(scenarios)}
    model_index = {m: i for i, m in enumerate(models)}
    tone_index = {t: i for i, t in enumerate(tones)}
    factor_index = {f: i for i, f in enumerate(factors)}

    # Observed unique cells
    cell_df = (
        df_modeling[
            ["scenario", "model", "tone", "factor_value", "factor_label"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Compute posterior P(she) for each cell
    rows = []
    p_samples_by_cell = []  # keep samples if we want differences later

    for _, row in cell_df.iterrows():
        s_name = row["scenario"]
        m_name = row["model"]
        t_name = row["tone"]
        f_label = row["factor_label"]

        s = scenario_index[s_name]
        m = model_index[m_name]
        t = tone_index[t_name]
        f = factor_index[f_label]

        # logit(p) = alpha + a_model[m] + a_scenario[s] + a_tone[t]
        #            + a_factor[f] + a_model_scenario[m, s]
        logit = (
            alpha_samples
            + a_model_samples[:, m]
            + a_scenario_samples[:, s]
            + a_tone_samples[:, t]
            + a_factor_samples[:, f]
            + a_model_scenario_samples[:, m, s]
        )
        p = inv_logit(logit)  # (S,)

        p_samples_by_cell.append(p)
        rows.append(
            {
                "scenario": s_name,
                "model": m_name,
                "tone": t_name,
                "factor_value": row["factor_value"],
                "factor_label": f_label,
                **summarize(p),
            }
        )

    cell_summary = pd.DataFrame(rows)
    outpath = outdir / "cell_probs_she_hierarchical.csv"
    print(f"Saving cell-level P(she) summaries to: {outpath}")
    cell_summary.to_csv(outpath, index=False)

    # Cross-model differences within each (scenario, factor_value, tone)
    diff_rows = []
    # Attach p_samples temporarily for each row index
    cell_df["p_samples_index"] = np.arange(len(cell_df))

    for (scenario_name, factor_val, tone_name), sub in cell_df.groupby(
        ["scenario", "factor_value", "tone"]
    ):
        # For this scenario × factor × tone, gather models and their p_samples
        sub = sub.sort_values("model")
        models_here = list(sub["model"])
        idxs = list(sub["p_samples_index"])

        p_list = [p_samples_by_cell[i] for i in idxs]  # each is (S,)

        for i, m1 in enumerate(models_here):
            for j, m2 in enumerate(models_here):
                if j <= i:
                    continue
                diff = p_list[j] - p_list[i]
                diff_rows.append(
                    {
                        "scenario": scenario_name,
                        "factor_value": factor_val,
                        "tone": tone_name,
                        "model1": m1,
                        "model2": m2,
                        **summarize(diff),
                    }
                )

    df_diffs = pd.DataFrame(diff_rows)
    outpath2 = outdir / "cell_model_differences_p_she.csv"
    print(f"Saving cell-level model differences in P(she) to: {outpath2}")
    df_diffs.to_csv(outpath2, index=False)


# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian hierarchical logistic analysis of pronoun bias across "
            "cover_letter, potluck, and travel scenarios and multiple models."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/exp_full_compare_models.csv",
        help="Path to the full experiment CSV (with 'scenario', 'model', 'tone', 'response_pronoun').",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/bayes_hier_full",
        help="Directory for hierarchical-model outputs.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print("        analysis_pt1: Hierarchical Bayes")
    print("==============================================")
    print(f"Input CSV: {csv_path.resolve()}")
    print(f"Output directory: {outdir.resolve()}")
    print("")

    print(f"Reading data from: {csv_path.resolve()}")
    df_raw = pd.read_csv(csv_path)

    # Prepare modeling dataframe
    print("Preparing modeling dataframe...")
    df_modeling = prepare_modeling_df(df_raw)
    print(f"Total rows after cleaning/filtering: {len(df_modeling)}\n")

    # Posterior trace file
    trace_path = outdir / "trace_hierarchical.nc"

    # Step 1: Fit model OR load trace (ask once)
    if trace_path.exists():
        print(f"Found existing posterior trace at: {trace_path.resolve()}")
        while True:
            choice = input(
                "Do you want to SKIP re-fitting and load this trace? (y/n): "
            ).strip().lower()
            if choice in ("y", "n"):
                break
            print("Please answer 'y' or 'n'.")
        if choice == "y":
            print("Loading existing posterior trace...\n")
            idata = az.from_netcdf(trace_path)
        else:
            print("Re-fitting hierarchical model from scratch...\n")
            idata = fit_hierarchical_model(df_modeling, outdir)
    else:
        print("No existing posterior trace found. Fitting hierarchical model...\n")
        idata = fit_hierarchical_model(df_modeling, outdir)

    # Step 2: Summaries and cell-level predictions
    print("\nRunning global and sigma summaries...")
    summarize_global_and_sigmas(idata, outdir)

    print("Running model-level summaries...")
    summarize_model_level(idata, df_modeling, outdir)

    print("Running scenario-level summaries...")
    summarize_scenario_level(idata, df_modeling, outdir)

    print("Running tone-level summaries...")
    summarize_tone_level(idata, df_modeling, outdir)

    print("Running factor-level summaries...")
    summarize_factor_level(idata, df_modeling, outdir)

    print("Computing cell-level posterior predictions and cross-model differences...")
    compute_cell_level_predictions(idata, df_modeling, outdir)

    print("\n==============================================")
    print("        analysis_pt1 COMPLETED")
    print("==============================================")


if __name__ == "__main__":
    main()
