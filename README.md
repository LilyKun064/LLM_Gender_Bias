# 📘 Bayesian Hierarchical Analysis of LLM Gender Bias

This repository contains the full Bayesian analysis pipeline for
quantifying **gender--pronoun bias** in large language models (LLMs).\
The analysis evaluates how different LLMs assign *he/she* pronouns
across multiple scenarios, writing tones, and scenario-specific factors.

This README provides:

-   A complete description of the experiment\
-   Full workflow for running the analysis\
-   Explanation of output files\
-   Statistical modeling details\
-   Directory structure guidelines\
-   Reproducibility notes

------------------------------------------------------------------------

# 🧠 1. Project Motivation

Large language models often infer the gender of an imaginary person when
generating text.\
This project measures:

-   How often models select **she** vs **he**\
-   How this varies by **scenario** (cover letter, potluck, travel)\
-   How writing **tone** affects judgment\
-   How **occupations**, **foods**, or **hobbies** affect inference\
-   How **different LLMs** compare (GPT-4.1-mini, GPT-4o-mini, Gemini,
    DeepSeek)

To avoid statistical issues such as *complete separation* in logistic
regression, the analysis uses a **Bayesian hierarchical logistic model**
implemented in **PyMC**.

------------------------------------------------------------------------

# 🧩 2. Repository Structure

    E:/LLM_Gender_Bias/
    │
    ├── code/
    │   ├── analysis_pt1.py          # Main hierarchical Bayesian analysis
    │   ├── analysis_pt2.py          # Token-level linguistic analysis
    │   ├── full_experiment.py       # Script to run all LLM calls
    │   └── utils/                   # Helper functions
    │
    ├── data/
    │   ├── raw/                     # Unprocessed CSV from experiment
    │   └── processed/               # Cleaned or subset data
    │
    ├── results/
    │   ├── bayes_hier_full/         # Bayesian model outputs
    │   ├── reasons_full_compare/    # Explanation/token stats
    │   └── figures/                 # Plots for the manuscript
    │
    └── README.md

------------------------------------------------------------------------

# 🚀 3. Running the Bayesian Hierarchical Analysis

## Step 1: Activate your virtual environment

``` bash
.\.venv\Scripts\activate        # Windows
source .venv/bin/activate      # macOS/Linux
```

Confirm:

    (.venv) E:\LLM_Gender_Bias>

------------------------------------------------------------------------

## Step 2: Run the analysis script

Basic command:

``` bash
python code/analysis_pt1.py --csv <input.csv> --outdir <output_directory>
```

### Example:

``` bash
python code/analysis_pt1.py ^
    --csv data/raw/exp_full_compare_models_20251120_074620.csv ^
    --outdir results/bayes_hier_full_test
```

------------------------------------------------------------------------

## Step 3: Interactive prompt (skip or refit)

If output directory already contains:

    trace_hierarchical.nc

you will see:

    Found existing posterior trace.
    Do you want to SKIP re-fitting and load this trace? (y/n):

-   **y** → Load existing posterior (FAST)\
-   **n** → Re-run full Bayesian sampling (SLOW: 10--30 min)

------------------------------------------------------------------------

# 📊 4. Output Files Explained

After running, the output directory (`--outdir`) will contain:

------------------------------------------------------------------------

## 4.1 Posterior Trace

### `trace_hierarchical.nc`

A complete ArviZ InferenceData object storing:

-   Posterior draws\
-   Log-likelihood\
-   Metadata and dimensions

This file enables reproducibility without re-running MCMC.

------------------------------------------------------------------------

## 4.2 Global Hyperparameters

### `summary_global_and_sigma.csv`

Contains posterior summaries of:

-   Global intercept `alpha`
-   σ (SD) of each random effect:
    -   `sigma_model`
    -   `sigma_scenario`
    -   `sigma_tone`
    -   `sigma_factor`
    -   `sigma_model_scenario`

These describe the *variability* in gender bias across models,
scenarios, tones, etc.

------------------------------------------------------------------------

## 4.3 Model-Level Bias

### `summary_model_baseline_p_she.csv`

Baseline probability that each model assigns **she**, holding other
factors at zero-centered levels.

### `summary_model_pairwise_diff_p_she.csv`

Pairwise comparisons:

    p(she | model2) – p(she | model1)

This answers: \> Which model is MOST gender-biased?

------------------------------------------------------------------------

## 4.4 Scenario-Level Bias

### `summary_scenario_baseline_p_she.csv`

Captures how gendered a scenario is (e.g., potluck vs cover letter)
regardless of model.

------------------------------------------------------------------------

## 4.5 Tone-Level Bias

### `summary_tone_baseline_p_she.csv`

Effect of **polite** vs **direct** writing tone on gender inference.

------------------------------------------------------------------------

## 4.6 Factor-Level Bias

### `summary_factor_baseline_p_she.csv`

Factor examples:

-   Occupation (cover letter)
-   Food choice (potluck)
-   Hobby/interest (travel)

Reveals which traits lead models to infer a feminine or masculine
persona.

------------------------------------------------------------------------

## 4.7 Cell-Level Predictions (Full Model)

### `cell_probs_she_hierarchical.csv`

Posterior P(she) for each unique combination:

    scenario × model × tone × occupation/food/hobby

### `cell_model_differences_p_she.csv`

For each scenario cell:

    p(she | model2) – p(she | model1)

This is the highest-resolution measure of model differences.

------------------------------------------------------------------------

# 🧮 5. Statistical Model Specification

The hierarchical logistic model:

    logit(p_i) =
          α
        + a_model[model_i]
        + a_scenario[scenario_i]
        + a_tone[tone_i]
        + a_factor[factor_i]
        + a_model_scenario[model_i, scenario_i]

Priors:

    alpha ~ Normal(0, 2)

    a_model[m]            ~ Normal(0, sigma_model)
    a_scenario[s]         ~ Normal(0, sigma_scenario)
    a_tone[t]             ~ Normal(0, sigma_tone)
    a_factor[f]           ~ Normal(0, sigma_factor)
    a_model_scenario[m,s] ~ Normal(0, sigma_model_scenario)

    sigma_* ~ Exponential(1)

This structure:

-   avoids classical logistic separation\
-   partially pools effects\
-   shares information across models and conditions\
-   provides uncertainty estimates for all effects

------------------------------------------------------------------------

# 🔁 6. Typical Workflow

### 1. **Run full experiment**

Produce a raw CSV under:

    data/raw/

### 2. **Run Bayesian analysis**

Produces `.nc` and summary CSVs.

### 3. **Inspect model-level differences**

Look at:

-   `summary_model_baseline_p_she.csv`
-   `model_pairwise_diff_p_she.csv`

### 4. **Generate plots**

Figures will be created later using these output CSVs.

### 5. **Integrate into manuscript**

Use results in:

-   Methods (model specification)
-   Results (model/scenario/tone-level findings)
-   Appendix (full tables)

------------------------------------------------------------------------

# 🔬 7. Reproducibility Notes

-   All sampling is done using **PyMC** (NUTS sampler).\
-   Seeds are not set; expect slight posterior variation.\
-   The `.nc` file ensures exact reproducibility.\
-   No GPU is required; PyMC runs on CPU only (recommended for Windows).

------------------------------------------------------------------------

# 📝 8. Suggested Citation

You may cite this analysis as:

> Jiang (2025). *Bayesian hierarchical logistic analysis of gender bias
> in large language models.*
>
> Code available at: `E:/LLM_Gender_Bias/code/analysis_pt1.py`.

------------------------------------------------------------------------

# 📞 9. Contact

For questions, code review, or manuscript integration help, contact:

**CJ (Chenkun Jiang)**\
LLM Gender Bias Research Project

------------------------------------------------------------------------

# ✅ End of README
