#!/usr/bin/env python
"""
make_human_coding_file.py

Randomly sample explanations for human coding and add binary code columns
for later machine learning.

Codebook columns (multi-label, 0/1):
- fact          : Factual / technical information
- tone          : Tone / communication style
- style         : Writing style / structure
- emotion       : Emotion / personality cues
- stereo        : Social / cultural stereotype
- avoid_stereo  : Counter-stereotype / avoiding stereotype
- other         : Other / miscellaneous
"""

import argparse
import pandas as pd
from pathlib import Path


CODE_COLUMNS = [
    "fact",
    "tone_reason",
    "style",
    "emotion",
    "stereo",
    "avoid_stereo",
    "other",
]


def make_human_coding_file(
    csv_path: str,
    out_path: str,
    n_samples: int = 200,
    seed: int = 123,
) -> None:
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if "response_why" not in df.columns:
        raise ValueError(
            "Expected a column named 'response_why' in the input CSV, "
            f"but found: {list(df.columns)}"
        )

    # Keep only rows with a non-empty explanation
    mask = df["response_why"].notna() & df["response_why"].astype(str).str.strip().ne("")
    df_expl = df.loc[mask].copy()
    n_available = len(df_expl)

    if n_available == 0:
        raise ValueError("No non-empty explanations found in 'response_why'.")

    if n_samples > n_available:
        print(
            f"Requested {n_samples} samples, but only {n_available} explanations "
            "available. Using all available rows instead."
        )
        n_samples = n_available

    # Sample rows for human coding
    sample_df = df_expl.sample(n=n_samples, random_state=seed).copy()
    sample_df.reset_index(drop=True, inplace=True)

    # Add a simple sample ID for convenience
    sample_df.insert(0, "sample_id", range(1, len(sample_df) + 1))

    # Keep only columns that are useful for coding + metadata
    # Adjust this list if you want more/less context.
    keep_cols = [
        "sample_id",
        "model",
        "scenario",
        "occupation",
        "food",
        "hobby_profile",
        "tone",
        "trial",
        "response_pronoun",
        "response_why",
    ]
    keep_cols = [c for c in keep_cols if c in sample_df.columns]
    sample_df = sample_df[keep_cols]

    # Add empty code columns (you will fill with 0/1)
    for col in CODE_COLUMNS:
        sample_df[col] = pd.NA  # or 0 if you prefer to start at 0

    # Optional free-text notes column
    sample_df["coder_notes"] = ""

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing human coding file to: {out_path}")
    sample_df.to_csv(out_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample explanations from the experiment file and create a "
            "human coding CSV with binary code columns."
        )
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the full experiment CSV "
             "(e.g., data/raw/exp_full_compare_models_20251120_074620.csv)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for the output human coding CSV "
             "(e.g., data/human_coding/human_coding_sample_200.csv)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of rows to sample for human coding (default: 200).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling (default: 123).",
    )

    args = parser.parse_args()
    make_human_coding_file(
        csv_path=args.csv,
        out_path=args.out,
        n_samples=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
