"""
Phase 4 — cleaning.py
Normalize CICFlowMeter CSV output so it matches the training feature spec exactly.
Without this, predictions are silently wrong.

Steps:
  1. Strip leading/trailing spaces from column names
  2. Replace Infinity / NaN with 0
  3. Assert all expected features are present
  4. Return DataFrame aligned to the saved feature order
"""

import json
import numpy as np
import pandas as pd


def clean_cicflow_output(df: pd.DataFrame, features_path: str) -> pd.DataFrame:
    """
    Args:
      df            — raw CICFlowMeter output DataFrame
      features_path — path to artifacts/features.json

    Returns:
      DataFrame with exactly the columns in features.json, in order,
      with Infinity/NaN replaced by 0.

    Raises:
      ValueError if any expected feature column is missing.
    """
    # 1. normalize column names to match training data
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 2. kill Infinity / NaN — model was trained on cleaned data
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3. load expected feature list
    with open(features_path) as f:
        expected: list[str] = json.load(f)

    # 4. fail loudly if columns are missing — silent misalignment is worse
    missing = [col for col in expected if col not in df.columns]
    if missing:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"CICFlowMeter output is missing {len(missing)} expected feature(s):\n"
            f"  {missing}\n"
            f"Available columns: {available}\n"
            "Check that your CICFlowMeter version matches the training data."
        )

    return df[expected]
