"""
Pipeline inference layer — loads artifacts once and exposes predict_flows().

Alert tiers mirror the production extension notebook:
  HIGH   attack_score >= 0.85 — escalate immediately
  MEDIUM attack_score >= 0.50 — flag for review
  LOW    attack_score <  0.50 — routine logging
"""

import os
import sys
import json
import functools
import numpy as np
import pandas as pd
import joblib

from pipeline.custom_transformers import CappedSMOTE

BASE = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(BASE, "model.pkl")
FEATURES_PATH = os.path.join(BASE, "features.json")
LABELS_PATH = os.path.join(BASE, "label_map.json")


@functools.lru_cache(maxsize=1)
def _load_artifacts():
    """Load and cache artifacts so disk I/O only happens once per process."""
    # model.pkl was pickled from a Jupyter notebook, where CappedSMOTE's
    # __module__ was recorded as '__main__'. Whatever process unpickles this
    # (uvicorn's --reload subprocess, a plain script, etc.) needs its own
    # __main__ to expose that name, or joblib.load() raises AttributeError.
    sys.modules["__main__"].CappedSMOTE = CappedSMOTE
    model = joblib.load(MODEL_PATH)
  # Feature order matters: the model expects columns in exactly this order.
    with open(FEATURES_PATH) as f:
        features: list[str] = json.load(f)
  # JSON object keys are always strings, so convert them back to integers
    # to match the encoded class codes the model predicts.
    with open(LABELS_PATH) as f:
        label_map: dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    return model, features, label_map


def predict_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
      df— DataFrame with the 20 model features (already cleaned + aligned).

    Returns:
      df with four new columns appended:
        attack_score  float  P(not benign)
        pred_label str predicted attack class name
        alert_tier str HIGH / MEDIUM / LOW
        action str recommended SOC action
    """
    model, features, label_map = _load_artifacts() #we load the artifcats from the function above

    X = df[features].values #Select the features in the order the model expects.
    proba = model.predict_proba(X) # Class probabilities for each flow: shape (n_flows, n_classes).

     # Overall attack likelihood. Assumes BENIGN is the first column of
    # model.classes_ (encoded as 0).
    attack = 1.0 - proba[:, 0] # P(attack) = 1 - P(benign)

   # Index of the single most probable class for each flow. Assumes class
    # codes are contiguous 0..n-1, so the column index equals the class code.
    cls = np.argmax(proba, axis=1)

    # Work on a copy so the caller's DataFrame isn't mutated.
    out = df.copy()
    out["attack_score"] = attack
    out["pred_label"] = [label_map[c] for c in cls]   # code -> class name
    out["alert_tier"] = [_tier(s) for s in attack]    # score -> HIGH/MEDIUM/LOW
    out["action"] = [_action(s) for s in attack]      # score -> SOC action
    return out

#returns our tiers
def _tier(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    return "LOW"

#How should we react based on the tier level
def _action(score: float) -> str:
    if score >= 0.85:
        return "ALERT - Escalate immediately"
    elif score >= 0.50:
        return "MONITOR - Flag for review"
    return "LOG - Routine logging"
