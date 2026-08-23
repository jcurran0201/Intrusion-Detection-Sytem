"""
Phase 3 — feature_extraction.py (CICFlowMeter stub)

CICFlowMeter collapses raw packets into the flow-level behavioral statistics
the model was trained on.  This module is the only place in the stack that
calls CICFlowMeter — wire it in once you have the JAR installed.

Install CICFlowMeter:
  1. Install Java 11+: brew install --cask temurin
  2. Download JAR from: https://github.com/ahlashkari/CICFlowMeter/releases
  3. Set env var: export CICFLOWMETER_JAR=/path/to/CICFlowMeter.jar
  4. Optionally set: export CICFLOWMETER_TIMEOUT=120  (seconds, default 120)

Once installed, extract_flows() will work end-to-end.
"""

import os
import subprocess
import tempfile
import glob
import pandas as pd

CICFLOWMETER_JAR     = os.environ.get("CICFLOWMETER_JAR", "")
CICFLOWMETER_TIMEOUT = int(os.environ.get("CICFLOWMETER_TIMEOUT", "120"))


def extract_flows(pcap_path: str) -> pd.DataFrame:
    """
    Run CICFlowMeter on pcap_path, parse the output CSV, return a DataFrame
    of flow-level features (one row per bidirectional flow).

    Raises:
      RuntimeError  — CICFlowMeter JAR not found / not configured
      subprocess.TimeoutExpired — capture took longer than CICFLOWMETER_TIMEOUT
      ValueError    — no flows extracted from the PCAP
    """
    if not CICFLOWMETER_JAR or not os.path.exists(CICFLOWMETER_JAR):
        raise RuntimeError(
            "CICFlowMeter JAR not found.  "
            "Set the CICFLOWMETER_JAR environment variable to the full path of the JAR.  "
            "See pipeline/feature_extraction.py for install instructions."
        )

    out_dir = tempfile.mkdtemp(prefix="cicflow_")
    try:
        cmd = [
            "java", "-jar", CICFLOWMETER_JAR,
            pcap_path,
            out_dir,
        ]
        subprocess.run(
            cmd,
            check=True,
            timeout=CICFLOWMETER_TIMEOUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        csv_files = glob.glob(os.path.join(out_dir, "*.csv"))
        if not csv_files:
            raise ValueError(f"CICFlowMeter produced no CSV output for {pcap_path}")

        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
    finally:
        # clean up CICFlowMeter temp output (not the input PCAP)
        for f in glob.glob(os.path.join(out_dir, "*")):
            os.unlink(f)
        os.rmdir(out_dir)

    if df.empty:
        raise ValueError(f"CICFlowMeter returned 0 flows for {pcap_path}")

    return df
