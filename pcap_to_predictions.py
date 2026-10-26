"""
Standalone local test: PCAP -> pip `cicflowmeter` -> rename columns to match
Java-CICFlowMeter-trained features.json -> your existing cleaning.py /
predict.py pipeline.

Run from your PROJECT ROOT (same level as pipeline/ and artifacts/):
    python pcap_to_predictions.py test_capture.pcap

This does NOT touch FastAPI app or Docker. It's a standalone script to
answer one question: does real-PCAP-derived data, once column-renamed,
produce sane predictions through your existing pipeline?

If this works, the natural next step is wiring this same logic into
feature_extraction.py so /predict can use it directly.
"""

import sys
import os
import json

import pandas as pd

from cicflowmeter.sniffer import create_sniffer
from pipeline.cleaning import clean_cicflow_output
from pipeline.predict import predict_flows

ARTIFACTS_BASE = os.path.join(os.path.dirname(__file__), "artifacts")
FEATURES_PATH = os.path.join(ARTIFACTS_BASE, "features.json")

# Mapping: pip `cicflowmeter` (hieulw/cicflowmeter) snake_case column name
# -> Java CICFlowMeter Title-Case name your model was trained on.
# Extend this if clean_cicflow_output() reports additional missing columns --
# it will tell you exactly which expected names weren't found.
RENAME_MAP = {
    "bwd_header_len":    "Bwd Header Length",
    "fwd_header_len":    "Fwd Header Length",
    "pkt_len_max":        "Packet Length Max",
    "fwd_seg_size_min":   "Fwd Seg Size Min",
    "fwd_pkt_len_mean":   "Fwd Packet Length Mean",
    "fwd_seg_size_avg":   "Fwd Segment Size Avg",
    "pkt_size_avg":       "Average Packet Size",
    "pkt_len_mean":       "Packet Length Mean",
    "totlen_fwd_pkts":    "Total Length of Fwd Packet",
    "init_bwd_win_byts":  "Bwd Init Win Bytes",
    "fwd_pkt_len_max":    "Fwd Packet Length Max",
    "init_fwd_win_byts":  "FWD Init Win Bytes",
    "tot_fwd_pkts":       "Total Fwd Packet",
    "ack_flag_cnt":       "ACK Flag Count",
    "bwd_pkts_s":         "Bwd Packets/s",
    "fwd_pkt_len_std":    "Fwd Packet Length Std",
    "flow_iat_mean":      "Flow IAT Mean",
    "pkt_len_var":        "Packet Length Variance",
    "pkt_len_std":        "Packet Length Std",
    "flow_pkts_s":        "Flow Packets/s",
    "flow_byts_s":        "Flow Bytes/s",
    "bwd_pkt_len_max":    "Bwd Packet Length Max",
    "bwd_pkt_len_mean":   "Bwd Packet Length Mean",
}

# Engineered ratio features -- NOT raw CICFlowMeter output. These were
# created in your notebook's feature-engineering step. The formulas below
# are a guess at standard fwd/bwd convention -- CHECK YOUR NOTEBOOK'S ACTUAL
# FEATURE ENGINEERING CELL and fix these if the direction/columns are wrong.
def add_engineered_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # work on a copy so the caller's DataFrame isn't modified
    # tiny value added to each denominator so a zero count (e.g. no backward bytes) doesn't divide by zero
    eps = 1e-6
    # columns are a mix of renamed Title-Case and raw snake_case (the bwd totals aren't in RENAME_MAP),
    # so this must run AFTER the rename
    df["Fwd_Bwd_Bytes_Ratio"] = df["Total Length of Fwd Packet"] / (df["totlen_bwd_pkts"] + eps)  # forward bytes vs backward bytes
    df["Fwd_Bwd_Packet_Ratio"] = df["Total Fwd Packet"] / (df["tot_bwd_pkts"] + eps)            # forward packets vs backward packets
    df["Init_Win_Ratio"] = df["FWD Init Win Bytes"] / (df["Bwd Init Win Bytes"] + eps)          # initial TCP window size, client vs server
    return df


def pcap_to_raw_csv(pcap_path: str, csv_path: str, timeout: float = 30.0) -> None:
    """Run the pip cicflowmeter tool correctly (keyword args, bypassing its
    own CLI's positional-argument bug)."""
    # build the sniffer (reads packets) and session (groups packets into flows, writes CSV rows)
    sniffer, session = create_sniffer(
        input_file=pcap_path,    # read packets from the PCAP file
        input_interface=None,    # no live network capture
        output_mode="csv",       # write flows as CSV rows
        output=csv_path,         # ...into this file
        fields=None,             # keep all flow features
        verbose=False,           # no per-packet logging
    )
    sniffer.start()                # start reading packets in a background thread
    sniffer.join(timeout=timeout)  # wait for it to finish reading the PCAP, up to `timeout` seconds

    # stop cicflowmeter's internal garbage-collection thread (expires idle flows on a timer)
    # hasattr guard: these are private attributes and may not exist in every cicflowmeter version
    if hasattr(session, "_gc_stop"):
        session._gc_stop.set()                # signal the gc thread to stop
        session._gc_thread.join(timeout=2.0)  # wait up to 2s for it to actually exit

    # write out every flow still open when the PCAP ended; otherwise those rows are lost
    session.flush_flows()


def main(pcap_path: str):
    # fixed-name scratch file in the current folder (overwritten each run, never deleted)
    raw_csv = "raw_flows_tmp.csv"

    # Step 1: PCAP -> raw flow CSV
    print(f"[1/4] Running pip cicflowmeter on {pcap_path} ...")
    pcap_to_raw_csv(pcap_path, raw_csv)

    # Step 2: load flows, rename to training column names, add the 3 ratio features
    print(f"[2/4] Loading {raw_csv} and renaming columns ...")
    df = pd.read_csv(raw_csv)
    df = df.rename(columns=RENAME_MAP)
    df = add_engineered_ratios(df)

    # load the exact feature list (and order) the model was trained on
    with open(FEATURES_PATH) as f:
        expected = json.load(f)

    # check every feature the model needs is present before predicting
    missing_after_rename = [c for c in expected if c not in df.columns]
    if missing_after_rename:
        # debugging aid: show what's missing and which raw columns are left over,
        # so you can find the matching pip-tool name and add it to RENAME_MAP
        print(f"\n  Still missing {len(missing_after_rename)} expected column(s) after rename:")
        print(f"  {missing_after_rename}")
        print("  Add these to RENAME_MAP once you find the matching pip-tool column name.")
        print(f"  Available (unrenamed-remaining) columns: {sorted(set(df.columns) - set(RENAME_MAP.values()))}")
        return  # stop here instead of predicting on misaligned data

    # Step 3: clean (inf/NaN -> 0, reorder columns) and score with the saved model
    print("[3/4] Running clean_cicflow_output() + predict_flows() ...")
    clean_df = clean_cicflow_output(df, FEATURES_PATH)
    pred_df = predict_flows(clean_df)

    # Step 4: print per-flow results
    print("[4/4] Results:")
    print(pred_df[["attack_score", "pred_label", "alert_tier", "action"]].to_string())

    # one-line summary: total flows and how many landed in each alert tier
    tiers = pred_df["alert_tier"].value_counts().to_dict()
    print(f"\nSummary -- total flows: {len(pred_df)}, tiers: {tiers}")


# only runs when executed directly (python pcap_to_predictions.py file.pcap), not when imported
if __name__ == "__main__":
    # require exactly one argument: the PCAP path (sys.argv[0] is the script name itself)
    if len(sys.argv) != 2:
        print("Usage: python pcap_to_predictions.py <input.pcap>")
        sys.exit(1)  # non-zero exit code signals an error to the shell
    main(sys.argv[1])
