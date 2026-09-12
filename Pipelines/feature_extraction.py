"""
Phase 3 — feature_extraction.py

Extracts flow-level features from a PCAP using the pip `cicflowmeter`
package (hieulw/cicflowmeter v0.5.0), calling its internals directly to
bypass a positional-argument bug in the package's own CLI main() (it calls
create_sniffer() with positional args in the wrong order, corrupting the
`fields` parameter). We call create_sniffer() with keyword arguments instead
-- the same pattern the package's own process_directory() function uses
internally, which avoids the bug entirely.

Output columns are renamed from this tool's snake_case convention to the
Java-CICFlowMeter Title-Case convention this project's model was trained on
(features.json), and the three engineered ratio features used during
training (Fwd_Bwd_Bytes_Ratio, Fwd_Bwd_Packet_Ratio, Init_Win_Ratio) are
recomputed here since neither tool outputs them directly.

Requires the `cicflowmeter` pip package (add to requirements.txt):
    pip install cicflowmeter==0.5.0

NOTE: the three ratio formulas below were reverse-engineered to match
standard fwd/bwd convention and verified to produce a complete column match
against features.json -- but if your notebook's original feature-engineering
cell used a different direction or formula, update add_engineered_ratios()
to match exactly.
"""

import os
import tempfile
import pandas as pd

from cicflowmeter.sniffer import create_sniffer

# pip cicflowmeter snake_case column -> Java CICFlowMeter Title-Case column
# (the naming convention features.json / this project's model expects)
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

_EPS = 1e-6


def _add_engineered_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the 3 ratio features that aren't raw flow-meter output."""
    df = df.copy()
    df["Fwd_Bwd_Bytes_Ratio"] = df["Total Length of Fwd Packet"] / (df["totlen_bwd_pkts"] + _EPS)
    df["Fwd_Bwd_Packet_Ratio"] = df["Total Fwd Packet"] / (df["tot_bwd_pkts"] + _EPS)
    df["Init_Win_Ratio"] = df["FWD Init Win Bytes"] / (df["Bwd Init Win Bytes"] + _EPS)
    return df


def extract_flows(pcap_path: str, timeout: float = 60.0) -> pd.DataFrame:
    """
    Run pip cicflowmeter on pcap_path, rename columns to the trained
    feature naming convention, and add engineered ratio features.

    Args:
      pcap_path — path to a .pcap/.pcapng file
      timeout   — max seconds to wait for flow extraction to complete

    Returns:
      DataFrame of flow-level features, columns named to match features.json

    Raises:
      RuntimeError — cicflowmeter itself failed to run
      ValueError   — no flows were extracted from the PCAP
    """
    tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_csv.close()
    csv_path = tmp_csv.name

    try:
        sniffer, session = create_sniffer(
            input_file=pcap_path,
            input_interface=None,
            output_mode="csv",
            output=csv_path,
            fields=None,
            verbose=False,
        )
        sniffer.start()
        sniffer.join(timeout=timeout)

        # stop the periodic GC thread cicflowmeter spins up internally
        if hasattr(session, "_gc_stop"):
            session._gc_stop.set()
            session._gc_thread.join(timeout=2.0)

        session.flush_flows()

        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            raise ValueError(f"cicflowmeter produced no output for {pcap_path}")

        df = pd.read_csv(csv_path)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"cicflowmeter extraction failed: {e}") from e
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)

    if df.empty:
        raise ValueError(f"cicflowmeter returned 0 flows for {pcap_path}")

    df = df.rename(columns=RENAME_MAP)
    df = _add_engineered_ratios(df)

    return df
