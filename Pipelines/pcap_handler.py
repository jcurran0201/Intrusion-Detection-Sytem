"""
Phase 6 — pcap_handler.py
pyshark's only job: validate an uploaded PCAP and write it to a stable temp path.
The rest of the pipeline (CICFlowMeter → cleaning → model) reads from that path.
"""

import os
import tempfile
import pyshark


def save_pcap_bytes(file_bytes: bytes) -> str:
    """
    Write raw PCAP bytes to a NamedTemporaryFile, validate it with pyshark,
    return the temp file path.  Caller is responsible for os.unlink() after use.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        _validate(tmp.name)
    except Exception:
        os.unlink(tmp.name)
        raise
    return tmp.name


def save_pcap_path(src_path: str) -> str:
    """
    Copy an on-disk PCAP to a new temp file so downstream steps always work
    from a consistent path and the original file is never modified.
    """
    with open(src_path, "rb") as f:
        return save_pcap_bytes(f.read())


def _validate(path: str) -> None:
    """
    Open the PCAP with pyshark and read the first packet to confirm the file
    is valid and non-empty.  Raises ValueError for unreadable or empty files.
    """
    cap = pyshark.FileCapture(path, keep_packets=False)
    try:
        pkt = next(iter(cap))          # raises StopIteration if empty
        _ = pkt.highest_layer          # access a field to force full parse
    except StopIteration:
        raise ValueError(f"PCAP at {path} contains no packets.")
    except Exception as e:
        raise ValueError(f"PCAP validation failed: {e}") from e
    finally:
        cap.close()
