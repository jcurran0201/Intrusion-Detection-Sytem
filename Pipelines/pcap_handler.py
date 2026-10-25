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
        tmp.write(file_bytes) #write pcap bytes into the file
        tmp.flush() #Pushes data past Python's internal buffer
        tmp.close() #close the file
        _validate(tmp.name) #validate with the function below
    except Exception:
        os.unlink(tmp.name) # don't leave invalid uploads sitting on disk
        raise # re-raise the original error unchanged for the caller
    return tmp.name


def save_pcap_path(src_path: str) -> str:
    """
    Copy an on-disk PCAP to a new temp file so downstream steps always work
    from a consistent path and the original file is never modified.
    """
    with open(src_path, "rb") as f: #read-only access in binary mode
        return save_pcap_bytes(f.read()) # read the whole file as bytes, copy it into a validated temp .pcap, return the temp path


def _validate(path: str) -> None:
    """
    Open the PCAP with pyshark and read the first packet to confirm the file
    is valid and non-empty.  Raises ValueError for unreadable or empty files.
    """
    cap = pyshark.FileCapture(path, keep_packets=False)
    try:
        pkt = next(iter(cap)) # grab just the first packet; raises StopIteration if empty
        _ = pkt.highest_layer # pyshark is lazy: touching a field forces tshark to fully parse it, so corrupt files fail here
     # no packets at all: turn it into a clear "empty file" error
    except StopIteration:
        raise ValueError(f"PCAP at {path} contains no packets.")
    # anything else (corrupt file, not a PCAP, tshark error): report as bad input, keeping the original error attached
    except Exception as e:
        raise ValueError(f"PCAP validation failed: {e}") from e
    # always shut down the tshark process, pass or fail, so it doesn't leak on every upload
    finally:
        cap.close()
