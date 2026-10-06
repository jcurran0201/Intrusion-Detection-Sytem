# Module docstring: documents the endpoints for anyone reading the file (FastAPI's /docs uses the route docstrings, not this one)
"""
Phase 10 — FastAPI inference API

Endpoints:
  POST /predict          Upload a PCAP; returns per-flow predictions as JSON.
  POST /predict/flows    Upload a CICFlowMeter CSV (pre-extracted flows); bypasses PCAP/Java entirely.
  POST /live/start       Start buffering live traffic (rolling window).
  POST /live/stop        Stop live capture; returns predictions for buffered flows.
  GET  /live/status      Is the live capture running?
  GET  /healthz          Sanity check — confirms model loaded successfully.

All prediction endpoints push their results into Elasticsearch (best-effort —
an ES outage degrades to log-only, it never breaks the API response).

Run:
  uvicorn api.main:app --reload --port 8000
"""

import os
import io
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch, helpers as es_helpers
from pipeline.pcap_handler import save_pcap_bytes
from pipeline.feature_extraction import extract_flows
from pipeline.cleaning import clean_cicflow_output
from pipeline.predict import predict_flows, _load_artifacts

# dirname(__file__) = api/, then ../artifacts → project_root/artifacts; works no matter where uvicorn is launched from
ARTIFACTS_BASE = os.path.join(os.path.dirname(__file__), "..", "artifacts")
# Ordered training feature list; cleaning uses it so inference columns match training columns exactly
FEATURES_PATH = os.path.join(ARTIFACTS_BASE, "features.json")

#Elasticsearch config 
# ES address from an env var, local default otherwise; in Docker/ECS you set ES_HOST without touching code
ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")   # same pattern as CICFLOWMETER_JAR
# Index name (≈ a table) where alert documents land; Kibana's data view points at this
ES_INDEX = os.environ.get("ES_INDEX", "ids-alerts")

# Named logger so you can filter or set levels for just this module
logger = logging.getLogger("ids.api")

# live capture state
# ⚠ Module-level globals = one capture per process; with multiple uvicorn workers each has its own copy
# Handle to the background capture task; /live/stop cancels it, /live/status checks .done()
_live_task: asyncio.Task | None = None
# Path of the temp PCAP the current capture is writing to
_live_pcap: str | None = None
# Finished prediction records; None = nothing ready (not finished, or already collected)
_live_result: list | None = None


# Cached ES client, created lazily so a missing ES doesn't stop the app from starting
_es_client: Elasticsearch | None = None

# Explicit index schema so ES doesn't guess types (a guessed "text" field breaks aggregations)
# ⚠ No flow identifiers (Src/Dst IP, ports, protocol, Flow ID, flow time). alerts can't be traced to a host
ES_MAPPING = {
    "mappings": {
        "properties": {
            # date type → Kibana time picker and time-series charts work
            "timestamp": {"type": "date"},
            # numeric → range filters, averages, histograms
            "attack_score": {"type": "float"},
            # keyword = exact-match string (not tokenized) → filters and pie charts
            "alert_tier": {"type": "keyword"},
            # attack class name (e.g., DDoS, PortScan, BENIGN)
            "predicted_class": {"type": "keyword"},
            # decision-engine recommendation
            "action": {"type": "keyword"},
            # which endpoint produced this record
            "source": {"type": "keyword"},
        }
    }
}


# Returns the cached ES client, connects if needed, or returns None if ES is unreachable
def _get_es_client() -> Elasticsearch | None:
    """ create (and cache) the Elastic Search client. Returns None if Elastic Search is unreachable."""
    # Required because we assign to the module-level variable below
    global _es_client
    # Already connected earlier, reuse it
    if _es_client is not None:
        return _es_client
    try:
        # Build the client; each HTTP call capped at 3s (⚠ default retries multiply this)
        client = Elasticsearch(ES_HOST, request_timeout=3)
        # ping() returns False rather than raising when ES is down
        if not client.ping():
            # Force a failure into the except branch
            raise ConnectionError(f"Ping failed for {ES_HOST}")
        # First run → index doesn't exist yet
        if not client.indices.exists(index=ES_INDEX):
            # Create it with the explicit mapping (⚠ body= is deprecated in elasticsearch-py 8; use mappings=)
            client.indices.create(index=ES_INDEX, body=ES_MAPPING)
        # Only cache once ping and index check both succeed
        _es_client = client
        return _es_client
    # Any connection problem lands here
    except Exception as e:
        # Log it; nothing is cached, so the next call retries (⚠ every request pays the timeout while ES is down)
        logger.warning(f"Elasticsearch unavailable at {ES_HOST}: {e}")
        # Callers treat None as "skip ES"
        return None


# Decorator turns this generator into FastAPI's startup/shutdown hook
@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm the model cache on startup so the first request isn't slow
    _load_artifacts()
    # warm the ES connection too; logs a warning if ES is down, doesn't crash startup
    _get_es_client()
    # Before yield = startup; app serves requests while paused here; after yield would be shutdown
    yield


# The application object; uvicorn finds it via api.main:app
app = FastAPI(
    # Shown at the top of /docs
    title="IDS Inference API",
    # Subtitle on /docs
    description="Behavioral network intrusion detection — flow-level ML classification.",
    # Version shown on /docs
    version="1.0.0",
    # Hook up the startup function above
    lifespan=lifespan,
)


# helpers

# Full pipeline, synchronous and heavy → always call via asyncio.to_thread
def _run_pipeline(pcap_path: str) -> list[dict]:
    """PCAP path → list of per-flow prediction dicts."""
    # Step 1: CICFlowMeter turns packets into one row per flow
    raw_df = extract_flows(pcap_path)
    # Step 2: align columns to training features, fix inf/NaN
    clean_df = clean_cicflow_output(raw_df, FEATURES_PATH)
    # Step 3: model scores each flow and assigns tier + action
    pred_df = predict_flows(clean_df)
    # Keep only output columns; orient="records" → one dict per flow, ready for JSON
    return pred_df[
        ["attack_score", "pred_label", "alert_tier", "action"]
    ].to_dict(orient="records")


# Counts flows per alert tier for the response summary
def _alert_summary(records: list[dict]) -> dict:
    # Pull the tier out of each record (⚠ KeyError on an {"error": ...} record)
    tiers = [r["alert_tier"] for r in records]
    return {
        # Number of classified flows
        "total_flows": len(records),
        # Count of each tier
        "HIGH": tiers.count("HIGH"),
        "MEDIUM": tiers.count("MEDIUM"),
        "LOW": tiers.count("LOW"),
    }


# Sends prediction records to ES; never raises (⚠ synchronous, so it blocks the event loop when called from async routes)
def _ingest_to_es(records: list[dict], source: str) -> None:
    """
    Bulk-ingest prediction records into Elasticsearch.  Best-effort — logs and
    returns silently on failure so an ES outage never breaks a prediction
    response.  `source` tags which endpoint produced the records (predict,
    predict_flows, live) so Kibana can filter/compare them.
    """
    # Cached client, or try to connect
    client = _get_es_client()
    # ES down → log and bail; prediction response is unaffected
    if client is None:
        logger.warning(f"Skipping ES ingestion for {len(records)} record(s) — ES unavailable.")
        return

    # One UTC timestamp for the batch (⚠ ingestion time, not flow time; every flow in a batch gets the same value)
    now = datetime.now(timezone.utc)

    # Generator yielding one ES action per record, so bulk() streams instead of building a big list
    def _docs():
        for r in records:
            yield {
                # Target index
                "_index": ES_INDEX,
                # Document body
                "_source": {
                    # ISO 8601 string, e.g. 2026-10-05T14:03:22+00:00
                    "timestamp": now.isoformat(),
                    "attack_score": r["attack_score"],
                    "alert_tier": r["alert_tier"],
                    # ⚠ Renamed: pred_label (API) → predicted_class (ES)
                    "predicted_class": r["pred_label"],
                    "action": r["action"],
                    # "predict" / "predict_flows" / "live"
                    "source": source,
                },
            }

    try:
        # Batched upload; raise_on_error=False returns rejected docs instead of throwing
        success, failed = es_helpers.bulk(client, _docs(), raise_on_error=False)
        # Some docs were rejected → log counts
        if failed:
            logger.warning(f"ES ingestion: {success} succeeded, {len(failed)} failed.")
    # Connection-level failure (ES died mid-request, etc.)
    except Exception as e:
        logger.warning(f"ES bulk ingestion error: {e}")


# routes

# GET /healthz — conventional liveness-check path for ECS/k8s probes
@app.get("/healthz")
# Plain def (not async) → FastAPI runs it in a threadpool automatically
def healthz():
    # Returns cached artifacts loaded at startup
    model, features, label_map = _load_artifacts()
    return {
        # ⚠ Always "ok"; doesn't report ES connectivity
        "status": "ok",
        # Should match the training feature count
        "model_features": len(features),
        # Class names the model can output
        "classes": list(label_map.values()),
    }


# POST /predict — main PCAP upload endpoint
@app.post("/predict")
# File(...) = required multipart upload ("..." means no default)
async def predict(file: UploadFile = File(...)):
    """
    Upload a PCAP file.  Returns a list of flow-level predictions plus a
    summary of alert tier counts.

    Requires CICFlowMeter to be installed (see pipeline/feature_extraction.py).
    """
    # endswith takes a tuple → accepts either extension (⚠ AttributeError if filename is None)
    if not file.filename.endswith((".pcap", ".pcapng")):
        # 400 Bad Request — wrong file type
        raise HTTPException(400, "File must be a .pcap or .pcapng")

    # Read whole upload into memory (⚠ no size limit)
    raw_bytes = await file.read()
    # Reject empty uploads
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    # Initialized so the finally block can check it even if saving fails
    pcap_path = None
    try:
        # Disk write in a worker thread so other requests keep being served
        pcap_path = await asyncio.to_thread(save_pcap_bytes, raw_bytes)
        # Heavy CICFlowMeter + inference work, also in a thread
        records   = await asyncio.to_thread(_run_pipeline, pcap_path)
    # CICFlowMeter missing
    except RuntimeError as e:
        # CICFlowMeter not installed — return a clear message instead of 500
        raise HTTPException(503, str(e))
    # Data parsed but unusable (e.g., zero flows)
    except ValueError as e:
        # 422 Unprocessable Entity
        raise HTTPException(422, str(e))
    # Runs on success or failure
    finally:
        # Temp file exists → remove it
        if pcap_path and os.path.exists(pcap_path):
            # Prevents temp PCAPs piling up on disk
            os.unlink(pcap_path)

    # Push to Elasticsearch (best-effort)
    _ingest_to_es(records, source="predict")

    # Tier counts plus full per-flow results
    return JSONResponse({
        "summary": _alert_summary(records),
        "flows":   records,
    })


# POST /predict/flows — CSV path that skips CICFlowMeter
@app.post("/predict/flows")
# Named *_endpoint so it doesn't shadow the imported predict_flows function
async def predict_flows_endpoint(file: UploadFile = File(...)):
    """
    Upload a CICFlowMeter CSV directly (already-extracted flow features).
    Bypasses extract_flows() entirely, so this works regardless of the
    Java 11/CICFlowMeter version mismatch — the recommended path for demos
    and for any environment where live PCAP capture isn't available.
    """
    # Only accept CSVs
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a .csv of CICFlowMeter flow features.")

    # Read upload into memory
    raw_bytes = await file.read()
    # Reject empty uploads
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    # ⚠ Everything in this try runs on the event loop (no to_thread) → a big CSV freezes other requests
    try:
        # BytesIO makes bytes look like a file → no temp file needed
        raw_df   = pd.read_csv(io.BytesIO(raw_bytes))
        # Same cleaning as the PCAP path → consistent features
        clean_df = clean_cicflow_output(raw_df, FEATURES_PATH)
        # Same model
        pred_df  = predict_flows(clean_df)
        # Output columns only, one dict per flow
        records  = pred_df[
            ["attack_score", "pred_label", "alert_tier", "action"]
        ].to_dict(orient="records")
    # Missing/misaligned columns, etc.
    except ValueError as e:
        raise HTTPException(422, str(e))
    # ⚠ Never reached: ParserError subclasses ValueError, so the branch above catches it first — swap the order
    except pd.errors.ParserError as e:
        raise HTTPException(422, f"Could not parse CSV: {e}")

    # Push to Elasticsearch (best-effort)
    _ingest_to_es(records, source="predict_flows")

    # Tier counts plus full per-flow results
    return JSONResponse({
        "summary": _alert_summary(records),
        "flows":   records,
    })


# POST /live/start — begin capturing from a network interface
@app.post("/live/start")
# Non-file params on a POST become query params (?interface=eth0&window_seconds=60); en0 is the macOS default
async def live_start(interface: str = "en0", window_seconds: int = 30):
    """
    Start a pyshark LiveCapture on `interface` that buffers `window_seconds`
    of traffic, dumps it to a temp PCAP, runs the full pipeline, and stores
    the result.  Only one capture can run at a time.
    """
    # We reassign these module-level variables
    global _live_task, _live_pcap, _live_result

    # Task exists and hasn't finished → capture already in progress (⚠ two simultaneous requests can both pass)
    if _live_task and not _live_task.done():
        # 409 Conflict
        raise HTTPException(409, "A live capture is already running.  POST /live/stop first.")

    # Clear results from any previous capture
    _live_result = None

    # Inner coroutine that runs in the background after this request returns
    async def _capture():
        # Reassigning globals inside the nested function
        global _live_pcap, _live_result
        # Lazy import → app still starts on machines without pyshark/tshark
        import pyshark

        # Uniquely named temp file; delete=False keeps it on disk after close so tshark can open it by path
        tmp = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
        # Close our handle; we only needed the name
        tmp.close()
        # Store the path globally
        _live_pcap = tmp.name

        # Blocking capture + pipeline, executed in a worker thread
        def _sniff_and_run():
            # ⚠ pyshark needs an event loop in this thread; add asyncio.set_event_loop(asyncio.new_event_loop()) first
            # tshark writes captured packets to the temp PCAP
            cap = pyshark.LiveCapture(interface=interface, output_file=_live_pcap)
            # Block for window_seconds while capturing
            cap.sniff(timeout=window_seconds)
            # Stop tshark and flush the file
            cap.close()
            # Same PCAP → flows → predictions pipeline as /predict
            return _run_pipeline(_live_pcap)

        try:
            # Run the blocking capture off the event loop
            _live_result = await asyncio.to_thread(_sniff_and_run)
        # Note: CancelledError is not an Exception subclass, so a cancel skips this block
        except Exception as e:
            # Stash the error so /live/stop can report it
            _live_result = [{"error": str(e)}]
        # ⚠ On cancel this runs immediately while the worker thread is still sniffing/writing
        finally:
            # Clean up the temp PCAP
            if os.path.exists(_live_pcap):
                os.unlink(_live_pcap)
            # Reset the path
            _live_pcap = None

    # Schedule _capture() in the background so this response returns immediately
    _live_task = asyncio.create_task(_capture())
    # Confirm what was started
    return {"status": "started", "interface": interface, "window_seconds": window_seconds}


# POST /live/stop — end capture and collect results
@app.post("/live/stop")
async def live_stop():
    """
    Cancel the live capture (if still running) and return whatever flows
    have been classified so far.
    """
    # ⚠ Docstring overpromises: classification only happens after the full window, so stopping early returns nothing
    # We reassign these module-level variables
    global _live_task, _live_result

    # Capture still running
    if _live_task and not _live_task.done():
        # Cancel the asyncio task (⚠ cancels the await, not the worker thread — tshark keeps running)
        _live_task.cancel()
        try:
            # Wait for the task to acknowledge cancellation
            await _live_task
        # Expected after cancel()
        except asyncio.CancelledError:
            # Swallow it
            pass

    # Nothing finished (cancelled early or never started)
    if _live_result is None:
        return {"status": "stopped", "flows": [], "summary": {}}

    # Grab the results
    result = _live_result
    # Clear the global so results are returned only once
    _live_result = None

    # skip ingestion if _run_pipeline errored out and stashed an {"error": ...} record
    if result and "error" not in result[0]:
        # Push to Elasticsearch (best-effort)
        _ingest_to_es(result, source="live")

    return {
        "status":  "stopped",
        # ⚠ KeyError on the {"error": ...} record → 500
        "summary": _alert_summary(result),
        "flows":   result,
    }


# GET /live/status — poll capture state without stopping it
@app.get("/live/status")
async def live_status():
    # True only if a task exists and hasn't finished
    running = bool(_live_task and not _live_task.done())
    return {
        "running":      running,
        # Finished results waiting to be collected via /live/stop
        "has_results":  _live_result is not None,
        # Number of result records (an error record counts as 1)
        "result_count": len(_live_result) if _live_result else 0,
    }
