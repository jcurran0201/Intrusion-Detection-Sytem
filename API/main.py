"""
Phase 10 — FastAPI inference API

Endpoints:
  POST /predict          Upload a PCAP; returns per-flow predictions as JSON.
  POST /live/start       Start buffering live traffic (rolling window).
  POST /live/stop        Stop live capture; returns predictions for buffered flows.
  GET  /live/status      Is the live capture running?
  GET  /healthz          Sanity check — confirms model loaded successfully.

Run:
  uvicorn api.main:app --reload --port 8000
"""

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

from pipeline.pcap_handler      import save_pcap_bytes
from pipeline.feature_extraction import extract_flows
from pipeline.cleaning           import clean_cicflow_output
from pipeline.predict            import predict_flows, _load_artifacts

ARTIFACTS_BASE   = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FEATURES_PATH    = os.path.join(ARTIFACTS_BASE, "features.json")

# ── Elasticsearch config ──────────────────────────────────────────────────────
ES_HOST  = os.environ.get("ES_HOST", "http://localhost:9200")   # same pattern as CICFLOWMETER_JAR
ES_INDEX = os.environ.get("ES_INDEX", "ids-alerts")

logger = logging.getLogger("ids.api")

# ── live capture state ────────────────────────────────────────────────────────
_live_task:   asyncio.Task | None = None
_live_pcap:   str | None          = None
_live_result: list | None         = None


_es_client: Elasticsearch | None = None

ES_MAPPING = {
    "mappings": {
        "properties": {
            "timestamp":       {"type": "date"},
            "attack_score":    {"type": "float"},
            "alert_tier":      {"type": "keyword"},
            "predicted_class": {"type": "keyword"},
            "action":          {"type": "keyword"},
            "source":          {"type": "keyword"},   # which endpoint produced this record
        }
    }
}


def _get_es_client() -> Elasticsearch | None:
    """Lazily create (and cache) the ES client. Returns None if ES is unreachable."""
    global _es_client
    if _es_client is not None:
        return _es_client
    try:
        client = Elasticsearch(ES_HOST, request_timeout=3)
        if not client.ping():
            raise ConnectionError(f"Ping failed for {ES_HOST}")
        if not client.indices.exists(index=ES_INDEX):
            client.indices.create(index=ES_INDEX, body=ES_MAPPING)
        _es_client = client
        return _es_client
    except Exception as e:
        logger.warning(f"Elasticsearch unavailable at {ES_HOST}: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # warm the model cache on startup so the first request isn't slow
    _load_artifacts()
    _get_es_client()   # warm the ES connection too; logs a warning if ES is down, doesn't crash startup
    yield


app = FastAPI(
    title="IDS Inference API",
    description="Behavioral network intrusion detection — flow-level ML classification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_pipeline(pcap_path: str) -> list[dict]:
    """PCAP path → list of per-flow prediction dicts."""
    raw_df   = extract_flows(pcap_path)
    clean_df = clean_cicflow_output(raw_df, FEATURES_PATH)
    pred_df  = predict_flows(clean_df)
    return pred_df[
        ["attack_score", "pred_label", "alert_tier", "action"]
    ].to_dict(orient="records")


def _alert_summary(records: list[dict]) -> dict:
    tiers = [r["alert_tier"] for r in records]
    return {
        "total_flows": len(records),
        "HIGH":   tiers.count("HIGH"),
        "MEDIUM": tiers.count("MEDIUM"),
        "LOW":    tiers.count("LOW"),
    }


def _ingest_to_es(records: list[dict], source: str) -> None:
    """
    Bulk-ingest prediction records into Elasticsearch.  Best-effort — logs and
    returns silently on failure so an ES outage never breaks a prediction
    response.  `source` tags which endpoint produced the records (predict,
    predict_flows, live) so Kibana can filter/compare them.
    """
    client = _get_es_client()
    if client is None:
        logger.warning(f"Skipping ES ingestion for {len(records)} record(s) — ES unavailable.")
        return

    now = datetime.now(timezone.utc)

    def _docs():
        for r in records:
            yield {
                "_index": ES_INDEX,
                "_source": {
                    "timestamp":       now.isoformat(),
                    "attack_score":    r["attack_score"],
                    "alert_tier":      r["alert_tier"],
                    "predicted_class": r["pred_label"],
                    "action":          r["action"],
                    "source":          source,
                },
            }

    try:
        success, failed = es_helpers.bulk(client, _docs(), raise_on_error=False)
        if failed:
            logger.warning(f"ES ingestion: {success} succeeded, {len(failed)} failed.")
    except Exception as e:
        logger.warning(f"ES bulk ingestion error: {e}")


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    model, features, label_map = _load_artifacts()
    return {
        "status": "ok",
        "model_features": len(features),
        "classes": list(label_map.values()),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a PCAP file.  Returns a list of flow-level predictions plus a
    summary of alert tier counts.

    Requires CICFlowMeter to be installed (see pipeline/feature_extraction.py).
    """
    if not file.filename.endswith((".pcap", ".pcapng")):
        raise HTTPException(400, "File must be a .pcap or .pcapng")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    pcap_path = None
    try:
        pcap_path = save_pcap_bytes(raw_bytes)
        records   = _run_pipeline(pcap_path)
    except RuntimeError as e:
        # CICFlowMeter not installed — return a clear message instead of 500
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        if pcap_path and os.path.exists(pcap_path):
            os.unlink(pcap_path)

    _ingest_to_es(records, source="predict")

    return JSONResponse({
        "summary": _alert_summary(records),
        "flows":   records,
    })


@app.post("/predict/flows")
async def predict_flows_endpoint(file: UploadFile = File(...)):
    """
    Upload a CICFlowMeter CSV directly (already-extracted flow features).
    Bypasses extract_flows() entirely, so this works regardless of the
    Java 11/CICFlowMeter version mismatch — the recommended path for demos
    and for any environment where live PCAP capture isn't available.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "File must be a .csv of CICFlowMeter flow features.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        raw_df   = pd.read_csv(io.BytesIO(raw_bytes))
        clean_df = clean_cicflow_output(raw_df, FEATURES_PATH)
        pred_df  = predict_flows(clean_df)
        records  = pred_df[
            ["attack_score", "pred_label", "alert_tier", "action"]
        ].to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except pd.errors.ParserError as e:
        raise HTTPException(422, f"Could not parse CSV: {e}")

    _ingest_to_es(records, source="predict_flows")

    return JSONResponse({
        "summary": _alert_summary(records),
        "flows":   records,
    })


@app.post("/live/start")
async def live_start(interface: str = "en0", window_seconds: int = 30):
    """
    Start a pyshark LiveCapture on `interface` that buffers `window_seconds`
    of traffic, dumps it to a temp PCAP, runs the full pipeline, and stores
    the result.  Only one capture can run at a time.
    """
    global _live_task, _live_pcap, _live_result

    if _live_task and not _live_task.done():
        raise HTTPException(409, "A live capture is already running.  POST /live/stop first.")

    _live_result = None

    async def _capture():
        global _live_pcap, _live_result
        import pyshark

        tmp = tempfile.NamedTemporaryFile(suffix=".pcap", delete=False)
        tmp.close()
        _live_pcap = tmp.name

        cap = pyshark.LiveCapture(interface=interface, output_file=_live_pcap)
        cap.sniff(timeout=window_seconds)
        cap.close()

        try:
            _live_result = _run_pipeline(_live_pcap)
        except Exception as e:
            _live_result = [{"error": str(e)}]
        finally:
            if os.path.exists(_live_pcap):
                os.unlink(_live_pcap)
            _live_pcap = None

    _live_task = asyncio.create_task(_capture())
    return {"status": "started", "interface": interface, "window_seconds": window_seconds}


@app.post("/live/stop")
async def live_stop():
    """
    Cancel the live capture (if still running) and return whatever flows
    have been classified so far.
    """
    global _live_task, _live_result

    if _live_task and not _live_task.done():
        _live_task.cancel()
        try:
            await _live_task
        except asyncio.CancelledError:
            pass

    if _live_result is None:
        return {"status": "stopped", "flows": [], "summary": {}}

    result = _live_result
    _live_result = None

    # skip ingestion if _run_pipeline errored out and stashed an {"error": ...} record
    if result and "error" not in result[0]:
        _ingest_to_es(result, source="live")

    return {
        "status":  "stopped",
        "summary": _alert_summary(result),
        "flows":   result,
    }


@app.get("/live/status")
async def live_status():
    running = bool(_live_task and not _live_task.done())
    return {
        "running":      running,
        "has_results":  _live_result is not None,
        "result_count": len(_live_result) if _live_result else 0,
    }
