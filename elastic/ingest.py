"""
Phase 12 — Elasticsearch ingestion

Bulk-indexes classified flow records into the ids-alerts index so Kibana
can visualize attack patterns over time.

Usage:
  from elastic.ingest import get_client, ingest_flows
  es = get_client()
  ingest_flows(es, flows_df, source="upload", pcap_file="capture.pcap")

Environment variables:
  ES_HOST      — Elasticsearch URL (default: http://localhost:9200)
  ES_API_KEY   — API key for cloud deployments (optional)
  ES_USER      — username (optional, basic auth)
  ES_PASSWORD  — password (optional, basic auth)
"""

import os
from datetime import datetime, timezone
from typing import Iterator

import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from elastic.schema import INDEX_NAME, create_index

ES_HOST     = os.environ.get("ES_HOST",     "http://localhost:9200")
ES_API_KEY  = os.environ.get("ES_API_KEY",  "")
ES_USER     = os.environ.get("ES_USER",     "")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")


def get_client() -> Elasticsearch:
    """Build an Elasticsearch client from environment variables."""
    kwargs: dict = {"hosts": [ES_HOST]}

    if ES_API_KEY:
        kwargs["api_key"] = ES_API_KEY
    elif ES_USER and ES_PASSWORD:
        kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)

    es = Elasticsearch(**kwargs)
    if not es.ping():
        raise ConnectionError(
            f"Cannot reach Elasticsearch at {ES_HOST}. "
            "Is it running?  docker run -p 9200:9200 -e 'discovery.type=single-node' "
            "-e 'xpack.security.enabled=false' elasticsearch:8.13.4"
        )
    return es


def _doc_iter(flows: pd.DataFrame, source: str, pcap_file: str) -> Iterator[dict]:
    """Yield one ES action dict per flow row."""
    ts = datetime.now(timezone.utc).isoformat()
    for _, row in flows.iterrows():
        doc = row.to_dict()
        doc["@timestamp"] = ts
        doc["source"]     = source
        doc["pcap_file"]  = pcap_file
        yield {"_index": INDEX_NAME, "_source": doc}


def ingest_flows(
    es:        Elasticsearch,
    flows:     pd.DataFrame,
    source:    str = "upload",
    pcap_file: str = "",
    ensure_index: bool = True,
) -> tuple[int, list]:
    """
    Bulk-index classified flows into Elasticsearch.

    Args:
      es           — Elasticsearch client (from get_client())
      flows        — DataFrame returned by predict.predict_flows()
      source       — "upload" or "live"
      pcap_file    — original PCAP filename (for traceability)
      ensure_index — create the index if it doesn't exist yet

    Returns:
      (success_count, errors)
    """
    if ensure_index:
        create_index(es)

    success, errors = bulk(
        es,
        _doc_iter(flows, source, pcap_file),
        raise_on_error=False,
        stats_only=False,
    )
    return success, errors


def query_recent_alerts(es: Elasticsearch, tier: str = "HIGH", n: int = 100) -> pd.DataFrame:
    """
    Fetch the most recent `n` alerts of a given tier from Elasticsearch.
    Useful for Kibana triage views and API /alerts endpoints.
    """
    resp = es.search(
        index=INDEX_NAME,
        body={
            "size": n,
            "query": {"term": {"alert_tier": tier}},
            "sort": [{"@timestamp": {"order": "desc"}}],
        },
    )
    hits = [h["_source"] for h in resp["hits"]["hits"]]
    return pd.DataFrame(hits)
