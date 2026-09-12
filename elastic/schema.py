"""
Phase 12 — Elasticsearch index schema

The IDS alert index.  One document per classified network flow.
Kibana reads from this index for all dashboards.

Usage:
  from elastic.schema import create_index
  create_index(es_client)
"""

INDEX_NAME = "ids-alerts"

# ILM policy — keep hot for 7 days, cold for 30, then delete
ILM_POLICY = {
    "policy": {
        "phases": {
            "hot":    {"actions": {"rollover": {"max_age": "7d", "max_size": "50gb"}}},
            "cold":   {"min_age": "30d", "actions": {"freeze": {}}},
            "delete": {"min_age": "60d", "actions": {"delete": {}}},
        }
    }
}

INDEX_SETTINGS = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,           # bump to 1 in multi-node prod
        "refresh_interval":   "5s",
    },
    "mappings": {
        "properties": {
            # ── event metadata ─────────────────────────────────────────────
            "@timestamp":   {"type": "date"},
            "source":       {"type": "keyword"},   # "upload" | "live"
            "pcap_file":    {"type": "keyword"},

            # ── ML prediction ──────────────────────────────────────────────
            "attack_score": {"type": "float"},
            "pred_label":   {"type": "keyword"},
            "alert_tier":   {"type": "keyword"},   # HIGH | MEDIUM | LOW
            "action":       {"type": "keyword"},

            # ── flow features (top-20 stored for investigation) ────────────
            "Flow Duration":                {"type": "float"},
            "Total Fwd Packet":             {"type": "float"},
            "Total Bwd packets":            {"type": "float"},
            "Total Length of Fwd Packet":   {"type": "float"},
            "Total Length of Bwd Packet":   {"type": "float"},
            "Fwd Packet Length Mean":       {"type": "float"},
            "Bwd Packet Length Mean":       {"type": "float"},
            "Flow Bytes/s":                 {"type": "float"},
            "Flow Packets/s":               {"type": "float"},
            "Flow IAT Mean":                {"type": "float"},
            "Fwd IAT Mean":                 {"type": "float"},
            "Bwd IAT Mean":                 {"type": "float"},
            "FIN Flag Count":               {"type": "integer"},
            "SYN Flag Count":               {"type": "integer"},
            "RST Flag Count":               {"type": "integer"},
            "PSH Flag Count":               {"type": "integer"},
            "ACK Flag Count":               {"type": "integer"},
            "Average Packet Size":          {"type": "float"},
            "Packet Length Mean":           {"type": "float"},
            "Packet Length Variance":       {"type": "float"},
        }
    },
}


def create_index(es, recreate: bool = False) -> None:
    """
    Create the IDS alert index in Elasticsearch.

    Args:
      es        — elasticsearch.Elasticsearch client
      recreate  — if True, delete existing index first (dev use only)
    """
    if recreate and es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted existing index: {INDEX_NAME}")

    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index already exists: {INDEX_NAME}")
