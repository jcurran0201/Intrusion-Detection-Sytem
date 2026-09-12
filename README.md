# ML-Based Intrusion Detection System

An end-to-end machine learning pipeline for network intrusion detection, built on flow-level traffic features and integrated with an Elastic SIEM stack for alert visualization. The project takes raw packet captures through feature extraction, classification, and alert triage — the same shape as a real detection engineering pipeline, built at portfolio scale.

---

## Architecture

```
PCAP file
   │
   ▼
pcap_handler.py validates the capture (pyshark) and writes it to a stable temp path
   │
   ▼
feature_extraction.py extracts flow-level statistics via the pip `cicflowmeter`
package, renames its output columns to match the trained feature schema, and
recomputes 3 engineered ratio features (Fwd_Bwd_Bytes_Ratio, Fwd_Bwd_Packet_Ratio,
Init_Win_Ratio) that aren't raw flow-meter output
   │
   ▼
cleaning.py aligns columns to the trained feature schema, strips Inf/NaN,
fails loudly on missing features instead of silently misaligning
   │
   ▼
predict.py loads the trained Random Forest model + scaler, scores each flow,
assigns an attack class and an alert tier (HIGH / MEDIUM / LOW)
   │
   ▼
FastAPI layer exposes prediction as an API
   │
   ▼
Elasticsearch + Kibana — alerts are automatically ingested into an ids-alerts*
index on every prediction and visualized on a dashboard (alert tier breakdown,
attack class distribution, attack score over time, HIGH-tier alert table)
```

Each stage is a standalone module, so any step can be tested, swapped, or re-run independently of the others.

**Why two packet libraries?** `pyshark` and `cicflowmeter` do different jobs and aren't interchangeable. `pyshark` is used only in `pcap_handler.py` to *validate* an uploaded file — confirming it's a real, non-empty, parseable capture before anything else touches it. `cicflowmeter` is used only in `feature_extraction.py` to *extract* flow-level statistics (packet counts, byte counts, IAT, flag counts, etc.) — the actual features the model was trained on. `pyshark` never sees or produces the model's features; `cicflowmeter` never validates the raw file. Splitting these responsibilities means a malformed upload fails fast with a clear error, rather than surfacing as a confusing failure deeper in feature extraction.

**Engineering note — async/blocking I/O conflict:** PCAP validation (`pyshark`) and flow extraction both perform blocking I/O. Since the FastAPI `/predict` route is `async def`, calling these directly conflicts with the framework's own event loop — `pyshark` attempts to start a nested event loop inside one that's already running, raising `RuntimeError: Cannot run the event loop while another loop is running`. Both calls are dispatched through `asyncio.to_thread()` instead, running them in a separate thread rather than blocking (or colliding with) the main event loop.

---

## Model

| | |
|---|---|
| **Algorithm** | Random Forest |
| **Dataset** | CIC-IDS network flow data |
| **Classes** | 8 — Benign, Exploits, Fuzzers, Recon, Generic, DoS, Shellcode, Rare_Attack (a merged class combining Analysis, Backdoor, and Worms, which were individually too small to model reliably) |
| **Macro F1** | ~0.618 |
| **Weighted F1** | ~0.907 |
| **Preprocessing** | Variance filter → correlation filter → low-signal feature drop → train/test split → SMOTE (train only) → mutual information selection → log transform → RobustScaler |

**Why two F1 numbers, and why macro F1 is the one that matters here:** weighted F1 is dominated by the Benign majority class, so a high weighted F1 can look reassuring while hiding poor performance on rare attack types — exactly the classes a real IDS most needs to catch. Macro F1 weights every class equally regardless of size, so it's the more honest metric for judging performance on an imbalanced, multi-class detection problem. Model selection in this project is based on macro F1 for that reason.

**Why Random Forest:** it matched more complex alternatives (XGBoost, MLP) on both metrics, which indicates the ceiling here is set by the data rather than model capacity — more model complexity wasn't going to buy much more accuracy, so the simpler, more interpretable model was the better choice.

**Feature selection:** the top 20 features by mutual information include several engineered ratio features (`Fwd_Bwd_Bytes_Ratio`, `Init_Win_Ratio`, `Fwd_Bwd_Packet_Ratio`) ranking near the top, which validates the feature engineering approach. Port numbers and protocol type were deliberately never used as features — flow statistics capture behavioral signatures without relying on values an attacker can easily spoof or vary.

---

## ML pipeline & methodology

The preprocessing order matters more than it looks like it should, and getting it wrong is a common source of silently inflated metrics:

```
variance filter → correlation filter → low-signal feature drop → train/test split
→ SMOTE (train only) → mutual information selection → log transform → scale
```

**Why this order:**

- **Split before SMOTE.** SMOTE is fit and applied only to the training data. Doing it before the split lets synthetic minority-class samples leak into the test set, which inflates evaluation metrics on classes that are barely represented in the real data.
- **SMOTE before mutual information.** MI is computed after oversampling so the minority classes have enough signal for MI scores to be meaningful — computing MI on the original, heavily imbalanced data would undervalue features that only separate the rare classes.
- **MI after the split, not before.** Selecting features using the full dataset's relationship to the label — including the test set — is itself a leakage path, just a subtler one than the SMOTE-into-test-set case. Both are "the same species of leakage" but they get caught (or missed) at different points, so they're treated as separate checks rather than one rule.
- **SMOTE inside a CV pipeline, not applied once up front.** For cross-validation, SMOTE is wrapped in an `imblearn.pipeline.Pipeline` so synthetic samples are regenerated fresh within each fold rather than being generated once and then split across folds — the latter would let a fold "see" synthetic neighbors derived from data in another fold.
- **SMOTE is capped at ~10% of the majority class size.** For classes with very few real samples (like Worms), unrestricted oversampling produces synthetic points that are near-duplicates of a handful of real ones, which manufactures apparent signal rather than reflecting real class structure.

**Why VIF isn't part of this pipeline:** Variance Inflation Factor addresses multicollinearity that destabilizes *linear* model coefficients. Random Forest splits on one feature at a time and doesn't compute or rely on coefficients, so correlated features don't destabilize it the same way. VIF is relevant during EDA for understanding feature relationships, but it isn't a concern for the model that was ultimately selected.

**Model comparison:** Random Forest was evaluated against other tree-based and boosted approaches (Logistic Regression, XGBoost, MLP), and converged to nearly identical performance across the tree-based/boosted models — which is itself informative. It suggests the performance ceiling here is set by the data (flow-level snapshots, class imbalance, limited samples for rare attack types) rather than by model capacity. Throwing more model complexity at the problem past this point has diminishing returns, so the simpler model was chosen. (Logistic Regression significantly underperformed the others and is not treated as a meaningful baseline.)

---

## Exploratory analysis

A few of the diagnostic plots produced during feature analysis:

- **Mutual information ranking** — which flow features carry the most signal for classification
- **Class mean heatmap** — normalized feature values by attack class, showing which classes are separable and which overlap
- **TCP flag composition** — flag counts by class, useful for spotting SYN-flood / RST-burst style signatures
- **Flow duration / inter-arrival time distributions** — long-tailed, log-transformed before modeling
- **Packets/s vs bytes/s scatter** — separates high-volume flood traffic from low-and-slow attack patterns

---

## Alert tiers

Each scored flow is bucketed into a tier based on `attack_score` (P(not benign)):

| Tier | Threshold | Action |
|---|---|---|
| HIGH | ≥ 0.85 | Escalate immediately |
| MEDIUM | ≥ 0.50 | Flag for review |
| LOW | < 0.50 | Routine logging |

---

## Project structure

```
├── api/
│   └── main.py              # FastAPI layer: /predict, /predict/flows, /live/start,
│                             #   /live/stop, /live/status, /healthz — all prediction
│                             #   endpoints auto-ingest results into Elasticsearch
├── pipeline/
│   ├── pcap_handler.py       # validates PCAPs, writes to a stable temp path (pyshark)
│   ├── feature_extraction.py # runs pip cicflowmeter, renames columns, adds
│   │                         #   engineered ratio features
│   ├── cleaning.py           # aligns extracted features to the trained schema
│   └── predict.py            # loads model/scaler/label map, scores flows, assigns tiers
├── dashboard/
│   └── app.py                # Streamlit operator console — upload a PCAP or trigger
│                              #   live capture, see results for that run
├── artifacts/
│   ├── model.pkl              # trained Random Forest model
│   ├── features.json          # ordered feature list the model expects
│   ├── scaler.joblib           # fitted RobustScaler
│   └── label_map.json          # class index → attack label mapping
├── IDS_detection.ipynb        # primary ML training notebook
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## API

FastAPI service defined in `api/main.py`. Interactive docs (Swagger UI) are auto-generated at `/docs` once the server is running.

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Confirms the model, feature list, and label map loaded successfully. Use this first to sanity-check the service before sending real data. |
| `/predict` | POST | Upload a `.pcap`/`.pcapng` file. Runs the full pipeline (validate → extract flows → clean → score) and returns per-flow predictions plus an alert-tier summary. Results are automatically ingested into Elasticsearch. |
| `/predict/flows` | POST | Upload a CSV of already-extracted flow features (matching `features.json`). Skips PCAP validation and flow extraction entirely — useful for testing the model/cleaning/scoring path in isolation, or for environments where PCAP capture isn't available. Also auto-ingests into Elasticsearch. |
| `/live/start` | POST | Starts a rolling live capture on a given network interface (`interface`, default `en0`) for a given window (`window_seconds`, default 30). Buffers traffic, then runs it through the same pipeline as `/predict` once the window closes. |
| `/live/stop` | POST | Stops the current live capture (if running) and returns predictions for whatever was captured. |
| `/live/status` | GET | Reports whether a live capture is currently running and whether results are ready. |

All prediction endpoints degrade gracefully if Elasticsearch is unreachable — ingestion is best-effort and logs a warning rather than failing the API response.

---

## Running the project

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Start Elasticsearch and Kibana**
```bash
docker-compose up -d
```
Verify Elasticsearch is up: `curl http://localhost:9200` should return cluster info JSON. Kibana takes longer to start — check `http://localhost:5601/api/status` before expecting the UI to respond.

**3. Start the API**
From the project root (not from inside `api/` or `pipeline/` — both need to be visible as sibling folders for imports to resolve):
```bash
uvicorn api.main:app --reload --port 8000
```

**4. Try it**
Open `http://localhost:8000/docs` for the interactive Swagger UI. `GET /healthz` first to confirm the model loaded, then `POST /predict` with a `.pcap` file, or `POST /predict/flows` with a CSV of pre-extracted features.

**5. View results in Kibana**
Open `http://localhost:5601`. If it's the first time, create a data view (Stack Management → Data Views) pointed at `ids-alerts*`, with **`timestamp`** selected as the time field (not the auto-suggested `@timestamp` — the app writes to `timestamp` specifically).

**6. (Optional) Streamlit operator console**
```bash
streamlit run dashboard/app.py
```
Requires the FastAPI server to already be running. Defaults to `http://localhost:8001` as the API URL — adjust in the sidebar if you're running the API on a different port.

---

## Known limitations

Documented tradeoffs rather than oversights:

- **PCAP flow extraction uses a Python reimplementation, not the original Java CICFlowMeter.** The upstream Java tool (`ahlashkari/CICFlowMeter`) has no pre-built binary — it's a from-source Gradle/Maven build with no published releases. This project instead uses the `hieulw/cicflowmeter` pip package, calling its internals directly (its own CLI has an unrelated argument-order bug, worked around here) and mapping its output columns to the naming convention the model was trained on. Column *names* were verified to match completely; the underlying flow-statistic *computations* may differ slightly from the Java tool's, since neither tool's internals were cross-validated line-for-line.
- **DoS detection ceiling.** DoS is fundamentally a temporal, cross-flow pattern (many flows in a short window), while the model scores flows independently. Per-flow features can't fully capture this — a windowed aggregation feature would be needed to close the gap.
- **Rare_Attack class reliability.** The classes merged into `Rare_Attack` (particularly Worms, ~49 real samples) don't have enough data to produce statistically reliable per-class metrics; this is reported honestly rather than papered over with synthetic oversampling.
- **No concept drift handling.** The dataset reflects a fixed, controlled lab environment; the model isn't retrained against live traffic drift.
- **Live auto-capture (`/live/start`) requires raw network interface access**, which is a further constraint on running it inside a container — separate from the flow-extraction question above.
- **AWS ECS deployment is out of scope by design**, not a gap — this project targets a local, demoable deployment.

---

## Roadmap

- [x] Wire prediction output directly into Elasticsearch ingestion
- [x] Get real PCAP-derived flows through the full pipeline into the dashboard
- [ ] Finish remaining Kibana dashboard panels (attack class bar chart, attack score over time, HIGH-tier table — alert tier pie chart complete)
- [ ] MITRE ATT&CK mapping table (attack classes → tactics/technique IDs)
- [ ] Per-tier incident response playbook (analyst actions + SLAs for HIGH/MEDIUM/LOW)
- [ ] Rule-vs-ML comparison for the DoS class (naive SYN-flood heuristic vs. RF on the same test split)

---

## Stack

Python · scikit-learn · imbalanced-learn (SMOTE) · SHAP · `cicflowmeter` (hieulw, Python) · pyshark · FastAPI · Elasticsearch · Kibana · Docker · Streamlit
