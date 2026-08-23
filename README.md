# ML-Based Intrusion Detection System

An end-to-end machine learning pipeline for network intrusion detection, built on flow-level traffic features and integrated with an Elastic SIEM stack for alert visualization. The project takes raw packet captures through feature extraction, classification, and alert triage — the same shape as a real detection engineering pipeline, built at portfolio scale.

This project is paired with a companion MS thesis, *ML-Based Intrusion Detection for Cloud-Hosted Financial Transaction Systems*, which explores the same problem space in more depth.

---

## Architecture 
PCAP file

│
▼

pcap_handler.py validates the capture (pyshark) and writes it to a stable temp path
│
▼

feature_extraction.py runs CICFlowMeter, converts raw packets into flow-level statistics

│
▼

cleaning.py aligns columns to the trained feature schema, strips Inf/NaN,

│ fails loudly on missing features instead of silently misaligning
▼

predict.py loads the trained Random Forest model + scaler, scores each flow,

│ assigns an attack class and an alert tier (HIGH / MEDIUM / LOW)
▼

FastAPI layer exposes prediction as an API

│
▼

Elasticsearch + Kibana alerts are ingested into an ids-alerts* index and visualized
on a dashboard (alert tier breakdown, attack class distribution,
attack score over time, HIGH-tier alert table)


Each stage is a standalone module, so any step can be tested, swapped, or re-run independently of the others.

---

## Model

| | |
|---|---|
| **Algorithm** | Random Forest |
| **Dataset** | CIC-IDS network flow data |
| **Classes** | 8 — Benign, Exploits, Fuzzers, Recon, Generic, DoS, Shellcode, Rare_Attack (a merged class combining Analysis, Backdoor, and Worms, which were individually too small to model reliably) |
| **Performance** | Weighted F1 ≈ 0.889 |
| **Preprocessing** | Variance filter → correlation filter → low-signal feature drop → train/test split → SMOTE (train only) → mutual information selection → log transform → RobustScaler |

**Why Random Forest:** it matched more complex alternatives on weighted F1, which indicates the ceiling here is set by the data rather than model capacity — more model complexity wasn't going to buy much more accuracy, so the simpler, more interpretable model was the better choice.

**Feature selection:** the top 20 features by mutual information include several engineered ratio features (`Fwd_Bwd_Bytes_Ratio`, `Init_Win_Ratio`, `Fwd_Bwd_Packet_Ratio`) ranking near the top, which validates the feature engineering approach. Port numbers and protocol type were deliberately never used as features — flow statistics capture behavioral signatures without relying on values an attacker can easily spoof or vary.

---

## ML pipeline & methodology

The preprocessing order matters more than it looks like it should, and getting it wrong is a common source of silently inflated metrics:

variance filter → correlation filter → low-signal feature drop → train/test split
→ SMOTE (train only) → mutual information selection → log transform → scale


**Why this order:**

- **Split before SMOTE.** SMOTE is fit and applied only to the training data. Doing it before the split lets synthetic minority-class samples leak into the test set, which inflates evaluation metrics on classes that are barely represented in the real data.
- **SMOTE before mutual information.** MI is computed after oversampling so the minority classes have enough signal for MI scores to be meaningful — computing MI on the original, heavily imbalanced data would undervalue features that only separate the rare classes.
- **MI after the split, not before.** Selecting features using the full dataset's relationship to the label — including the test set — is itself a leakage path, just a subtler one than the SMOTE-into-test-set case. Both are "the same species of leakage" but they get caught (or missed) at different points, so they're treated as separate checks rather than one rule.
- **SMOTE inside a CV pipeline, not applied once up front.** For cross-validation, SMOTE is wrapped in an `imblearn.pipeline.Pipeline` so synthetic samples are regenerated fresh within each fold rather than being generated once and then split across folds — the latter would let a fold "see" synthetic neighbors derived from data in another fold.
- **SMOTE is capped at ~10% of the majority class size.** For classes with very few real samples (like Worms), unrestricted oversampling produces synthetic points that are near-duplicates of a handful of real ones, which manufactures apparent signal rather than reflecting real class structure.

**Why VIF isn't part of this pipeline:** Variance Inflation Factor addresses multicollinearity that destabilizes *linear* model coefficients. Random Forest splits on one feature at a time and doesn't compute or rely on coefficients, so correlated features don't destabilize it the same way. VIF is relevant during EDA for understanding feature relationships, but it isn't a concern for the model that was ultimately selected.

**Model comparison:** Random Forest was evaluated against other tree-based and boosted approaches, and converged to nearly identical weighted F1 across them — which is itself informative. It suggests the performance ceiling here is set by the data (flow-level snapshots, class imbalance, limited samples for rare attack types) rather than by model capacity. Throwing more model complexity at the problem past this point has diminishing returns, so the simpler model was chosen.

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

├── pcap_handler.py # validates PCAPs, writes to a stable temp path (pyshark)
├── feature_extraction.py # wraps CICFlowMeter, produces flow-level feature CSVs
├── cleaning.py # aligns extracted features to the trained schema
├── predict.py # loads model/scaler/label map, scores flows, assigns tiers
├── main.py # FastAPI layer exposing prediction endpoints
└── artifacts/
├── model.pkl # trained Random Forest model
├── features.json # ordered feature list the model expects
└── label_map.json # class index → attack label mapping


---

## Known limitations

Documented tradeoffs rather than oversights:

- **Java version mismatch.** CICFlowMeter requires Java 11; the current environment runs Java 26. The live PCAP-capture endpoints (`/live/start`, `/live/stop`) are not functional in the Dockerized deployment as a result. The practical workaround is a `/predict/flows` endpoint that accepts pre-extracted flow features directly, bypassing the Java dependency for demo purposes.
- **Live capture also needs raw network interface access**, which is a further constraint on running it inside a container.
- **DoS detection ceiling.** DoS is fundamentally a temporal, cross-flow pattern (many flows in a short window), while the model scores flows independently. Per-flow features can't fully capture this — a windowed aggregation feature would be needed to close the gap.
- **Rare_Attack class reliability.** The classes merged into `Rare_Attack` (particularly Worms, ~49 real samples) don't have enough data to produce statistically reliable per-class metrics; this is reported honestly rather than papered over with synthetic oversampling.
- **No concept drift handling.** The dataset reflects a fixed, controlled lab environment; the model isn't retrained against live traffic drift.
- **Elasticsearch ingestion runs from batch predictions today**, not automatically from the live inference path — the two are not yet wired together.

---

## Roadmap

- [ ] Wire `predict.py` output directly into Elasticsearch ingestion
- [ ] Finish Kibana dashboard (alert tier pie, attack class bar, attack score over time, HIGH-tier table)

---

## Stack

Python · scikit-learn · imbalanced-learn (SMOTE) · SHAP · CICFlowMeter · pyshark · FastAPI · Elasticsearch · Kibana · Docker
