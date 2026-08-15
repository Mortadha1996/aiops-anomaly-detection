# AIOps — Predictive Infrastructure Anomaly Detection

An end-to-end MLOps pipeline that learns what healthy infrastructure looks like, then flags anomalies before they turn into incidents.

Built to answer a practical operations problem: threshold-based alerting either fires too late (the disk is already full) or too often (CPU spiked for four seconds at 3am). This system learns the normal behaviour of each node from real telemetry and scores every 5-minute window against it.

**Production model: Random Forest, 0.984 F1.**

---

## Why this exists

Classic monitoring alerts on a rule someone wrote once: `cpu > 90% for 5 minutes`. That rule doesn't know that this particular node always spikes during the nightly batch, or that 70% CPU combined with a rising error rate and falling free memory is the shape of a problem forming.

Anomaly detection on multivariate telemetry catches the second case. That's what this pipeline does.

One design decision drives everything else: **recall is weighted higher than precision**. Missing a real anomaly costs an outage. A false alarm costs an engineer three minutes. The scoring formula reflects that.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA COLLECTION (ETL)          every 15 min, cron    │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus ──► metrics (CPU, RAM, disk, container mem, node up)│
│  Loki       ──► logs (container, PostgreSQL, syslog)            │
│                        │                                        │
│                        ▼                                        │
│              clean + transform (Pandas)                          │
│                        │                                        │
│                        ▼                                        │
│         PostgreSQL: infra_metrics · infra_logs                   │
└─────────────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — ML PIPELINE                     every hour, cron     │
├─────────────────────────────────────────────────────────────────┤
│  1. Feature engineering   → 5-min windows per node, 10 features │
│  2. Labelling             → rules + unsupervised bootstrap      │
│  3. Augmentation          → SMOTE + Gaussian (train set only)   │
│  4. Split                 → stratified 60 / 20 / 20             │
│  5. Model competition     → 7 models, weighted scoring          │
│  6. Export                → ONNX + MLflow tracking              │
└─────────────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — SYNTHETIC ANOMALY INJECTION                          │
├─────────────────────────────────────────────────────────────────┤
│  6 failure types injected to bootstrap and validate detection   │
└─────────────────────────────────────────────────────────────────┘
                         │
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — GRAFANA                        refresh every 5 min   │
├─────────────────────────────────────────────────────────────────┤
│  KPI overview · node health · anomaly timeline · logs · replica │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1 — Data collection

`infra_etl.py`, scheduled every 15 minutes.

Pulls metrics from the Prometheus HTTP API and logs from Loki, normalises them with Pandas, and writes to two PostgreSQL tables.

| Table | Contents |
|---|---|
| `infra_metrics` | CPU %, memory %, disk %, container memory MB, node up/down — per node, timestamped |
| `infra_logs` | Log lines with parsed level (ERROR / WARNING / INFO), source and timestamp |

Roughly 36 metric rows and 1,000 log rows per cycle across the cluster.

---

## Layer 2 — ML pipeline

`infra_ml.py`, scheduled hourly. Retrains from scratch on accumulated data, then scores the most recent windows.

### Feature engineering

Raw metrics and logs are aggregated into **5-minute windows per node**, producing 10 features:

```
cpu_pct              mem_pct              disk_pct
container_mem_mb     node_up              total_logs
error_count          warning_count        error_rate
warning_rate
```

Combining metrics with log-derived features matters: a node at 80% memory with a flat error rate is busy, the same node with a climbing error rate is failing.

### Labelling — unsupervised bootstrap

Infrastructure telemetry arrives without anomaly labels. Nobody tags "this window was bad". So labels are generated before any supervised model can train:

| Method | Role |
|---|---|
| **Isolation Forest** | Splits the feature space randomly with trees. Points isolated in few cuts are outliers. |
| **One-Class SVM** | Learns an RBF boundary around normal data. Anything outside is flagged. |
| **Rule-based thresholds** | Domain knowledge — known bad states that must always be labelled. |

The three signals are combined into the final training label. Two unsupervised perspectives plus explicit domain rules is more robust than any one of them alone.

### Augmentation

SMOTE (Synthetic Minority Oversampling) plus Gaussian noise balances the dataset — real anomalies are rare, so the minority class needs help.

**Augmentation is applied to the training set only.** Validation and test sets stay untouched real data. Augmenting them would inflate the scores and tell you nothing about production behaviour.

### Split

Stratified 60 / 20 / 20 — train, validation, test. Stratified because class balance must be preserved across splits when the positive class is rare.

### Model competition

Seven models train and compete on every run:

| Model | Type | Score | Note |
|---|---|---|---|
| **Random Forest** | Supervised | **0.984** | Current production model. 200 trees voting in parallel. Strong on tabular data, robust to noise. |
| XGBoost | Supervised | 0.977 | Sequential boosting, each tree correcting the last. Good on complex interactions. |
| LightGBM | Supervised | 0.977 | Same boosting principle, leaf-wise growth. Faster, lighter on memory. |
| MLP (neural net) | Supervised | 0.972 | Three hidden layers 64→32→16, ReLU. Learns non-linear feature combinations. Has won runs as data quality improved. |
| Logistic Regression | Supervised | 0.935 | Linear baseline. Interpretable, but can't capture feature interactions. |
| One-Class SVM | Unsupervised | 0.076 | Used for labelling, not prediction. |
| Isolation Forest | Unsupervised | 0.021 | Same — bootstrap only. |

**Scoring formula:**

```
score = 0.4 × F1  +  0.35 × Recall  +  0.25 × ROC-AUC
```

Recall carries the second-highest weight deliberately. In infrastructure monitoring a false negative is an outage; a false positive is a glance at a dashboard. The formula encodes that asymmetry rather than optimising accuracy blindly.

The unsupervised models scoring near zero is expected — they are bootstrap labellers, not classifiers, and they're kept in the competition table for transparency.

### Export

The winning model is exported to **ONNX**, so inference runs anywhere without a Python or scikit-learn runtime. Every run is tracked in **MLflow**: parameters, metrics, artefacts and model version.

---

## Layer 3 — Synthetic anomaly injection

`inject_anomalies.py`

Real anomalies are rare by definition, which starves supervised training early on. Six anomaly types are generated from real infrastructure thresholds and mixed into the dataset with `label=1`:

- CPU spike (> 90%)
- Memory pressure (< 5% available)
- Disk saturation (> 90%)
- Node down (`up = 0`)
- Container OOM (> 7 GB)
- Replication lag spike

As real anomalies accumulate from normal operations and chaos testing, the synthetic share is reduced.

---

## Layer 4 — Grafana

Five dashboard sections, refreshing every 5 minutes.

**KPI overview** — total anomalies (12h), critical count, nodes up, active model name, errors last hour, affected nodes

**Node health** — CPU, memory and disk percentage per node, straight from Prometheus

**ML anomaly detection** — anomaly timeline by severity (critical / high / medium / low), anomaly score over time, per-node gauge, and a recent-anomalies table with a plain-text reason for each

**Log analysis** — error rate per node, log volume by level, live error stream

**Database & replication** — PostgreSQL and Patroni logs, replication events over time

The plain-text reason column matters: an anomaly score of 0.91 tells an on-call engineer nothing. *"Memory 94%, error rate 6× baseline, node-03"* tells them where to look.

---

## Automation

| Job | Schedule | Mechanism |
|---|---|---|
| ETL collection | every 15 min | `/etc/cron.d/etl` |
| ML retrain + inference | hourly | `/etc/cron.d/infra_ml` |
| Anomaly cleanup (12h retention) | every ML run | built into `infra_ml.py` |
| MLflow server | always on | systemd service |
| Grafana refresh | every 5 min | dashboard config |

---

## Stack

**Data** — Prometheus · Loki · PostgreSQL · Pandas
**ML** — scikit-learn · XGBoost · LightGBM · imbalanced-learn (SMOTE) · MLflow · ONNX
**Ops** — cron · systemd · Grafana

---

## Repository structure

```
.
├── etl/
│   └── infra_etl.py              # Layer 1 — Prometheus + Loki → PostgreSQL
├── ml/
│   ├── infra_ml.py               # Layer 2 — orchestrates the full pipeline
│   ├── features.py               # windowing and feature engineering
│   ├── labelling.py              # Isolation Forest + One-Class SVM + rules
│   ├── augmentation.py           # SMOTE + Gaussian noise
│   ├── models.py                 # the seven competitors
│   └── scoring.py                # weighted scoring + model selection
├── synthetic/
│   └── inject_anomalies.py       # Layer 3 — six failure types
├── grafana/
│   └── dashboards/               # exported dashboard JSON
├── sql/
│   └── schema.sql                # infra_metrics, infra_logs
├── cron/
│   ├── etl
│   └── infra_ml
├── requirements.txt
└── README.md
```

---

## Running it

```bash
git clone https://github.com/Mortadha1996/aiops-anomaly-detection
cd aiops-anomaly-detection
pip install -r requirements.txt

# configure connections
cp config.example.yaml config.yaml
# edit: Prometheus URL, Loki URL, PostgreSQL DSN

# initialise the database
psql -f sql/schema.sql

# bootstrap training data
python synthetic/inject_anomalies.py

# run the pipeline
python etl/infra_etl.py
python ml/infra_ml.py
```

---

## Notes

This repository is an independent reimplementation of a system I designed and operate in production. It runs on synthetic and anonymised data — no employer telemetry, configuration or infrastructure detail is included.

---

## Author

**Mortadha Riahi** — Infrastructure & Platform Engineer

RHCE · RHCSA · Red Hat Ansible (×2) · AWS Solutions Architect · CCNA

[LinkedIn](https://www.linkedin.com/in/mortadha-riahi/) · [Medium](https://medium.com/@mortariahi.mr)
