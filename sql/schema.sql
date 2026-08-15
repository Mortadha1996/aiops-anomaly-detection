CREATE TABLE IF NOT EXISTS infra_metrics (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    node TEXT NOT NULL,
    cpu_pct DOUBLE PRECISION,
    mem_pct DOUBLE PRECISION,
    disk_pct DOUBLE PRECISION,
    container_mem_mb DOUBLE PRECISION,
    node_up INTEGER,
    UNIQUE(ts, node)
);

CREATE INDEX IF NOT EXISTS idx_infra_metrics_ts ON infra_metrics(ts);
CREATE INDEX IF NOT EXISTS idx_infra_metrics_node_ts ON infra_metrics(node, ts);

CREATE TABLE IF NOT EXISTS infra_logs (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    node TEXT NOT NULL,
    source TEXT,
    level TEXT,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_infra_logs_ts ON infra_logs(ts);
CREATE INDEX IF NOT EXISTS idx_infra_logs_node_ts ON infra_logs(node, ts);

CREATE TABLE IF NOT EXISTS anomaly_results (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    node TEXT NOT NULL,
    anomaly_probability DOUBLE PRECISION,
    severity TEXT,
    reason TEXT,
    model_name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_results_ts ON anomaly_results(ts);
