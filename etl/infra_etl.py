#!/usr/bin/env python3
import sys, os
from pathlib import Path
from datetime import datetime, timezone
import requests, yaml, psycopg2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def cfg():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def prom_query(base, query, timeout):
    r = requests.get(base.rstrip("/") + "/api/v1/query", params={"query": query},
                     timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]["result"]

def prom_value(results, default=0.0):
    if not results:
        return default
    try:
        return float(results[0]["value"][1])
    except Exception:
        return default

def collect_metrics(c):
    p = c["prometheus"]
    q = {
        "cpu_pct": '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
        "mem_pct": '100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)',
        "disk_pct": '100 * (1 - node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"} / node_filesystem_size_bytes{fstype!~"tmpfs|overlay"})',
        "container_mem_mb": 'sum by(instance)(container_memory_working_set_bytes{container!="POD",container!=""}) / 1024 / 1024',
        "node_up": 'min by(instance)(up{job=~"node.*"})'
    }
    now = datetime.now(timezone.utc)
    vals = {k: prom_value(prom_query(p["url"], v, p.get("timeout",15)),
                          1.0 if k == "node_up" else 0.0) for k,v in q.items()}
    return [{"ts": now, "node": "cluster", **vals}]

def collect_logs(c):
    # Loki query is intentionally broad. Adjust selector to your labels.
    l = c["loki"]
    query = '{job=~".+"}'
    end = int(datetime.now(timezone.utc).timestamp() * 1e9)
    start = end - 15 * 60 * 1_000_000_000
    try:
        r = requests.get(l["url"].rstrip("/") + "/loki/api/v1/query_range",
                         params={"query": query, "start": start, "end": end, "limit": 1000, "direction":"backward"},
                         timeout=l.get("timeout",15))
        r.raise_for_status()
        streams = r.json()["data"]["result"]
    except Exception as e:
        print(f"Loki collection warning: {e}")
        return []
    rows = []
    for stream in streams:
        labels = stream.get("stream", {})
        node = labels.get("instance") or labels.get("node") or "cluster"
        source = labels.get("job", "unknown")
        for ts, msg in stream.get("values", []):
            upper = msg.upper()
            level = "ERROR" if "ERROR" in upper else "WARNING" if "WARN" in upper else "INFO"
            rows.append({"ts": datetime.fromtimestamp(int(ts)/1e9, timezone.utc),
                         "node": node, "source": source, "level": level, "message": msg})
    return rows

def main():
    c = cfg()
    conn = psycopg2.connect(c["postgres"]["dsn"])
    try:
        metrics = collect_metrics(c)
        logs = collect_logs(c)
        with conn, conn.cursor() as cur:
            for x in metrics:
                cur.execute("""INSERT INTO infra_metrics
                    (ts,node,cpu_pct,mem_pct,disk_pct,container_mem_mb,node_up)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (ts,node) DO UPDATE SET
                    cpu_pct=EXCLUDED.cpu_pct, mem_pct=EXCLUDED.mem_pct,
                    disk_pct=EXCLUDED.disk_pct, container_mem_mb=EXCLUDED.container_mem_mb,
                    node_up=EXCLUDED.node_up""",
                    (x["ts"],x["node"],x["cpu_pct"],x["mem_pct"],x["disk_pct"],
                     x["container_mem_mb"],x["node_up"]))
            for x in logs:
                cur.execute("""INSERT INTO infra_logs
                    (ts,node,source,level,message) VALUES (%s,%s,%s,%s,%s)""",
                    (x["ts"],x["node"],x["source"],x["level"],x["message"]))
        print(f"ETL complete: {len(metrics)} metric rows, {len(logs)} log rows")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
