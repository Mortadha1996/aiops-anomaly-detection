import numpy as np
import pandas as pd

FEATURES = [
    "cpu_pct","mem_pct","disk_pct","container_mem_mb","node_up",
    "total_logs","error_count","warning_count","error_rate","warning_rate"
]

def build_features(metrics: pd.DataFrame, logs: pd.DataFrame, window_minutes=5):
    m = metrics.copy()
    l = logs.copy()
    if m.empty:
        return pd.DataFrame(columns=FEATURES + ["ts","node"])
    m["ts"] = pd.to_datetime(m["ts"], utc=True)
    l["ts"] = pd.to_datetime(l["ts"], utc=True) if not l.empty else pd.Series(dtype="datetime64[ns, UTC]")
    m["window"] = m["ts"].dt.floor(f"{window_minutes}min")
    agg = m.groupby(["window","node"], as_index=False).agg({
        "cpu_pct":"mean","mem_pct":"mean","disk_pct":"mean",
        "container_mem_mb":"mean","node_up":"min"
    })
    if l.empty:
        lg = pd.DataFrame(columns=["window","node","total_logs","error_count","warning_count"])
    else:
        l["window"] = l["ts"].dt.floor(f"{window_minutes}min")
        l["is_error"] = l["level"].str.upper().eq("ERROR")
        l["is_warning"] = l["level"].str.upper().isin(["WARNING","WARN"])
        lg = l.groupby(["window","node"], as_index=False).agg(
            total_logs=("level","size"),
            error_count=("is_error","sum"),
            warning_count=("is_warning","sum")
        )
    out = agg.merge(lg, on=["window","node"], how="left").fillna(0)
    out["error_rate"] = np.divide(out["error_count"], out["total_logs"].clip(lower=1))
    out["warning_rate"] = np.divide(out["warning_count"], out["total_logs"].clip(lower=1))
    return out.rename(columns={"window":"ts"})[["ts","node"] + FEATURES]
