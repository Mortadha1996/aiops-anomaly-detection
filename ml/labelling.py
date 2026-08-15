import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def generate_labels(X: pd.DataFrame, random_state=42):
    z = X.copy()
    cols = ["cpu_pct","mem_pct","disk_pct","container_mem_mb","node_up",
            "total_logs","error_count","warning_count","error_rate","warning_rate"]
    A = z[cols].replace([np.inf,-np.inf],0).fillna(0)
    if len(A) < 10:
        return np.zeros(len(A), dtype=int)

    iso = IsolationForest(contamination=min(0.12, max(0.02, 2/len(A))),
                          random_state=random_state, n_estimators=200)
    iso_out = (iso.fit_predict(A) == -1)

    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=min(0.1, max(0.02, 2/len(A))))
    oc_out = (oc.fit_predict(A) == -1)

    rules = (
        (z["cpu_pct"] > 90) |
        (z["mem_pct"] > 95) |
        (z["disk_pct"] > 90) |
        (z["node_up"] < 0.5) |
        (z["container_mem_mb"] > 7168) |
        (z["error_rate"] > 0.30)
    ).to_numpy()

    # Two unsupervised votes OR explicit domain rule.
    labels = ((iso_out.astype(int) + oc_out.astype(int)) >= 2) | rules
    return labels.astype(int)
