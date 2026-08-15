import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

def augment_training(X, y, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return X, y

    k = max(1, min(5, counts.min()-1))
    Xr, yr = SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)

    rng = np.random.default_rng(random_state)
    minority = np.flatnonzero(yr == 1)
    if len(minority):
        noise = rng.normal(0, 0.01, size=(len(minority), Xr.shape[1]))
        scale = np.maximum(np.std(Xr, axis=0), 1e-6)
        Xr[minority] += noise * scale
    return Xr, yr
