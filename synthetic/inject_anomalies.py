#!/usr/bin/env python3
import sys, os
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ml.infra_ml import demo_data

def main():
    out=ROOT/"data"
    out.mkdir(exist_ok=True)
    df=demo_data()
    rng=np.random.default_rng(42)
    # Explicit six failure types, injected into copies of real/demo windows.
    idx=rng.choice(df.index,size=min(180,len(df)),replace=False)
    chunks=np.array_split(idx,6)
    for i,ids in enumerate(chunks):
        if i==0: df.loc[ids,"cpu_pct"]=rng.uniform(91,99,len(ids))          # CPU spike
        if i==1: df.loc[ids,"mem_pct"]=rng.uniform(96,99,len(ids))          # memory pressure
        if i==2: df.loc[ids,"disk_pct"]=rng.uniform(92,99,len(ids))         # disk saturation
        if i==3: df.loc[ids,"node_up"]=0                                     # node down
        if i==4: df.loc[ids,"container_mem_mb"]=rng.uniform(7200,10000,len(ids)) # OOM
        if i==5: df.loc[ids,"error_rate"]=rng.uniform(.4,.9,len(ids))       # replication/log failure proxy
    df.to_csv(out/"synthetic_telemetry.csv",index=False)
    print(f"Wrote {len(df)} synthetic windows to {out/'synthetic_telemetry.csv'}")

if __name__=="__main__":
    main()
