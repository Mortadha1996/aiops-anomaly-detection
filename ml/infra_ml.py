#!/usr/bin/env python3
import sys, os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml, psycopg2, joblib, mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from skl2onnx import to_onnx

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import build_features, FEATURES
from ml.labelling import generate_labels
from ml.augmentation import augment_training
from ml.models import get_models
from ml.scoring import weighted_score, choose_best

def load_cfg():
    with open(ROOT/"config.yaml") as f:
        return yaml.safe_load(f)

def demo_data():
    rng = np.random.default_rng(42)
    n_nodes, n = 6, 2500
    rows, logs = [], []
    start = pd.Timestamp.utcnow().floor("5min") - pd.Timedelta(minutes=5*n)
    nodes = [f"node-{i:02d}" for i in range(1,n_nodes+1)]
    for i in range(n):
        ts = start + pd.Timedelta(minutes=5*i)
        for node in nodes:
            cpu = np.clip(rng.normal(45,12),0,100)
            mem = np.clip(rng.normal(55,10),0,100)
            disk = np.clip(rng.normal(50,12),0,100)
            cmem = max(200,rng.normal(1800,300))
            up = 1
            if rng.random() < 0.025:
                cpu = rng.uniform(91,99)
            if rng.random() < 0.018:
                mem = rng.uniform(96,99)
            if rng.random() < 0.012:
                disk = rng.uniform(92,99)
            if rng.random() < 0.008:
                up = 0
            if rng.random() < 0.015:
                cmem = rng.uniform(7200,10000)
            total = int(rng.poisson(10))
            errors = int(rng.binomial(max(total,1), 0.03))
            warnings = int(rng.binomial(max(total-errors,1), 0.08))
            rows.append([ts,node,cpu,mem,disk,cmem,up,total,errors,warnings])
    d = pd.DataFrame(rows, columns=["ts","node","cpu_pct","mem_pct","disk_pct",
                                     "container_mem_mb","node_up","total_logs",
                                     "error_count","warning_count"])
    d["error_rate"] = d.error_count/d.total_logs.clip(lower=1)
    d["warning_rate"] = d.warning_count/d.total_logs.clip(lower=1)
    return d

def load_data(c):
    try:
        conn = psycopg2.connect(c["postgres"]["dsn"])
        m = pd.read_sql("SELECT ts,node,cpu_pct,mem_pct,disk_pct,container_mem_mb,node_up FROM infra_metrics", conn)
        l = pd.read_sql("SELECT ts,node,source,level,message FROM infra_logs", conn)
        conn.close()
        if len(m):
            return build_features(m,l,c["pipeline"].get("window_minutes",5))
    except Exception as e:
        print("PostgreSQL unavailable; using demo telemetry:", e)
    return demo_data()

def metrics(y, pred, proba):
    f1 = f1_score(y,pred,zero_division=0)
    rec = recall_score(y,pred,zero_division=0)
    try: auc = roc_auc_score(y,proba)
    except Exception: auc = 0.5
    return f1,rec,auc,weighted_score(f1,rec,auc)

def save_anomalies(c, features, model, model_name):
    X = features[FEATURES].replace([np.inf,-np.inf],0).fillna(0)
    proba = model.predict_proba(X)[:,1]
    out = features[["ts","node"]].copy()
    out["anomaly_probability"] = proba
    out["severity"] = pd.cut(proba, [-1,.25,.5,.75,1.01],
                             labels=["low","medium","high","critical"])
    def reason(r):
        f = []
        if r.cpu_pct > 90: f.append(f"CPU {r.cpu_pct:.0f}%")
        if r.mem_pct > 95: f.append(f"memory {r.mem_pct:.0f}%")
        if r.disk_pct > 90: f.append(f"disk {r.disk_pct:.0f}%")
        if r.node_up < .5: f.append("node down")
        if r.container_mem_mb > 7168: f.append(f"container memory {r.container_mem_mb:.0f}MB")
        if r.error_rate > .30: f.append(f"error rate {r.error_rate:.1%}")
        return ", ".join(f) or "multivariate deviation from normal"
    out["reason"] = features.apply(reason,axis=1)
    out["model_name"] = model_name
    out = out[out.anomaly_probability >= .5]
    try:
        conn=psycopg2.connect(c["postgres"]["dsn"])
        with conn, conn.cursor() as cur:
            for _,r in out.iterrows():
                cur.execute("""INSERT INTO anomaly_results
                (ts,node,anomaly_probability,severity,reason,model_name)
                VALUES (%s,%s,%s,%s,%s,%s)""",
                (r.ts,r.node,float(r.anomaly_probability),str(r.severity),
                 r.reason,model_name))
        conn.close()
    except Exception as e:
        out.to_csv(ROOT/"anomaly_results.csv",index=False)
        print("Could not write PostgreSQL anomaly_results; saved anomaly_results.csv:",e)
    return out

def main():
    c=load_cfg()
    f=load_data(c)
    if len(f)<30:
        raise RuntimeError("Not enough telemetry windows.")
    y=generate_labels(f,c["pipeline"].get("random_state",42))
    # Guarantee both classes for a fresh/demo run.
    if y.sum()==0:
        y[((f.cpu_pct>85)|(f.mem_pct>85)|(f.disk_pct>85))] = 1
    X=f[FEATURES].replace([np.inf,-np.inf],0).fillna(0).astype(float)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.40,stratify=y,random_state=42)
    Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.50,stratify=ytmp,random_state=42)
    Xtr_aug,ytr_aug=augment_training(Xtr,ytr,42)

    Path(ROOT/"models").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(c["mlflow"]["tracking_uri"])
    mlflow.set_experiment(c["mlflow"]["experiment"])

    results=[]
    with mlflow.start_run(run_name="model-competition"):
        for name,model in get_models(42).items():
            if name=="Isolation Forest":
                model=IsolationForest(n_estimators=200,contamination="auto",random_state=42)
                model.fit(Xtr)
                pred=(model.predict(Xv)==-1).astype(int)
                proba=np.where(model.decision_function(Xv)<0,1,0).astype(float)
            else:
                model.fit(Xtr_aug,ytr_aug)
                pred=model.predict(Xv)
                proba=model.predict_proba(Xv)[:,1]
            f1,rec,auc,score=metrics(yv,pred,proba)
            results.append({"name":name,"model":model,"f1":f1,"recall":rec,
                            "roc_auc":auc,"score":score})
            mlflow.log_metrics({f"{name}_f1":f1,f"{name}_recall":rec,
                                f"{name}_roc_auc":auc,f"{name}_score":score})
            print(f"{name:20} F1={f1:.3f} Recall={rec:.3f} AUC={auc:.3f} Score={score:.3f}")

        best=choose_best(results)
        mlflow.log_param("winning_model",best["name"])
        mlflow.log_metric("winning_score",best["score"])

    # Refit winner on train+validation augmented data, keep test untouched.
    Xtrain=pd.concat([Xtr,Xv])
    ytrain=np.concatenate([ytr,yv])
    Xa,ya=augment_training(Xtrain,ytrain,42)
    winner=best["model"]
    if best["name"]=="Isolation Forest":
        winner.fit(Xtrain)
    else:
        winner.fit(Xa,ya)

    if hasattr(winner,"predict_proba"):
        p=winner.predict_proba(Xte)[:,1]
        pred=(p>=.5).astype(int)
        f1,rec,auc,score=metrics(yte,pred,p)
        print(f"TEST {best['name']}: F1={f1:.3f} Recall={rec:.3f} AUC={auc:.3f} Score={score:.3f}")

    joblib.dump(winner,ROOT/"models"/"production.joblib")
    with open(ROOT/"models"/"metadata.json","w") as fh:
        json.dump({"model":best["name"],"features":FEATURES},fh,indent=2)

    # ONNX export for supervised sklearn-compatible winner.
    try:
        if hasattr(winner,"predict_proba"):
            sample=np.asarray(Xtrain.iloc[:1],dtype=np.float32)
            onx=to_onnx(winner, sample, target_opset=17)
            with open(ROOT/"models"/"production.onnx","wb") as fh: fh.write(onx.SerializeToString())
            print("ONNX exported.")
    except Exception as e:
        print("ONNX export skipped:",e)

    anomalies=save_anomalies(c,f,winner,best["name"])
    print(f"Winner: {best['name']} | anomalies written: {len(anomalies)}")

if __name__=="__main__":
    main()
