import sys, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

D = sys.argv[1] if len(sys.argv) > 1 else "data/azure"
rd = lambda n: pd.read_csv(f"{D}/{n}.csv", parse_dates=["datetime"] if n != "PdM_machines" else None)
tel, err, mnt, fail, mac = map(rd, ["PdM_telemetry", "PdM_errors", "PdM_maint", "PdM_failures", "PdM_machines"])
tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
sens = ["volt", "rotate", "pressure", "vibration"]

g = tel.groupby("machineID")[sens]
for w in (3, 24):
    r = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    tel[[f"{s}_{w}h" for s in sens]] = r.values

# error counts in the last 24h (per error type)
e = pd.crosstab([err["machineID"], err["datetime"].dt.floor("h")], err["errorID"]).reset_index()
e.columns.name = None
e = e.rename(columns={"datetime": "datetime"})
tel = tel.merge(e, on=["machineID", "datetime"], how="left")
errcols = [f"error{i}" for i in range(1, 6)]
for c in errcols:
    if c not in tel: tel[c] = 0
    tel[c] = tel[c].fillna(0)
    tel[c] = tel.groupby("machineID")[c].transform(lambda s: s.rolling(24, min_periods=1).sum())

# days since last maintenance per component
for i in range(1, 5):
    comp = f"comp{i}"
    m = mnt[mnt["comp"] == comp][["machineID", "datetime"]].assign(last=lambda x: x["datetime"]).sort_values("datetime")
    tel = pd.merge_asof(tel.sort_values("datetime"), m, on="datetime", by="machineID", direction="backward")
    tel[f"days_since_{comp}"] = ((tel["datetime"] - tel["last"]).dt.total_seconds() / 86400).fillna(365).clip(upper=365)
    tel = tel.drop(columns="last")
tel = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)

tel = tel.merge(mac, on="machineID", how="left")
models = sorted(tel["model"].unique())
tel["model"] = tel["model"].map({m: i for i, m in enumerate(models)})

# label: any failure in the next 24 hours
f = fail[["machineID", "datetime"]].copy(); f["datetime"] = f["datetime"].dt.floor("h")
tel["y"] = 0
key = set(zip(f["machineID"], f["datetime"]))
fh = pd.DataFrame(list(key), columns=["machineID", "datetime"])
for k in range(0, 24):
    sh = fh.assign(datetime=fh["datetime"] - pd.Timedelta(hours=k))
    tel = tel.merge(sh.assign(_h=1), on=["machineID", "datetime"], how="left")
    tel["y"] = np.maximum(tel["y"], tel["_h"].fillna(0).astype(int)); tel = tel.drop(columns="_h")

features = [f"{s}_{w}h" for w in (3, 24) for s in sens] + errcols + [f"days_since_comp{i}" for i in range(1, 5)] + ["age", "model"]
cut = tel["datetime"].max() - pd.Timedelta(days=90)
tr, te = tel[tel["datetime"] < cut], tel[tel["datetime"] >= cut]
print("rows", len(tel), "fail-rate", tel["y"].mean().round(4), "train/test", len(tr), len(te))
rf = RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=30,
                            class_weight="balanced", n_jobs=-1, random_state=42).fit(tr[features], tr["y"])
p = rf.predict_proba(te[features])[:, 1]
auc = roc_auc_score(te["y"], p) if te["y"].nunique() > 1 else float("nan")
rep = classification_report(te["y"], (p >= .5).astype(int), output_dict=True, zero_division=0)
print("AUC", auc)

def tree_json(t):
    T = t.tree_; val = T.value[:, 0, :]
    return {"f": T.feature.tolist(), "t": [round(float(x), 4) for x in T.threshold],
            "l": T.children_left.tolist(), "r": T.children_right.tolist(),
            "p": (val[:, 1] / val.sum(1)).round(4).tolist()}

out = {"dataset": "azure_pdm", "features": features, "machine_types": [str(m) for m in models],
       "ranges": {c: [float(tel[c].min()), float(tel[c].max()), float(tel[c].median())] for c in features},
       "importance": dict(sorted(zip(features, rf.feature_importances_.round(4).tolist()), key=lambda kv: -kv[1])),
       "metrics": {"roc_auc": round(float(auc), 4), "accuracy": round(rep["accuracy"], 4),
                   "fail_precision": round(rep.get("1", {}).get("precision", 0), 3),
                   "fail_recall": round(rep.get("1", {}).get("recall", 0), 3),
                   "rows": int(len(tel)), "fail_rate": round(float(tel["y"].mean()), 4)},
       "trees": [tree_json(e) for e in rf.estimators_]}
json.dump(out, open("model/azure_model.json", "w"), separators=(",", ":"))
joblib.dump(rf, "model/azure_rf.pkl", compress=3)
print("saved model/azure_model.json")
