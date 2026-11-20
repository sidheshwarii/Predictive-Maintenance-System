"""
train_industrial.py  -  Dataset 1: Industrial IoT (canozensoy), 500k rows.
Task: binary classification, will the machine fail within 7 days?
Outputs: model/industrial_model.json  (compact forest the website runs in-browser)
         model/industrial_rf.pkl      (same model for the FastAPI backend)
"""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

DATA = "factory_sensor_simulator_2040.csv"
TARGET = "Failure_Within_7_Days"
LEAK = ["Remaining_Useful_Life_days"]          # would give the answer away
ID_COLS = ["Machine_ID"]                        # exact-name match (earlier bug dropped AI_Override_Events)

df = pd.read_csv(DATA)
y = df[TARGET].astype(int)
df = df.drop(columns=[TARGET] + LEAK + ID_COLS)
# drop columns that are >90% empty (machine-specific sensors)
df = df.loc[:, df.isna().mean() < 0.9]
df["AI_Supervision"] = df["AI_Supervision"].astype(int)
types = sorted(df["Machine_Type"].unique())
df["Machine_Type"] = df["Machine_Type"].map({t: i for i, t in enumerate(types)})
df = df.fillna(df.median(numeric_only=True))
features = list(df.columns)
print("features:", features)

Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)
rf = RandomForestClassifier(n_estimators=60, max_depth=10, min_samples_leaf=30,
                            class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(Xtr, ytr)
p = rf.predict_proba(Xte)[:, 1]
auc = roc_auc_score(yte, p)
rep = classification_report(yte, (p >= .5).astype(int), output_dict=True)
print("AUC", auc); print(classification_report(yte, (p >= .5).astype(int)))

def tree_json(t):
    T = t.tree_
    val = T.value[:, 0, :]; prob = (val[:, 1] / val.sum(1)).round(4)
    return {"f": T.feature.tolist(), "t": [round(float(x), 4) for x in T.threshold],
            "l": T.children_left.tolist(), "r": T.children_right.tolist(), "p": prob.tolist()}

ranges = {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in features}
imp = dict(sorted(zip(features, rf.feature_importances_.round(4).tolist()), key=lambda kv: -kv[1]))
out = {"dataset": "industrial_iot", "features": features, "machine_types": types, "ranges": ranges,
       "importance": imp, "metrics": {"roc_auc": round(auc, 4), "accuracy": round(rep["accuracy"], 4),
       "fail_precision": round(rep["1"]["precision"], 3), "fail_recall": round(rep["1"]["recall"], 3),
       "rows": int(len(df)), "fail_rate": round(float(y.mean()), 4)},
       "trees": [tree_json(e) for e in rf.estimators_]}
json.dump(out, open("model/industrial_model.json", "w"), separators=(",", ":"))
joblib.dump(rf, "model/industrial_rf.pkl", compress=3)
