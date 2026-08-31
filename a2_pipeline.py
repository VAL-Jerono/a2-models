import re
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                              precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import optuna
import shap
import json
import os

os.makedirs("results", exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RNG = 42
COST_FN = 10000  # missed fraud (false negative)
COST_FP = 800    # wrongly flagged good txn (false positive)

# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN  (this stands in for "Assignment 1 output")
# ---------------------------------------------------------------------------
print("Loading...")
df = pd.read_csv("/mnt/user-data/uploads/mobile_money_statements.csv")

def parse_amount(s):
    s = str(s).strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace(",", "").replace("/-", "")
    is_debit = "Dr" in s
    s = s.replace("Dr", "").strip()
    try:
        v = float(s)
    except ValueError:
        return np.nan, np.nan
    if neg and v > 0:
        v = -v
    signed = -abs(v) if is_debit else v
    return abs(v), signed

parsed = df["amount"].apply(parse_amount)
df["amount_abs"] = parsed.apply(lambda x: x[0])
df["amount_signed"] = parsed.apply(lambda x: x[1])

def parse_time(s):
    s = str(s).strip()
    fmts = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%b %d, %Y %I:%M %p"]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return pd.NaT

print("Parsing timestamps...")
df["txn_dt"] = df["txn_time"].apply(parse_time)
n_bad_time = df["txn_dt"].isna().sum()
df = df.dropna(subset=["txn_dt"]).reset_index(drop=True)
print(f"Dropped {n_bad_time} rows with unparseable timestamps")

df["reg_id"] = df["reg_id"].str.upper()
df["hour"] = df["txn_dt"].dt.hour
df["dow"] = df["txn_dt"].dt.dayofweek
df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
df["is_night"] = df["hour"].between(0, 4).astype(int)
df["gps_missing"] = df["gps_lat"].isna().astype(int)
df["agent_missing"] = df["agent_id"].isna().astype(int)
df["device_missing"] = df["device_id"].isna().astype(int)

# frequency encodings (counts as of full history -- see leakage note in report)
for col in ["agent_id", "counterparty", "device_id", "msisdn"]:
    freq = df[col].value_counts()
    df[col + "_freq"] = df[col].map(freq).fillna(0)

df = df.sort_values("txn_dt").reset_index(drop=True)

# LEAKAGE CHECK: manual_review_score and settlement_status are produced
# *after* a transaction is investigated / settled -> they are outcome
# proxies, not features available at scoring time. We exclude them from
# the model and instead demonstrate the leakage empirically below.
leak_cols = ["manual_review_score", "settlement_status"]

feature_num = ["amount_abs", "amount_signed", "balance_after", "hour", "dow",
               "is_weekend", "is_night", "gps_missing", "agent_missing",
               "device_missing", "agent_id_freq", "counterparty_freq",
               "device_id_freq", "msisdn_freq"]
feature_cat = ["region", "segment", "txn_type"]

X = df[feature_num + feature_cat].copy()
y = df["is_fraud"].values
base_rate = y.mean()
print(f"Base fraud rate: {base_rate:.4f}, n={len(y)}")

pre = ColumnTransformer([
    ("num", StandardScaler(), feature_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cat),
])

# ---------------------------------------------------------------------------
# 2. BLOCKED / PURGED TIME-SERIES SPLIT
# ---------------------------------------------------------------------------
n = len(df)
n_train = int(n * 0.6)
n_gap = int(n * 0.02)
n_val = int(n * 0.19)
train_idx = np.arange(0, n_train)
val_idx = np.arange(n_train + n_gap, n_train + n_gap + n_val)
test_idx = np.arange(n_train + n_gap + n_val + n_gap, n)
print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} (gaps purge boundary leakage)")

X_train, y_train = X.iloc[train_idx], y[train_idx]
X_val, y_val = X.iloc[val_idx], y[val_idx]
X_test, y_test = X.iloc[test_idx], y[test_idx]

# ---------------------------------------------------------------------------
# 3. LADDER OF MODELS (same folds)
# ---------------------------------------------------------------------------
results = []

def eval_model(name, pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    pr_auc = average_precision_score(y_te, p)
    roc_auc = roc_auc_score(y_te, p)
    results.append({"model": name, "pr_auc": pr_auc, "roc_auc": roc_auc})
    return pipe, p

print("Training logistic regression...")
logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
logit_fit, p_logit = eval_model("Logistic Regression (balanced)", logit, X_train, y_train, X_val, y_val)

print("Training random forest...")
rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                                             class_weight="balanced_subsample",
                                                             n_jobs=-1, random_state=RNG))])
rf_fit, p_rf = eval_model("Random Forest", rf, X_train, y_train, X_val, y_val)

print("Training LightGBM (default params)...")
lgbm = Pipeline([("pre", pre), ("clf", lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                                            random_state=RNG, verbosity=-1))])
lgbm_fit, p_lgbm = eval_model("LightGBM (default)", lgbm, X_train, y_train, X_val, y_val)

print(pd.DataFrame(results))

# ---------------------------------------------------------------------------
# 4. IMBALANCE HANDLING: class weighting vs SMOTE, in-fold vs leaked
# ---------------------------------------------------------------------------
print("Imbalance comparison...")

# (a) correct: SMOTE applied only inside the training fold
imb_pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RNG)),
                         ("clf", lgb.LGBMClassifier(n_estimators=300, random_state=RNG, verbosity=-1))])
imb_pipe.fit(X_train, y_train)
p_smote_infold = imb_pipe.predict_proba(X_val)[:, 1]
pr_smote_infold = average_precision_score(y_val, p_smote_infold)

# (b) WRONG: SMOTE applied before the split (fit on train+val together) -> leakage demo
X_leak = pd.concat([X_train, X_val])
y_leak = np.concatenate([y_train, y_val])
pre_leak = ColumnTransformer([
    ("num", StandardScaler(), feature_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cat),
])
Xt_leak = pre_leak.fit_transform(X_leak)
sm = SMOTE(random_state=RNG)
Xres, yres = sm.fit_resample(Xt_leak, y_leak)
clf_leak = lgb.LGBMClassifier(n_estimators=300, random_state=RNG, verbosity=-1)
clf_leak.fit(Xres, yres)
# evaluate on the SAME validation rows (already resampled into the mix -> optimistic)
Xt_val_leak = pre_leak.transform(X_val)
p_leak = clf_leak.predict_proba(Xt_val_leak)[:, 1]
pr_leak = average_precision_score(y_val, p_leak)

imbalance_results = pd.DataFrame([
    {"approach": "class_weight='balanced' (LightGBM)", "pr_auc_val": results[2]["pr_auc"]},
    {"approach": "SMOTE inside training fold only (correct)", "pr_auc_val": pr_smote_infold},
    {"approach": "SMOTE fit before split, incl. val rows (LEAKED)", "pr_auc_val": pr_leak},
])
print(imbalance_results)

# ---------------------------------------------------------------------------
# 5. OPTUNA TUNING (LightGBM, budget-limited)
# ---------------------------------------------------------------------------
print("Optuna tuning...")
Xt_train = pre.fit_transform(X_train)
Xt_val = pre.transform(X_val)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": RNG,
        "verbosity": -1,
    }
    m = lgb.LGBMClassifier(**params)
    m.fit(Xt_train, y_train)
    p = m.predict_proba(Xt_val)[:, 1]
    return average_precision_score(y_val, p)

study = optuna.create_study(direction="maximize", study_name="a2_lgbm",
                             storage="sqlite:///results/optuna_study.db", load_if_exists=True,
                             pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=60, show_progress_bar=False)
print("Best PR-AUC (val):", study.best_value)
print("Best params:", study.best_params)

best_params = study.best_params.copy()
best_params.update({"class_weight": "balanced", "random_state": RNG, "verbosity": -1})
final_model_pre = pre
final_clf = lgb.LGBMClassifier(**best_params)
Xt_trainval = pre.transform(pd.concat([X_train, X_val]))
y_trainval = np.concatenate([y_train, y_val])
final_clf.fit(Xt_trainval, y_trainval)
Xt_test = pre.transform(X_test)
p_test_raw = final_clf.predict_proba(Xt_test)[:, 1]
pr_auc_test = average_precision_score(y_test, p_test_raw)
roc_auc_test = roc_auc_score(y_test, p_test_raw)
print(f"Tuned LightGBM on held-out test: PR-AUC={pr_auc_test:.4f} ROC-AUC={roc_auc_test:.4f}")

# ---------------------------------------------------------------------------
# 6. COST-BASED THRESHOLD (pre-calibration)
# ---------------------------------------------------------------------------
def cost_at_threshold(y_true, p, t, cost_fn=COST_FN, cost_fp=COST_FP):
    pred = (p >= t).astype(int)
    fn = ((pred == 0) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    return fn * cost_fn + fp * cost_fp, fn, fp

thresholds = np.linspace(0.01, 0.99, 197)
costs = [cost_at_threshold(y_test, p_test_raw, t)[0] for t in thresholds]
best_t_raw = thresholds[int(np.argmin(costs))]
best_cost_raw = min(costs)
cost_at_half, fn_half, fp_half = cost_at_threshold(y_test, p_test_raw, 0.5)
n_apps = len(y_test)
savings_per_1000_raw = (cost_at_half - best_cost_raw) / n_apps * 1000
print(f"[Pre-calibration] best t={best_t_raw:.3f} cost={best_cost_raw:.0f}  vs 0.5 cost={cost_at_half:.0f}  "
      f"savings/1000={savings_per_1000_raw:.1f}")

# ---------------------------------------------------------------------------
# 7. CALIBRATION
# ---------------------------------------------------------------------------
print("Calibration...")
brier_raw = brier_score_loss(y_test, p_test_raw)

from sklearn.frozen import FrozenEstimator
calibrator = CalibratedClassifierCV(FrozenEstimator(final_clf), method="isotonic")
# calibrate on validation set (kept separate from test)
calibrator.fit(Xt_val, y_val)
p_test_cal = calibrator.predict_proba(Xt_test)[:, 1]
brier_cal = brier_score_loss(y_test, p_test_cal)

frac_pos_raw, mean_pred_raw = calibration_curve(y_test, p_test_raw, n_bins=10, strategy="quantile")
frac_pos_cal, mean_pred_cal = calibration_curve(y_test, p_test_cal, n_bins=10, strategy="quantile")

costs_cal = [cost_at_threshold(y_test, p_test_cal, t)[0] for t in thresholds]
best_t_cal = thresholds[int(np.argmin(costs_cal))]
best_cost_cal = min(costs_cal)
savings_per_1000_cal = (cost_at_threshold(y_test, p_test_cal, 0.5)[0] - best_cost_cal) / n_apps * 1000

print(f"Brier raw={brier_raw:.4f} calibrated={brier_cal:.4f}")
print(f"[Post-calibration] best t={best_t_cal:.3f} cost={best_cost_cal:.0f} savings/1000={savings_per_1000_cal:.1f}")

# ---------------------------------------------------------------------------
# 8. SHAP (global + local)
# ---------------------------------------------------------------------------
print("SHAP...")
feat_names = (feature_num +
              list(pre.named_transformers_["cat"].get_feature_names_out(feature_cat)))
explainer = shap.TreeExplainer(final_clf)
sample_idx = np.random.RandomState(RNG).choice(len(X_test), size=min(2000, len(X_test)), replace=False)
Xt_test_sample = Xt_test[sample_idx] if not hasattr(Xt_test, "toarray") else Xt_test[sample_idx].toarray()
shap_values = explainer.shap_values(Xt_test_sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_shap}) \
    .sort_values("mean_abs_shap", ascending=False).head(15)
print(top_features)

# a single local explanation (highest-risk test transaction)
local_idx = int(np.argmax(p_test_cal))
local_row = Xt_test[local_idx] if not hasattr(Xt_test, "toarray") else Xt_test[local_idx].toarray()
local_shap = explainer.shap_values(local_row.reshape(1, -1))
if isinstance(local_shap, list):
    local_shap = local_shap[1]
local_expl = pd.DataFrame({"feature": feat_names, "shap_value": local_shap[0]}) \
    .sort_values("shap_value", key=abs, ascending=False).head(8)
print(local_expl)

# ---------------------------------------------------------------------------
# 9. FAIRNESS / SUBGROUP METRICS
# ---------------------------------------------------------------------------
print("Fairness...")
df_test = df.iloc[test_idx].copy()
df_test["p_cal"] = p_test_cal
df_test["pred"] = (df_test["p_cal"] >= best_t_cal).astype(int)
df_test["y_true"] = y_test

def subgroup_metrics(frame, col):
    rows = []
    for g, sub in frame.groupby(col):
        approval_rate = 1 - sub["pred"].mean()  # "approved" = not flagged
        fn = ((sub["pred"] == 0) & (sub["y_true"] == 1)).sum()
        pos = (sub["y_true"] == 1).sum()
        fnr = fn / pos if pos > 0 else np.nan
        rows.append({"group": g, "n": len(sub), "approval_rate": approval_rate, "fnr": fnr})
    return pd.DataFrame(rows)

fairness_region = subgroup_metrics(df_test, "region")
fairness_segment = subgroup_metrics(df_test, "segment")
print(fairness_region)
print(fairness_segment)

# ---------------------------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------------------------
pd.DataFrame(results).to_csv("results/model_comparison.csv", index=False)
imbalance_results.to_csv("results/imbalance_comparison.csv", index=False)
top_features.to_csv("results/shap_global_top15.csv", index=False)
local_expl.to_csv("results/shap_local_top_risk_txn.csv", index=False)
fairness_region.to_csv("results/fairness_region.csv", index=False)
fairness_segment.to_csv("results/fairness_segment.csv", index=False)

summary = {
    "base_rate": float(base_rate),
    "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
    "tuned_lgbm_test_pr_auc": float(pr_auc_test),
    "tuned_lgbm_test_roc_auc": float(roc_auc_test),
    "best_params": study.best_params,
    "pre_calibration": {"best_threshold": float(best_t_raw), "cost_at_best": float(best_cost_raw),
                         "cost_at_0.5": float(cost_at_half), "savings_per_1000": float(savings_per_1000_raw)},
    "brier_raw": float(brier_raw), "brier_calibrated": float(brier_cal),
    "post_calibration": {"best_threshold": float(best_t_cal), "cost_at_best": float(best_cost_cal),
                          "savings_per_1000": float(savings_per_1000_cal)},
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

np.save("results/calibration_curve_raw.npy", np.array([mean_pred_raw, frac_pos_raw]))
np.save("results/calibration_curve_cal.npy", np.array([mean_pred_cal, frac_pos_cal]))
np.save("results/pr_curve_test.npy", np.array(precision_recall_curve(y_test, p_test_cal)[:2], dtype=object))
np.save("results/cost_curve.npy", np.array([thresholds, costs, costs_cal]))

# ---------------------------------------------------------------------------
# 10. FIGURES (black and white, print-safe)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, costs, color="black", linestyle="-", linewidth=1.3, label="Raw scores")
ax.plot(thresholds, costs_cal, color="black", linestyle="--", linewidth=1.3, label="Calibrated scores")
ax.axvline(best_t_raw, color="black", linestyle=":", linewidth=0.9)
ax.axvline(best_t_cal, color="black", linestyle=":", linewidth=0.9)
ax.axvline(0.5, color="grey", linestyle="-.", linewidth=0.8, label="Naive 0.5 cutoff")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Total cost on test set (KES)")
ax.set_title("Cost vs threshold, pre- and post-calibration")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("results/cost_curve.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, label="Perfect calibration")
ax.plot(mean_pred_raw, frac_pos_raw, color="black", marker="o", markersize=4,
        linestyle="-", linewidth=1.1, label="Raw LightGBM")
ax.plot(mean_pred_cal, frac_pos_cal, color="black", marker="s", markersize=4,
        linestyle="--", linewidth=1.1, label="Isotonic-calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraud rate")
ax.set_title("Reliability diagram (test set)")
ax.legend(frameon=False, fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("results/calibration_curve.png", dpi=200)
plt.close(fig)

print("DONE")
