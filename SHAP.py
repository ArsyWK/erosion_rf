import json
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import verde as vd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier

from SigProject import (
    BLOCK_SPACING,
    BUFFER_RADIUS,
    OUTER_SPLITS,
    RANDOM_STATE,
    apply_buffer,
    balance_indices,
)

# region config
WORKDIR = pathlib.Path("../erosion_run_cache")
DATA_NPZ = WORKDIR / "data.npz"
BEST_PARAMS_JSON = WORKDIR / "best_params.json"
OUT_DIR = WORKDIR / "shap_outputs"
OUT_DIR.mkdir(exist_ok=True)
CACHE_PKL = OUT_DIR / "shap_folds_cache.pkl"
HTML_OUT = OUT_DIR / "feature_effect_explorer.html"

TARGET_NAME = "high-erosion class (multi-year BSI / NDVI anomaly)"
N_EXPLAIN_PER_FOLD = 800
STRATIFY_EXPLAIN = True     # 50/50 positives/negatives False = natural random sample
TRAIN_CAP = 200_000         # max balanced training pixels per fold model
N_BOOTSTRAPS = 1000         # block-bootstrap resamples
TOP_N_DEPENDENCE = 3
PDP_GRID = 25               # points per partial-dependence curve
PDP_ROWS = 400              # explained pixels used to average each PDP point
SCATTER_POINTS = 800        # SHAP points shown in the HTML explorer
DIRECTION_RHO = 0.2         # |Spearman| below this = "mixed / non-monotonic"
FORCE_RECOMPUTE = False     # True = ignore the cache

LABELS = {
    "elevation": "Elevation (m)",
    "slope": "Slope (deg)",
    "aspect_sin": "Aspect: east-west (sin)",
    "aspect_cos": "Aspect: north-south (cos)",
    "twi": "Topographic wetness (TWI)",
    "spi": "Stream power (SPI)",
    "rainfall": "Mean annual rainfall (mm)",
    "plan_curvature": "Plan curvature",
    "profile_curvature": "Profile curvature",
}
# endregion


# region load
d = np.load(DATA_NPZ)
X, y, coords = d["X"], d["y"], d["coords"]
feature_names = [str(c) for c in d["predictor_cols"]]
assert X.shape[1] == len(feature_names), "X columns and feature names disagree"
params = json.loads(BEST_PARAMS_JSON.read_text())
nice = [LABELS.get(n, n) for n in feature_names]
F = len(feature_names)
print("Features:", feature_names)
print("RF params:", params)
# endregion


# region SHAP + PDP per fold
def shap_class1(explainer, xs):
    try:
        sv = explainer(xs)
    except Exception as e:
        print(f"  additivity check failed ({e})")
        sv = explainer(xs, check_additivity=False)
    vals, base = np.asarray(sv.values), np.asarray(sv.base_values)
    if vals.ndim == 3:                                      # (rows, features, classes)
        vals = vals[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    return vals, np.broadcast_to(base, (len(vals),)).astype(float)


def explain_folds():
    outer = vd.BlockKFold(spacing=BLOCK_SPACING, n_splits=OUTER_SPLITS,
                          shuffle=True, random_state=RANDOM_STATE)
    folds = list(outer.split(coords))

    q_rows = np.random.default_rng(RANDOM_STATE).choice(
        len(X), size=min(200_000, len(X)), replace=False)
    grids = [np.unique(np.quantile(X[q_rows, j], np.linspace(0.02, 0.98, PDP_GRID)))
             for j in range(F)]

    parts = {k: [] for k in ("shap", "base", "X", "fold", "coords", "y")}
    pdp = [[] for _ in range(F)]
    used = []

    for k, (tv0, te) in enumerate(folds):
        tv, _ = apply_buffer(tv0, te, coords, BUFFER_RADIUS)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tv])) < 2:
            print("  skipped: a single class in test or train")
            continue

        rng = np.random.default_rng(RANDOM_STATE + k)
        bal = balance_indices(tv, y, rng)
        if TRAIN_CAP and len(bal) > TRAIN_CAP:
            bal = rng.choice(bal, size=TRAIN_CAP, replace=False)
        model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X[bal], y[bal])

        if STRATIFY_EXPLAIN:
            pos, neg = te[y[te] == 1], te[y[te] == 0]
            n_pos = min(len(pos), N_EXPLAIN_PER_FOLD // 2)
            n_neg = min(len(neg), N_EXPLAIN_PER_FOLD - n_pos)
            sel = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                                  rng.choice(neg, size=n_neg, replace=False)])
        else:
            sel = rng.choice(te, size=min(N_EXPLAIN_PER_FOLD, len(te)), replace=False)
        rng.shuffle(sel)
        print(f"  train {len(bal)} px | explaining {len(sel)} test px "
              f"({int(y[sel].sum())} positive)")

        xs = pd.DataFrame(X[sel], columns=feature_names)
        vals, base = shap_class1(shap.TreeExplainer(model), xs)

        parts["shap"].append(vals)
        parts["base"].append(base)
        parts["X"].append(X[sel])
        parts["fold"].append(np.full(len(sel), k + 1))
        parts["coords"].append(coords[sel])
        parts["y"].append(y[sel])
        used.append(k + 1)

        x_pdp = X[sel[:PDP_ROWS]]
        for j in range(F):
            curve = []
            for v in grids[j]:
                xv = x_pdp.copy()
                xv[:, j] = v
                curve.append(float(model.predict_proba(xv)[:, 1].mean() * 100))
            pdp[j].append(curve)

    if not used:
        raise RuntimeError("No fold could be explained")
    res = {k: np.concatenate(v) for k, v in parts.items()}
    res.update(grids=grids, pdp=pdp, folds_used=used)
    return res


fingerprint = dict(n=int(len(y)), params=params, block=BLOCK_SPACING, buffer=BUFFER_RADIUS,
                   n_explain=N_EXPLAIN_PER_FOLD, stratify=STRATIFY_EXPLAIN, cap=TRAIN_CAP,
                   grid=PDP_GRID, pdp_rows=PDP_ROWS, features=feature_names)
res = None
if CACHE_PKL.exists() and not FORCE_RECOMPUTE:
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    if cache.get("fp") == fingerprint:
        res = cache["res"]
        print("reusing SHAP/PDP")
if res is None:
    res = explain_folds()
    with open(CACHE_PKL, "wb") as f:
        pickle.dump({"fp": fingerprint, "res": res}, f)

vals, fold_ids = res["shap"], res["fold"]
K_used = list(res["folds_used"])
n = len(vals)
# endregion


# region importance + block bootstrap + fold stability
abs_vals = np.abs(vals)                                  # abs FIRST, then mean
mean_abs = abs_vals.mean(axis=0)
fold_mean = np.vstack([abs_vals[fold_ids == k].mean(axis=0) for k in K_used])    # (K, F)
fold_sd = fold_mean.std(axis=0, ddof=0)
top3_count = ((-fold_mean).argsort(axis=1).argsort(axis=1) < 3).sum(axis=0)

blk = (res["coords"] // BLOCK_SPACING).astype(np.int64)
cid = pd.Series(list(zip(fold_ids.tolist(), blk[:, 0].tolist(), blk[:, 1].tolist()))).factorize()[0]
C = int(cid.max()) + 1
sum_abs = np.zeros((C, F))
np.add.at(sum_abs, cid, abs_vals)
cnt = np.bincount(cid, minlength=C).astype(float)
rng = np.random.default_rng(RANDOM_STATE)
boot = np.empty((N_BOOTSTRAPS, F))
for b in range(N_BOOTSTRAPS):
    pick = rng.integers(0, C, size=C)
    boot[b] = sum_abs[pick].sum(axis=0) / cnt[pick].sum()
ci_low, ci_high = np.percentile(boot, [2.5, 97.5], axis=0)

direction, rho_all, pear_all, agree = [], [], [], []
for j in range(F):
    col, sv = res["X"][:, j], vals[:, j]
    if np.std(col) == 0 or np.std(sv) == 0:
        direction.append("no effect"); rho_all.append(0.0); pear_all.append(0.0); agree.append("0/0")
        continue
    rho = float(spearmanr(col, sv).correlation)
    rho_all.append(rho)
    pear_all.append(float(pearsonr(col, sv)[0]))
    if abs(rho) < DIRECTION_RHO:
        direction.append("mixed / non-monotonic")
    elif rho > 0:
        direction.append("higher value -> more erosion")
    else:
        direction.append("higher value -> less erosion")
    same = 0
    for k in K_used:
        m = fold_ids == k
        if np.std(col[m]) == 0 or np.std(sv[m]) == 0:
            continue
        rk = spearmanr(col[m], sv[m]).correlation
        if abs(rk) >= DIRECTION_RHO and np.sign(rk) == np.sign(rho):
            same += 1
    agree.append(f"{same}/{len(K_used)}")

result = pd.DataFrame({
    "Feature": nice,
    "Mean |SHAP| (prob. points)": mean_abs,
    "CI 2.5%": ci_low,
    "CI 97.5%": ci_high,
    "SD across folds": fold_sd,
    "Share of total (%)": 100 * mean_abs / mean_abs.sum(),
    "Top-3 in folds": [f"{int(c)}/{len(K_used)}" for c in top3_count],
    "Direction": direction,
    "Spearman rho": rho_all,
    "Pearson r": pear_all,
    "Folds agree on direction": agree,
}).sort_values("Mean |SHAP| (prob. points)", ascending=False).reset_index(drop=True)
result.insert(0, "Rank", result.index + 1)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
print("\n" + result.round(4).to_string(index=False))
result.to_csv(OUT_DIR / "shap_importance.csv", index=False)
# endregion


# region plots
#  Bar chart
order = result.sort_values("Mean |SHAP| (prob. points)")
idxs = [nice.index(f) for f in order["Feature"]]
pos_y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(8, 5))
err = np.vstack([order["Mean |SHAP| (prob. points)"] - order["CI 2.5%"],
                 order["CI 97.5%"] - order["Mean |SHAP| (prob. points)"]])
ax.barh(pos_y, order["Mean |SHAP| (prob. points)"], xerr=err, capsize=3, alpha=0.6)
for kk in range(fold_mean.shape[0]):
    ax.scatter(fold_mean[kk, idxs], pos_y, s=14, color="k", zorder=3,
               label="single fold" if kk == 0 else None)
ax.set_yticks(pos_y)
ax.set_yticklabels(order["Feature"])
ax.set_xlabel("Mean |SHAP| (average change in probability of the high class)")
ax.set_title("Feature importance across spatial folds (95% block-bootstrap CI)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT_DIR / "shap_bar.png", dpi=200)

exp = shap.Explanation(values=vals, base_values=res["base"], data=res["X"], feature_names=nice)

#  Beeswarm
plt.figure()
shap.plots.beeswarm(exp, max_display=F, show=False)
plt.title("How each factor pushes pixels toward the high  or low  class")
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")

# 3)Dependence plots for the top features
for feat in result["Feature"].head(TOP_N_DEPENDENCE):
    plt.figure()
    shap.plots.scatter(exp[:, feat], show=False)
    plt.title(f"Effect of {feat} on the probability of the high class")
    plt.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in feat)
    plt.savefig(OUT_DIR / f"shap_dependence_{safe}.png", dpi=200, bbox_inches="tight")
# endregion




def build_explorer():
    info = result.set_index("Feature")
    sc_idx = np.random.default_rng(RANDOM_STATE).choice(
        n, size=min(SCATTER_POINTS, n), replace=False)
    feats = []
    for j in range(F):
        grid = res["grids"][j]
        if len(grid) < 3:
            continue
        r = info.loc[nice[j]]
        feats.append({
            "name": nice[j],
            "grid": np.round(grid, 5).tolist(),
            "pdp": np.round(np.array(res["pdp"][j]), 3).tolist(),
            "median": float(np.median(X[:, j])),
            "sx": np.round(res["X"][sc_idx, j], 5).tolist(),
            "sy": np.round(vals[sc_idx, j] * 100, 3).tolist(),
            "importance": round(float(r["Mean |SHAP| (prob. points)"]) * 100, 2),
            "share": round(float(r["Share of total (%)"]), 1),
            "direction": str(r["Direction"]).replace("->", "→"),
            "rho": round(float(r["Spearman rho"]), 2),
            "pearson": round(float(r["Pearson r"]), 2),
            "agree": str(r["Folds agree on direction"]),
            "rank": int(r["Rank"]),
        })
    feats.sort(key=lambda f: f["rank"])
    payload = {"target": TARGET_NAME, "k": len(K_used), "features": feats}


build_explorer()
plt.show()
print(f"\nSaved table and figures to {OUT_DIR.resolve()}")
# endregion
