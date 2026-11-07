import json
import pathlib
import pickle
import sqlite3
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import verde as vd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from SigProject import (
    BAND_NAMES,
    BLOCK_SPACING,
    BUFFER_RADIUS,
    GIS_RASTER,
    INNER_SPLITS,
    NODATA,
    OUTER_SPLITS,
    RANDOM_STATE,
    TARGET_BAND,
    apply_buffer,
    balance_indices,
    evaluate_final_model,
    load_raster_dataset,
    nested_spatial_split,
    tune_and_evaluate,
)

WORKDIR = pathlib.Path("../erosion_run_cache")
WORKDIR.mkdir(exist_ok=True)

DATA_NPZ = WORKDIR / "data.npz"
SPLIT_PKL = WORKDIR / "split.pkl"
MODEL_PATH = WORKDIR / "erosion_rf_model.joblib"
BEST_PARAMS_JSON = WORKDIR / "best_params.json"
METRICS_JSON = WORKDIR / "metrics.json"
PRED_CLASS_TIF = WORKDIR / "erosion_prediction_class.tif"
PRED_PROBA_TIF = WORKDIR / "erosion_prediction_probability.tif"
FINGERPRINT_JSON = WORKDIR / "fingerprint.json"
SENS_FOLDS_CSV = WORKDIR / "buffer_sensitivity_folds.csv"
SENS_CSV = WORKDIR / "buffer_sensitivity.csv"

CACHE_FILES = [DATA_NPZ, SPLIT_PKL, MODEL_PATH, BEST_PARAMS_JSON,
               METRICS_JSON, PRED_CLASS_TIF, PRED_PROBA_TIF,
               SENS_FOLDS_CSV, SENS_CSV]

FORCE_RERUN = False

RUN_BUFFER_SENSITIVITY = True
BUFFER_GRID = [0, 500, 1000, 2144, 4000, 6000, 10000]   # metres
SENS_MAX_TRAIN = 200_000   # cap on balanced training pixels per fit


# region cache invalidation
def current_fingerprint():
    raster = pathlib.Path(GIS_RASTER)
    stat = raster.stat()
    return {
        "raster": str(raster),
        "raster_size": stat.st_size,
        "raster_mtime": int(stat.st_mtime),
        "band_names": list(BAND_NAMES),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "block_spacing": BLOCK_SPACING,
        "buffer_radius": BUFFER_RADIUS,
    }


def invalidate_stale_cache():
    fp = current_fingerprint()
    old = json.loads(FINGERPRINT_JSON.read_text()) if FINGERPRINT_JSON.exists() else None
    if FORCE_RERUN or old != fp:
        print("[cache] clearing cached results")
        for p in CACHE_FILES:
            p.unlink(missing_ok=True)
        FINGERPRINT_JSON.write_text(json.dumps(fp, indent=2))
# endregion


class PipelineState(TypedDict, total=False):
    status: str


def node_load_data(state: PipelineState) -> PipelineState:
    if DATA_NPZ.exists():
        print("[load_data] cache hit")
        return {"status": "data_loaded"}

    X, y, coords, predictor_cols, raster, transform, crs, valid_mask = load_raster_dataset(
        GIS_RASTER, BAND_NAMES, TARGET_BAND
    )
    np.savez(
        DATA_NPZ,
        X=X, y=y, coords=coords, valid_mask=valid_mask,
        predictor_cols=np.array(predictor_cols),
    )
    return {"status": "data_loaded"}


def node_spatial_split(state: PipelineState) -> PipelineState:
    if SPLIT_PKL.exists():
        print("[spatial_split] cache hit")
        return {"status": "split_done"}

    print("[spatial_split]")
    d = np.load(DATA_NPZ, allow_pickle=True)
    X, y, coords = d["X"], d["y"], d["coords"]

    X_trainval, y_trainval, X_test, y_test, inner_folds, bal_idx = nested_spatial_split(
        X, y, coords,
        outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
        block_spacing=BLOCK_SPACING, buffer_radius=BUFFER_RADIUS,
    )
    with open(SPLIT_PKL, "wb") as f:
        pickle.dump(
            dict(X_trainval=X_trainval, y_trainval=y_trainval,
                 X_test=X_test, y_test=y_test,
                 inner_folds=inner_folds, bal_idx=bal_idx,
                 predictor_cols=list(d["predictor_cols"])),
            f,
        )
    print(f"[spatial_split] train+val pixels: {len(y_trainval)}, test pixels: {len(y_test)}")
    return {"status": "split_done"}


def node_tune_model(state: PipelineState) -> PipelineState:
    if MODEL_PATH.exists():
        print("[tune_model] cache hit")
        return {"status": "model_trained"}

    print("[tune_model]")
    with open(SPLIT_PKL, "rb") as f:
        split = pickle.load(f)

    best_model, best_params = tune_and_evaluate(
        split["X_trainval"], split["y_trainval"], split["inner_folds"], split["bal_idx"]
    )
    joblib.dump(best_model, MODEL_PATH)
    BEST_PARAMS_JSON.write_text(json.dumps(best_params, indent=2))
    return {"status": "model_trained"}


def node_buffer_sensitivity(state: PipelineState) -> PipelineState:
    if not RUN_BUFFER_SENSITIVITY:
        print("[buffer_sensitivity] disabled")
        return {"status": "sensitivity_skipped"}
    if SENS_CSV.exists():
        print("[buffer_sensitivity] cache hit")
        return {"status": "sensitivity_done"}

    print("[buffer_sensitivity]")
    d = np.load(DATA_NPZ, allow_pickle=True)
    X, y, coords = d["X"], d["y"], d["coords"]
    params = json.loads(BEST_PARAMS_JSON.read_text())

    outer_cv = vd.BlockKFold(spacing=BLOCK_SPACING, n_splits=OUTER_SPLITS,
                             shuffle=True, random_state=RANDOM_STATE)
    folds = list(outer_cv.split(coords))

    rows = []
    for k, (tv0, te) in enumerate(folds):
        if len(np.unique(y[te])) < 2:
            print(f"  fold {k + 1}: single-class test set")
            continue
        for buf in BUFFER_GRID:
            tv, _ = apply_buffer(tv0, te, coords, buf)
            if len(np.unique(y[tv])) < 2:
                print(f"  fold {k + 1}, buffer {buf}")
                continue
            rng = np.random.default_rng(RANDOM_STATE + k)
            bal = balance_indices(tv, y, rng)
            if SENS_MAX_TRAIN and len(bal) > SENS_MAX_TRAIN:
                bal = rng.choice(bal, size=SENS_MAX_TRAIN, replace=False)
            model = RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=-1)
            model.fit(X[bal], y[bal])
            proba = model.predict_proba(X[te])[:, 1]
            rows.append({
                "fold": k + 1,
                "buffer_m": buf,
                "n_train_before_balance": int(len(tv)),
                "n_train": int(len(bal)),
                "n_test": int(len(te)),
                "roc_auc": float(roc_auc_score(y[te], proba)),
                "pr_auc": float(average_precision_score(y[te], proba)),
            })
            print(f"  fold {k + 1}, buffer {buf:>6} m: train {len(tv):>9} | "
                  f"AUC {rows[-1]['roc_auc']:.3f} | PR-AUC {rows[-1]['pr_auc']:.3f}")

    if not rows:
        raise RuntimeError("Buffer sensitivity produced no results")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(SENS_FOLDS_CSV, index=False)
    summary = per_fold.groupby("buffer_m").agg(
        n_folds=("fold", "nunique"),
        mean_train_pixels=("n_train_before_balance", "mean"),
        auc_mean=("roc_auc", "mean"),
        auc_sd=("roc_auc", lambda s: s.std(ddof=0)),
        pr_auc_mean=("pr_auc", "mean"),
    ).reset_index()
    print(summary.round(4).to_string(index=False))
    summary.to_csv(SENS_CSV, index=False)
    return {"status": "sensitivity_done"}


def node_evaluate(state: PipelineState) -> PipelineState:
    if METRICS_JSON.exists():
        print("[evaluate] cache hit")
        return {"status": "evaluated"}

    with open(SPLIT_PKL, "rb") as f:
        split = pickle.load(f)
    d = np.load(DATA_NPZ, allow_pickle=True)
    predictor_cols = list(d["predictor_cols"])
    best_model = joblib.load(MODEL_PATH)

    X_test, y_test = split["X_test"], split["y_test"]
    evaluate_final_model(best_model, X_test, y_test, predictor_cols)

    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_test": int(len(y_test)),
        "test_positive_fraction": float((y_test == 1).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    return {"status": "evaluated"}


def node_predict_raster(state: PipelineState) -> PipelineState:
    if PRED_CLASS_TIF.exists() and PRED_PROBA_TIF.exists():
        print("[predict_raster] cache hit")
        return {"status": "done"}

    d = np.load(DATA_NPZ, allow_pickle=True)
    valid_mask = d["valid_mask"]
    predictor_cols = list(d["predictor_cols"])
    best_model = joblib.load(MODEL_PATH)

    raster = rioxarray.open_rasterio(GIS_RASTER, masked=False)
    n_bands = len(BAND_NAMES)
    X_all = raster.values.reshape(n_bands, -1).T
    df_all = pd.DataFrame(X_all, columns=BAND_NAMES)
    X_all_predictors = df_all[predictor_cols].values

    pred_class_flat = np.full(X_all.shape[0], 255, dtype=np.uint8)      # 255 = nodata
    pred_proba_flat = np.full(X_all.shape[0], -1.0, dtype=np.float32)   # -1 = nodata
    pred_class_flat[valid_mask] = best_model.predict(X_all_predictors[valid_mask]).astype(np.uint8)
    pred_proba_flat[valid_mask] = best_model.predict_proba(X_all_predictors[valid_mask])[:, 1].astype(np.float32)

    height, width = raster.shape[1], raster.shape[2]
    out_meta = dict(driver="GTiff", height=height, width=width, count=1,
                    crs=raster.rio.crs, transform=raster.rio.transform(), compress="LZW")

    with rasterio.open(PRED_CLASS_TIF, "w", dtype="uint8", nodata=255, **out_meta) as dst:
        dst.write(pred_class_flat.reshape(height, width), 1)
    with rasterio.open(PRED_PROBA_TIF, "w", dtype="float32", nodata=-1.0, **out_meta) as dst:
        dst.write(pred_proba_flat.reshape(height, width), 1)

    print(f"[predict_raster] saved -> {PRED_CLASS_TIF}, {PRED_PROBA_TIF}")
    return {"status": "done"}


def build_graph(checkpointer):
    graph = StateGraph(PipelineState)
    graph.add_node("load_data", node_load_data)
    graph.add_node("spatial_split", node_spatial_split)
    graph.add_node("tune_model", node_tune_model)
    graph.add_node("buffer_sensitivity", node_buffer_sensitivity)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("predict_raster", node_predict_raster)

    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "spatial_split")
    graph.add_edge("spatial_split", "tune_model")
    graph.add_edge("tune_model", "buffer_sensitivity")
    graph.add_edge("buffer_sensitivity", "evaluate")
    graph.add_edge("evaluate", "predict_raster")
    graph.add_edge("predict_raster", END)

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    invalidate_stale_cache()

    conn = sqlite3.connect(WORKDIR / "checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_graph(checkpointer)

    THREAD_ID = "erosion_rf_run_1"
    config = {"configurable": {"thread_id": THREAD_ID}}

    final_state = app.invoke({"status": "start"}, config=config)
    print("\nPipeline finished:", final_state)