import pathlib
import pickle

import joblib
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import verde as vd
from rasterio.transform import xy
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

# region config
GIS_RASTER = pathlib.Path(r"C:\Users\freya\Downloads\GIS\tapa\Erosion\tapanuli_tengah_noRUSLE_erosion_30m.tif")

BAND_NAMES = ["elevation", "slope", "aspect_sin", "aspect_cos", "twi", "spi",
              "rainfall", "plan_curvature", "profile_curvature", "erosion_class"]
TARGET_BAND = "erosion_class"
PREDICTOR_BANDS = [b for b in BAND_NAMES if b != TARGET_BAND]

NODATA = -9999

OUTER_SPLITS = 5
INNER_SPLITS = 4
BLOCK_SPACING = 2810.9259980242846         # metres before 10000

BUFFER_RADIUS = 1405.4629990121423# before 2144
RANDOM_STATE = 42

WORKDIR = pathlib.Path("../erosion_run_cache")
WORKDIR.mkdir(exist_ok=True)
MODEL_PATH = WORKDIR / "erosion_rf_model.joblib"
SPLIT_PKL = WORKDIR / "split.pkl"
# endregion


# region load
def load_raster_dataset(raster_path, band_names, target_band):
    raster = rioxarray.open_rasterio(raster_path, masked=False)   # (bands, rows, cols)
    n_bands = raster.shape[0]
    if n_bands != len(band_names):
        raise ValueError(f"Bands not sufficient")

    transform = raster.rio.transform()
    crs = raster.rio.crs
    if crs is not None and crs.is_geographic:
        print("need a projected CRS")

    array = raster.values
    rows, cols = array.shape[1], array.shape[2]
    flat = array.reshape(n_bands, -1).T
    df = pd.DataFrame(flat, columns=band_names)

    valid_mask = (df != NODATA).all(axis=1) & np.isfinite(df).all(axis=1)
    valid_mask = valid_mask.values

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    easting, northing = xy(transform, row_idx.flatten(), col_idx.flatten())
    coords_all = np.column_stack([easting, northing])

    predictor_cols = [b for b in band_names if b != target_band]
    X = df.loc[valid_mask, predictor_cols].values
    y = df.loc[valid_mask, target_band].values.astype(int)
    coords = coords_all[valid_mask]

    print(f"{len(y)} valid ROI pixels "
          f"({int((y == 1).sum())} positive / {int((y == 0).sum())} negative)")

    return X, y, coords, predictor_cols, raster, transform, crs, valid_mask
# endregion


# region Balancing + spatial split
def balance_indices(idx, y, rng):
    idx = np.asarray(idx)
    labels = y[idx]
    pos = idx[labels == 1]
    neg = idx[labels == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError("one class is missing from this training set")
    out = np.concatenate([
        rng.choice(pos, size=n, replace=False),
        rng.choice(neg, size=n, replace=False),
    ])
    rng.shuffle(out)
    return out


def apply_buffer(train_idx, test_idx, coords, buffer_radius):
    if buffer_radius <= 0:
        return train_idx, test_idx

    test_tree = cKDTree(coords[test_idx])
    dist_to_test, _ = test_tree.query(coords[train_idx], k=1)
    return train_idx[dist_to_test >= buffer_radius], test_idx


def nested_spatial_split(X, y, coords, outer_splits, inner_splits, block_spacing,
                         buffer_radius, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)


    outer_cv = vd.BlockKFold(spacing=block_spacing, n_splits=outer_splits,
                             shuffle=True, random_state=random_state)
    trainval_idx, test_idx = next(iter(outer_cv.split(coords)))
    n_before = len(trainval_idx)
    trainval_idx, test_idx = apply_buffer(trainval_idx, test_idx, coords, buffer_radius)
    print(f"Outer buffer: train+val {n_before} -> {len(trainval_idx)} pixels "
          f"({len(test_idx)} test pixels)")
    if len(trainval_idx) < 0.3 * n_before:
        print("WARNING: buffer removed most training data; reduce BUFFER_RADIUS")

    X_test, y_test = X[test_idx], y[test_idx]
    X_trainval, y_trainval = X[trainval_idx], y[trainval_idx]
    coords_trainval = coords[trainval_idx]


    inner_cv = vd.BlockKFold(spacing=block_spacing, n_splits=inner_splits,
                             shuffle=True, random_state=random_state)
    inner_folds = []
    for tr, va in inner_cv.split(coords_trainval):
        tr, va = apply_buffer(tr, va, coords_trainval, buffer_radius)
        if len(np.unique(y_trainval[va])) < 2:
            print("WARNING: an inner validation fold has a single class; F1 will be unreliable")
        tr = balance_indices(tr, y_trainval, rng)
        inner_folds.append((tr, va))


    bal_idx = balance_indices(np.arange(len(y_trainval)), y_trainval, rng)

    print(f"Test set (natural): {int((y_test == 1).sum())} pos / {int((y_test == 0).sum())} neg")
    print(f"Final training set (balanced): {len(bal_idx)} pixels, "
          f"{int((y_trainval[bal_idx] == 1).sum())} pos / {int((y_trainval[bal_idx] == 0).sum())} neg")

    return X_trainval, y_trainval, X_test, y_test, inner_folds, bal_idx
# endregion


# region Tuning & evaluate
def tune_and_evaluate(X_trainval, y_trainval, inner_folds, bal_idx):
    param_grid = {
        "n_estimators": [100, 150, 180],
        "max_depth": [None, 5, 10],
        "min_samples_leaf": [1, 5, 8],
    }

    # refit=False: the default refit would train on the UNBALANCED train+val set.
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=inner_folds,
        scoring="f1",
        n_jobs=-1,
        refit=False,
    )
    grid.fit(X_trainval, y_trainval)

    print("Best params:", grid.best_params_)
    print("Best spatial validation F1:", grid.best_score_)

    final_model = RandomForestClassifier(
        **grid.best_params_, random_state=RANDOM_STATE, n_jobs=-1
    )
    final_model.fit(X_trainval[bal_idx], y_trainval[bal_idx])
    return final_model, grid.best_params_


def evaluate_final_model(model, X_test, y_test, feature_names):
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    print(f"Accuracy:          {accuracy_score(y_test, test_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
    print(f"Precision:         {precision_score(y_test, test_pred):.4f}")
    print(f"Recall:            {recall_score(y_test, test_pred):.4f}")
    print(f"F1 score:          {f1_score(y_test, test_pred):.4f}")
    print(f"ROC AUC:           {roc_auc_score(y_test, test_proba):.4f}")
    print(f"PR AUC (AP):       {average_precision_score(y_test, test_proba):.4f}")

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, test_pred, target_names=["Low erosion", "High erosion"]))

    print("\nFeature importances:")
    for name, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")
# endregion


# if __name__ == "__main__":
#     X, y, coords, predictor_cols, raster, transform, crs, valid_mask = load_raster_dataset(
#         GIS_RASTER, BAND_NAMES, TARGET_BAND
#     )
#
#     X_trainval, y_trainval, X_test, y_test, inner_folds, bal_idx = nested_spatial_split(
#         X, y, coords,
#         outer_splits=OUTER_SPLITS, inner_splits=INNER_SPLITS,
#         block_spacing=BLOCK_SPACING, buffer_radius=BUFFER_RADIUS,
#     )
#
#     best_model, best_params = tune_and_evaluate(X_trainval, y_trainval, inner_folds, bal_idx)
#     evaluate_final_model(best_model, X_test, y_test, predictor_cols)
#
#     # Save for the SHAP script
#     joblib.dump(best_model, MODEL_PATH)
#     with open(SPLIT_PKL, "wb") as f:
#         pickle.dump({"X_test": X_test, "y_test": y_test, "predictor_cols": predictor_cols}, f)
#     print(f"\nSaved model -> {MODEL_PATH}\nSaved split -> {SPLIT_PKL}")
#
#     # Susceptibility maps for every valid pixel
#     n_bands = len(BAND_NAMES)
#     X_all = raster.values.reshape(n_bands, -1).T
#     df_all = pd.DataFrame(X_all, columns=BAND_NAMES)
#     X_all_predictors = df_all[predictor_cols].values
#
#     pred_class_flat = np.full(X_all.shape[0], 255, dtype=np.uint8)      # 255 = nodata
#     pred_proba_flat = np.full(X_all.shape[0], -1.0, dtype=np.float32)   # -1 = nodata
#
#     pred_class_flat[valid_mask] = best_model.predict(X_all_predictors[valid_mask]).astype(np.uint8)
#     pred_proba_flat[valid_mask] = best_model.predict_proba(X_all_predictors[valid_mask])[:, 1].astype(np.float32)
#
#     height, width = raster.shape[1], raster.shape[2]
#     pred_class_map = pred_class_flat.reshape(height, width)
#     pred_proba_map = pred_proba_flat.reshape(height, width)
#
#     out_meta = dict(driver="GTiff", height=height, width=width, count=1,
#                     crs=crs, transform=transform, compress="LZW")
#
#     with rasterio.open("erosion_susceptibility_class.tif", "w", dtype="uint8", nodata=255, **out_meta) as dst:
#         dst.write(pred_class_map, 1)
#     with rasterio.open("erosion_susceptibility.tif", "w", dtype="float32", nodata=-1.0, **out_meta) as dst:
#         dst.write(pred_proba_map, 1)
#
#     print(f"High-erosion pixels: {(pred_class_map == 1).sum()}, "
#           f"low-erosion: {(pred_class_map == 0).sum()}")