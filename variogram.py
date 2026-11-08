import json
import pathlib

import numpy as np
import pandas as pd
import rioxarray
import skgstat as skg
from rasterio.transform import xy

# region config
GIS_RASTER = pathlib.Path(r"YOUR_TIF_FILE")

BAND_NAMES = ["elevation", "slope", "aspect_sin", "aspect_cos", "twi", "spi",
              "rainfall", "plan_curvature", "profile_curvature", "erosion_class"]
TARGET_BAND = "erosion_class"
PREDICTOR_BANDS = [b for b in BAND_NAMES if b != TARGET_BAND]
NODATA = -9999

SAMPLE_SIZE = 4000
N_REPEATS = 3 # repeat with random state to ensure single random noise doesnt ruin ranges estimate
MODEL = "spherical"
N_LAGS = 20
MAXLAGS = [3000, 6000, 12000, 24000] # max step of observed array pos
DETREND = True
HIT_BOUND_FRACTION = 0.9 # if hit bound uncorrelated outside observed point
RANDOM_STATE = 42
# endregion


def load_bands(raster_path, band_names):
    raster = rioxarray.open_rasterio(raster_path, masked=False)
    n_bands = raster.shape[0]
    if n_bands != len(band_names):
        raise ValueError(f"Raster has {n_bands} bands but BAND_NAMES has {len(band_names)} entries")

    transform = raster.rio.transform()
    rows, cols = raster.shape[1], raster.shape[2]
    flat = raster.values.reshape(n_bands, -1).T
    df = pd.DataFrame(flat, columns=band_names)


    valid = (df != NODATA).all(axis=1) & np.isfinite(df).all(axis=1)
    valid = valid.values

    row_idx, col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    easting, northing = xy(transform, row_idx.flatten(), col_idx.flatten())
    coords_all = np.column_stack([easting, northing])

    return df[valid].reset_index(drop=True), coords_all[valid]

# tips -- removing macroregional can give variance X
# short explanation -- macroregional trends means observing point A to B not A and B the middle get caught
def remove_linear_trend(values, coords):
    """ using ols to remove macroregional trends from tampering the local spatial autocorrelation """
    A = np.column_stack([np.ones(len(coords)), coords[:, 0], coords[:, 1]])
    beta, *_ = np.linalg.lstsq(A, values, rcond=None)
    return values - A @ beta


def fit_one(values, coords, maxlag, seed, detrend):
    rng = np.random.default_rng(seed)
    if len(values) > SAMPLE_SIZE:
        idx = rng.choice(len(values), size=SAMPLE_SIZE, replace=False)
        values, coords = values[idx], coords[idx]
    if detrend:
        values = remove_linear_trend(values, coords)

    V = skg.Variogram(
        coordinates=coords, values=values,
        model=MODEL, n_lags=N_LAGS, maxlag=maxlag, use_nugget=True,
    )
    rng_, sill, nugget = V.parameters[:3]
    return float(rng_), float(sill), float(nugget), float(V.rmse)


def analyse_band(values, coords, is_target):
    out = {}
    detrend = DETREND and not is_target
    for maxlag in MAXLAGS:
        fits = []
        for r in range(N_REPEATS):
            try:
                fits.append(fit_one(values, coords, maxlag, RANDOM_STATE + r, detrend))
            except Exception as e:
                print(f"    fit failed (maxlag={maxlag}, repeat={r}): {e}")
        if not fits:
            continue
        arr = np.array(fits)
        rng_med = float(np.median(arr[:, 0]))
        out[maxlag] = {
            "range_m": rng_med,
            "sill": float(np.median(arr[:, 1])),
            "nugget": float(np.median(arr[:, 2])),
            "rmse": float(np.median(arr[:, 3])),
            "hit_bound": bool(rng_med >= HIT_BOUND_FRACTION * maxlag),
        }
    return out


def main():
    df, coords = load_bands(GIS_RASTER, BAND_NAMES)
    extent_x = coords[:, 0].max() - coords[:, 0].min()
    extent_y = coords[:, 1].max() - coords[:, 1].min()
    print(f"{len(df)} valid pixels | extent {extent_x/1000:.0f} x {extent_y/1000:.0f} km\n")

    results = {}
    for band in PREDICTOR_BANDS + [TARGET_BAND]:
        values = df[band].values.astype(float)
        if np.std(values) == 0:
            print(f"{band}: constant, skipped")
            continue
        is_target = band == TARGET_BAND
        print(f"{band}{'  (TARGET)' if is_target else ''}")
        res = analyse_band(values, coords, is_target)
        results[band] = res
        for maxlag, r in res.items():
            flag = "  <- hit upper bound (no sill, unreliable)" if r["hit_bound"] else ""
            print(f"  maxlag {maxlag:>6} m: range {r['range_m']:8.0f} m | "
                  f"sill {r['sill']:.4g} | nugget {r['nugget']:.4g}{flag}")

    def reliable_ranges(band):
        return [r["range_m"] for r in results.get(band, {}).values() if not r["hit_bound"]]

    target_ranges = reliable_ranges(TARGET_BAND)
    pred_ranges = [x for b in PREDICTOR_BANDS for x in reliable_ranges(b)]

    summary = {"per_band": results, "extent_m": [float(extent_x), float(extent_y)]}
    if target_ranges:
        buffer_m = float(np.median(target_ranges))
        summary["suggested_buffer_radius_m"] = buffer_m
        summary["suggested_block_spacing_m"] = buffer_m * 2
        print(f"Target autocorrelation range (reliable fits): ~{buffer_m:.0f} m")
        print(f"  BUFFER_RADIUS   ~ {buffer_m:.0f} m")
        print(f"  BLOCK_SPACING   >= {buffer_m * 2:.0f} m (blocks should be larger than the buffer)")
        if buffer_m > 0.25 * min(extent_x, extent_y):
            print("  WARNING: buffer is large relative to the study area; check how much "
                  "training data the buffer removes.")
    else:
        print("No reliable range for the target: every fit hit its upper bound.")
        print("The data has a trend or very long-range structure at these lags; "
              "choose the buffer from a small lag window (<= 3 km) or by testing "
              "how much training data each buffer size removes.")
    if pred_ranges:
        print(f"(For reference, median reliable predictor range: {np.median(pred_ranges):.0f} m)")

    out_path = pathlib.Path("variogram_ranges.json")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()