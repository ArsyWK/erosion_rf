import ee

# region GEE init
PROJECT_ID = 'YOUR_PROJECT_ID'

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

gaul = ee.FeatureCollection('FAO/GAUL/2025/level2')
roi = gaul.filter(ee.Filter.And(
    ee.Filter.eq('GAUL1_NAME', 'Sumatera Utara'),
    ee.Filter.eq('GAUL2_NAME', 'Tapanuli Tengah')
)).geometry()

roi_buf = roi.buffer(3000)

chirps_daily = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation')
# endregion

# region Config
SCALE = 30
HIGH_EROSION_T = 180
TARGET_CRS = 'EPSG:32647'
EPS = 1e-6
RAINFALL_YEARS = list(range(2019, 2024))     # 5-yr mean is a minimum
ESRI_YEAR = 2023
OC_SCALE = 5
LS_MAX = 50
EROSION_PERCENTILE = 67
MERIT_CELL_M = 90
ESRI_REMAPPED = True
FLAT_SLOPE_DEG = 1.0
NODATA = -9999
RUN_EXPORT = True

target_proj = ee.Projection(TARGET_CRS).atScale(SCALE)
# endregion


# region general func
def to_grid(image, method='bilinear'):
    """Resample onto the common 30 m UTM grid (do NOT clip before this)."""
    return image.resample(method).reproject(target_proj)
# endregion


# region ESRI class codes
if ESRI_REMAPPED:
    ESRI_CODES = {'water': 1, 'trees': 2, 'flooded_veg': 3, 'crops': 4, 'built': 5,
                  'bare': 6, 'snow': 7, 'clouds': 8, 'rangeland': 9}
else:
    ESRI_CODES = {'water': 1, 'trees': 2, 'flooded_veg': 4, 'crops': 5, 'built': 7,
                  'bare': 8, 'snow': 9, 'clouds': 10, 'rangeland': 11}
# endregion


# region DEM & terrain computed on buffered
dem_raw = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(roi_buf)
dem = to_grid(dem_raw, 'bilinear').rename('elevation').toFloat().unmask(0)

slope_deg = ee.Terrain.slope(dem).rename('slope').toFloat()
aspect_deg = ee.Terrain.aspect(dem).toFloat()
slope_rad = slope_deg.multiply(3.14159265359 / 180.0)
tan_slope = slope_rad.tan().max(EPS)
sin_slope = slope_rad.sin().max(EPS)

slope_pct = slope_rad.tan().multiply(100.0).rename('slope_pct')

flat = slope_deg.lt(FLAT_SLOPE_DEG)

aspect_rad = aspect_deg.multiply(3.14159265359 / 180.0)
aspect_sin = aspect_rad.sin().where(flat, 0).rename('aspect_sin').toFloat()
aspect_cos = aspect_rad.cos().where(flat, 0).rename('aspect_cos').toFloat()
# endregion

# region Hydrology
flow_acc_raw = ee.Image('MERIT/Hydro/v1_0_1').select('upa').clip(roi_buf)   # km^2
flow_acc = to_grid(flow_acc_raw, 'bilinear')
specific_area = flow_acc.multiply(1e6).divide(MERIT_CELL_M)                  # m^2 per m width

twi = specific_area.divide(tan_slope).add(EPS).log().rename('twi').toFloat()
spi = specific_area.multiply(tan_slope).add(EPS).log().rename('spi').toFloat()
# endregion
#  region rainfall annual total
rain_start = f'{RAINFALL_YEARS[0]}-01-01'
rain_end = f'{RAINFALL_YEARS[-1] + 1}-01-01'          # end date is exclusive
rainfall_raw = chirps_daily.filterDate(rain_start, rain_end).sum() \
    .divide(len(RAINFALL_YEARS)).clip(roi_buf)
rainfall = to_grid(rainfall_raw, 'bilinear').rename('rainfall').toFloat()
# endregion


# region Second-order finite difference on DEM
L = float(SCALE)

zx_kernel = ee.Kernel.fixed(3, 3, [[0, 0, 0], [-1 / (2 * L), 0, 1 / (2 * L)], [0, 0, 0]])
zy_kernel = ee.Kernel.fixed(3, 3, [[0, 1 / (2 * L), 0], [0, 0, 0], [0, -1 / (2 * L), 0]])
zxx_kernel = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1 / L**2, -2 / L**2, 1 / L**2], [0, 0, 0]])
zyy_kernel = ee.Kernel.fixed(3, 3, [[0, 1 / L**2, 0], [0, -2 / L**2, 0], [0, 1 / L**2, 0]])
zxy_kernel = ee.Kernel.fixed(3, 3, [[-1 / (4 * L**2), 0, 1 / (4 * L**2)],
                                     [0, 0, 0],
                                     [1 / (4 * L**2), 0, -1 / (4 * L**2)]])

zx = dem.convolve(zx_kernel).rename('zx')
zy = dem.convolve(zy_kernel).rename('zy')
zxx = dem.convolve(zxx_kernel).rename('zxx')
zyy = dem.convolve(zyy_kernel).rename('zyy')
zxy = dem.convolve(zxy_kernel).rename('zxy')
deriv_stack = ee.Image.cat([zx, zy, zxx, zyy, zxy])
# endregion


# region profile & plan curvature
profile_curv = deriv_stack.expression(
    '-(zxx*zx*zx + 2*zxy*zx*zy + zyy*zy*zy) / ((zx*zx + zy*zy + eps) * pow(1 + zx*zx + zy*zy, 1.5))',
    {'zx': zx, 'zy': zy, 'zxx': zxx, 'zyy': zyy, 'zxy': zxy, 'eps': EPS}
).rename('profile_curvature').toFloat()

plan_curv = deriv_stack.expression(
    '-(zxx*zy*zy - 2*zxy*zx*zy + zyy*zx*zx) / pow(zx*zx + zy*zy + eps, 1.5)',
    {'zx': zx, 'zy': zy, 'zxx': zxx, 'zyy': zyy, 'zxy': zxy, 'eps': EPS}
).rename('plan_curvature').toFloat()

profile_curv = profile_curv.where(flat, 0).clamp(-1, 1).rename('profile_curvature')
plan_curv = plan_curv.where(flat, 0).clamp(-1, 1).rename('plan_curvature')
# endregion


# region ESRI land cover
ESRI_ASSET = 'projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS'
esri_all = ee.ImageCollection(ESRI_ASSET)

esri_col = esri_all.filterBounds(roi_buf) \
    .filterDate(f'{ESRI_YEAR}-01-01', f'{ESRI_YEAR + 1}-01-01')

lulc10 = esri_col.mosaic().select(0).clip(roi_buf).reproject(crs=TARGET_CRS, scale=10)
lulc_class = lulc10.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=64) \
    .reproject(target_proj).rename('lulc')

land_mask = lulc_class.neq(ESRI_CODES['water'])
# endregion

# region erosion

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(roi)
      .filterDate('2023-01-01', '2024-12-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

def add_indices(img):
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    bsi = img.expression(
        '((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))',
        {
            'B11': img.select('B11'), # SWIR 1
            'B4': img.select('B4'),   # Red
            'B8': img.select('B8'),   # NIR
            'B2': img.select('B2')    # Blue
        }
    ).rename('BSI')
    return img.addBands([ndvi, bsi])

s2_indices = s2.map(add_indices)
bsi_max = s2_indices.select('BSI').reduce(ee.Reducer.percentile([90])).rename('bsi_p90')
ndvi_min = s2_indices.select('NDVI').reduce(ee.Reducer.percentile([10])).rename('ndvi_p10')

erosion_empirical_target = (
    bsi_max.gt(0.10)
    .And(ndvi_min.lt(0.35))
    .And(slope_deg.gt(8.5))
    .rename('erosion_class')
    .toFloat()
)
# endregion


# region extract
PREDICTOR_BANDS = ['elevation', 'slope', 'aspect_sin', 'aspect_cos', 'twi', 'spi',
                   'rainfall', 'plan_curvature', 'profile_curvature']

raster_stack = ee.Image.cat([
    dem, slope_deg, aspect_sin, aspect_cos, twi, spi, rainfall,
    plan_curv, profile_curv,
    erosion_empirical_target,
]).toFloat().updateMask(land_mask).clip(roi).unmask(NODATA)

task = ee.batch.Export.image.toDrive(
    image=raster_stack,
    description='Tapanuli_Tengah_Erosion_noRUSLE_30m',
    folder='GEE_Exports',
    fileNamePrefix='tapanuli_tengah_noRUSLE_erosion_30m',
    scale=SCALE,
    crs=TARGET_CRS,
    region=roi,
    maxPixels=int(1e9),
    fileFormat='GeoTIFF',
    formatOptions={'noData': NODATA},
)
# endregion




if RUN_EXPORT:
    task.start()
    print("Export started:", task.id)
