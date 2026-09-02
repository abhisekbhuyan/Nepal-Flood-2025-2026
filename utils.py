"""
utils.py
Shared helpers: lightweight geo-referencing, raster I/O, and band math.

This project ships without GDAL/rasterio/geopandas (no internet access was
available to install them in this environment), so georeferencing is
implemented manually: every raster is a numpy array plus a small affine
transform (stored as a sidecar .json "world file"). This keeps every array
pixel mappable to real WGS84 lat/lon for the Kathmandu Valley / Koshi-Bagmati
study area, and the code is written so it drops in cleanly if you later
swap the synthetic arrays for real rasterio DatasetReader.read() arrays.
"""
import json
import os
import numpy as np
from PIL import Image
import tifffile

# ---------------------------------------------------------------------------
# Study area definition
# ---------------------------------------------------------------------------
# Approx bounding box covering Kathmandu Valley and the lower/mid
# Koshi-Bagmati basin reach used for this case study (WGS84, EPSG:4326)
STUDY_AREA = {
    "name": "Kathmandu Valley / Koshi-Bagmati Basin (subset)",
    "lon_min": 85.15,
    "lon_max": 85.55,
    "lat_min": 27.55,
    "lat_max": 27.85,
}

BAND_ORDER = ["Blue", "Green", "Red", "NIR", "SWIR1"]  # Sentinel-2-like


def affine_transform(width, height, area=STUDY_AREA):
    """Return (pixel_size_x, pixel_size_y, top_left_lon, top_left_lat)."""
    px_x = (area["lon_max"] - area["lon_min"]) / width
    px_y = (area["lat_max"] - area["lat_min"]) / height
    return {
        "width": width,
        "height": height,
        "px_size_x": px_x,
        "px_size_y": px_y,
        "top_left_lon": area["lon_min"],
        "top_left_lat": area["lat_max"],
        "crs": "EPSG:4326",
    }


def pixel_to_lonlat(row, col, transform):
    lon = transform["top_left_lon"] + (col + 0.5) * transform["px_size_x"]
    lat = transform["top_left_lat"] - (row + 0.5) * transform["px_size_y"]
    return lon, lat


def save_raster(path_tif, array, transform, band_names=None, description=""):
    """Save a (bands,H,W) or (H,W) float32 array as a GeoTIFF-like multi-page
    TIFF with an accompanying .json transform (acts as our 'world file' +
    CRS record)."""
    os.makedirs(os.path.dirname(path_tif), exist_ok=True)
    arr = array.astype(np.float32)
    tifffile.imwrite(path_tif, arr, description=description)
    meta = dict(transform)
    meta["band_names"] = band_names or BAND_ORDER
    meta["shape"] = list(arr.shape)
    with open(path_tif + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return path_tif


def load_raster(path_tif):
    arr = tifffile.imread(path_tif)
    with open(path_tif + ".json") as f:
        meta = json.load(f)
    return arr, meta


def stretch_to_uint8(band, lo_pct=2, hi_pct=98):
    lo, hi = np.percentile(band, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((band - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def save_rgb_png(path_png, r, g, b, title=None):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    rgb = np.dstack([stretch_to_uint8(r), stretch_to_uint8(g), stretch_to_uint8(b)])
    Image.fromarray(rgb).save(path_png)
    return path_png


def normalized_diff(a, b):
    denom = (a + b)
    denom[denom == 0] = 1e-6
    return (a - b) / denom
