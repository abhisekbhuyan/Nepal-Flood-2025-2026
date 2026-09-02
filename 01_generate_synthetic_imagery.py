"""
01_generate_synthetic_imagery.py

Generates physically-plausible, terrain-driven multi-temporal 5-band
(Blue, Green, Red, NIR, SWIR1 - Sentinel-2-like) imagery for two dates:
  - PRE-FLOOD  (pre-monsoon baseline)
  - POST-FLOOD (post-monsoon-peak acquisition)

*** DATA NOTE ***
This environment has no internet access, so real Sentinel-2/Landsat scenes
(e.g. from Copernicus Data Space Ecosystem, USGS EarthExplorer, or Google
Earth Engine) could not be downloaded. Instead this script synthesizes
imagery from a generated DEM + land-cover template for the Kathmandu
Valley / Koshi-Bagmati basin footprint, with realistic reflectance ranges,
river-network hydrology, and a monsoon flood-extent simulation. The entire
downstream pipeline (indices, ML classification, GIS mapping, reporting)
is written to consume real satellite bands unmodified -- swap this script
for a real image loader (rasterio.open(...).read()) and everything else
runs unchanged. See README.md "Using real satellite imagery" section.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from utils import affine_transform, save_raster, save_rgb_png, STUDY_AREA

rng = np.random.default_rng(42)

W, H = 512, 512  # ~ 90m/px effective over the ~0.4deg study box


def make_terrain():
    """Fractal-noise DEM with a dominant river valley (Bagmati-like) running
    SW-NE plus a tributary, embedded in a bowl-shaped valley (Kathmandu)."""
    base = rng.normal(0, 1, (H, W))
    dem = np.zeros((H, W))
    amp = 1.0
    for octave in range(6):
        sigma = max(1, 64 // (2 ** octave))
        dem += amp * gaussian_filter(base, sigma=sigma)
        amp *= 0.55
    dem = (dem - dem.min()) / (dem.max() - dem.min())

    yy, xx = np.mgrid[0:H, 0:W]
    # valley bowl: lower elevation near center-south (Kathmandu Valley floor)
    cy, cx = H * 0.55, W * 0.45
    bowl = np.sqrt(((yy - cy) / (H * 0.55)) ** 2 + ((xx - cx) / (W * 0.55)) ** 2)
    dem = dem * 0.6 + bowl * 0.4

    # main river channel: a smooth curve from SW to NE, carved lower
    t = np.linspace(0, 1, 400)
    river_y = (H * 0.85) - t * (H * 0.55) + 25 * np.sin(t * 6)
    river_x = (W * 0.10) + t * (W * 0.80) + 20 * np.cos(t * 4)
    river_mask = np.zeros((H, W))
    for ry, rx in zip(river_y, river_x):
        rr, cc = int(ry), int(rx)
        if 0 <= rr < H and 0 <= cc < W:
            river_mask[rr, cc] = 1
    river_dist = gaussian_filter(river_mask, sigma=6)
    river_dist = river_dist / river_dist.max()
    dem -= river_dist * 0.35

    # tributary
    t2 = np.linspace(0, 1, 250)
    trib_y = (H * 0.15) + t2 * (H * 0.45)
    trib_x = (W * 0.75) - t2 * (W * 0.35) + 15 * np.sin(t2 * 5)
    trib_mask = np.zeros((H, W))
    for ry, rx in zip(trib_y, trib_x):
        rr, cc = int(ry), int(rx)
        if 0 <= rr < H and 0 <= cc < W:
            trib_mask[rr, cc] = 1
    trib_dist = gaussian_filter(trib_mask, sigma=4)
    trib_dist = trib_dist / trib_dist.max()
    dem -= trib_dist * 0.22

    dem = (dem - dem.min()) / (dem.max() - dem.min())
    river_proximity = np.maximum(river_dist, trib_dist * 0.8)
    return dem, river_proximity


def make_landcover(dem, river_proximity):
    """Rule-based land-cover template: water, urban, agriculture, forest,
    barren -- based on elevation + proximity to the river network, tuned to
    resemble the dense urban core of Kathmandu/Lalitpur/Bhaktapur ringed by
    peri-urban agriculture and forested hills."""
    lc = np.full((H, W), "forest", dtype=object)

    water = river_proximity > 0.55
    urban_core = (dem < 0.42) & (river_proximity < 0.5) & (river_proximity > 0.08)
    urban_core = gaussian_filter(urban_core.astype(float), 3) > 0.35
    agriculture = (dem < 0.55) & (~urban_core) & (~water) & (river_proximity > 0.12)
    barren = dem > 0.82

    lc[water] = "water"
    lc[agriculture] = "agriculture"
    lc[urban_core] = "urban"
    lc[barren] = "barren"
    return lc, water


CLASS_PRESETS = {
    # class: (means[B,G,R,NIR,SWIR1], intra-class texture std[B,G,R,NIR,SWIR1])
    "water":       ([0.03, 0.04, 0.03, 0.02, 0.01], [0.006, 0.006, 0.005, 0.004, 0.003]),
    "urban":       ([0.13, 0.14, 0.17, 0.22, 0.28], [0.018, 0.018, 0.02, 0.025, 0.03]),
    "agriculture": ([0.05, 0.09, 0.07, 0.38, 0.20], [0.012, 0.014, 0.012, 0.035, 0.02]),
    "forest":      ([0.03, 0.06, 0.04, 0.42, 0.16], [0.006, 0.008, 0.006, 0.025, 0.015]),
    "barren":      ([0.16, 0.18, 0.22, 0.26, 0.30], [0.015, 0.015, 0.016, 0.02, 0.02]),
    "flooded":     ([0.06, 0.07, 0.06, 0.09, 0.05], [0.012, 0.012, 0.01, 0.015, 0.01]),
}


def band_reflectance_for_class(cls, n, rng):
    """Mean reflectance only; spatial texture is added separately in
    build_scene as a spatially-correlated field (see note there)."""
    means, _ = CLASS_PRESETS[cls]
    return np.tile(np.array(means, dtype=np.float32), (n, 1))


def simulate_flood(dem, river_proximity, water_pre, intensity=0.62):
    """Monsoon flood: inundation extends outward from the existing channel
    into low-lying land, modulated by elevation and a stochastic 'rainfall
    intensity' field (captures the fact that real floods are not a clean
    elevation threshold)."""
    rain_field = gaussian_filter(rng.normal(0, 1, (H, W)), sigma=18)
    rain_field = (rain_field - rain_field.min()) / (rain_field.max() - rain_field.min())
    susceptibility = (1 - dem) * 0.55 + river_proximity * 0.55 + rain_field * intensity * 0.5
    threshold = np.percentile(susceptibility, 100 - 14)  # ~14% of scene newly flooded at peak
    new_flood = (susceptibility > threshold) & (~water_pre)
    # keep it spatially connected to existing water / low elevation (buffer)
    connectivity = gaussian_filter(water_pre.astype(float), sigma=10)
    new_flood = new_flood & (connectivity > 0.02)
    flood_extent = water_pre | new_flood
    return flood_extent, new_flood


def build_scene(dem, river_proximity, lc, flood_mask=None, sensor_noise=0.003):
    """Assign per-class mean reflectance, then add a spatially-correlated
    texture field per class+band (mimicking real within-class heterogeneity
    like crop rows / rooftop material mix / canopy gaps) plus a small
    amount of fine, spatially-independent sensor noise."""
    bands = np.zeros((H, W, 5), dtype=np.float32)
    classes = np.unique(lc)
    class_mask_map = {}
    for cls in classes:
        mask = (lc == cls)
        if flood_mask is not None:
            mask = mask & (~flood_mask)
        class_mask_map[cls] = mask
        n = mask.sum()
        if n == 0:
            continue
        bands[mask] = band_reflectance_for_class(cls, n, rng)
    if flood_mask is not None and flood_mask.sum() > 0:
        class_mask_map["flooded"] = flood_mask
        bands[flood_mask] = band_reflectance_for_class("flooded", flood_mask.sum(), rng)

    for cls, mask in class_mask_map.items():
        if mask.sum() == 0:
            continue
        _, tex_std = CLASS_PRESETS[cls]
        for b in range(5):
            field = gaussian_filter(rng.normal(0, 1, (H, W)), sigma=3.0)
            field = field / (field.std() + 1e-6)
            bands[..., b][mask] += (field[mask] * tex_std[b])

    # mild sensor PSF blur, then fine independent sensor noise
    for b in range(bands.shape[-1]):
        bands[..., b] = gaussian_filter(bands[..., b], sigma=0.6)
    bands += rng.normal(0, sensor_noise, bands.shape)
    bands = np.clip(bands, 0, 1)
    return np.transpose(bands, (2, 0, 1))  # -> (bands, H, W)


def main():
    print(f"Study area: {STUDY_AREA['name']}")
    print(f"Grid: {W}x{H} px  |  bbox: {STUDY_AREA}")

    dem, river_proximity = make_terrain()
    lc, water_pre = make_landcover(dem, river_proximity)
    flood_extent, new_flood = simulate_flood(dem, river_proximity, water_pre)

    transform = affine_transform(W, H)

    pre_bands = build_scene(dem, river_proximity, lc, flood_mask=None)
    post_bands = build_scene(dem, river_proximity, lc, flood_mask=flood_extent)

    save_raster("data/pre_flood/pre_flood_bands.tif", pre_bands, transform,
                description="Pre-flood (pre-monsoon baseline) 5-band synthetic composite")
    save_raster("data/post_flood/post_flood_bands.tif", post_bands, transform,
                description="Post-flood (post-monsoon-peak) 5-band synthetic composite")

    np.save("data/dem.npy", dem)
    np.save("data/river_proximity.npy", river_proximity)
    np.save("data/landcover_labels.npy", lc)
    np.save("data/flood_extent_truth.npy", flood_extent)
    np.save("data/new_flood_truth.npy", new_flood)
    import json
    with open("data/transform.json", "w") as f:
        json.dump(transform, f, indent=2)

    # quick-look true-color composites (R,G,B = bands 2,1,0)
    save_rgb_png("outputs/maps/preview_pre_flood_rgb.png",
                 pre_bands[2], pre_bands[1], pre_bands[0])
    save_rgb_png("outputs/maps/preview_post_flood_rgb.png",
                 post_bands[2], post_bands[1], post_bands[0])

    pct_flooded = 100 * flood_extent.sum() / flood_extent.size
    pct_new = 100 * new_flood.sum() / new_flood.size
    print(f"Reference (simulated ground-truth) total inundation: {pct_flooded:.2f}% of scene")
    print(f"Reference newly-flooded (vs pre-flood water): {pct_new:.2f}% of scene")
    print("Saved: data/pre_flood/, data/post_flood/, outputs/maps/preview_*.png")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
