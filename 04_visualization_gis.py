"""
04_visualization_gis.py

Produces the GIS deliverables:
  - Georeferenced flood-extent map (lat/lon axes) over the study area
  - Pre/Post/Change 3-panel comparison figure
  - Land-cover classification map with legend
  - Zonal flood-impact statistics by administrative sub-area (proxy grid
    standing in for Kathmandu/Lalitpur/Bhaktapur + basin districts)
  - Flood-extent polygon export to GeoJSON (vectorized from the raster
    mask via marching squares, coordinates in WGS84 lon/lat)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage import measure

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_raster, pixel_to_lonlat, STUDY_AREA

CLASSES = ["water", "flooded", "urban", "agriculture", "forest", "barren"]
CLASS_COLORS = {
    "water": "#1f78b4", "flooded": "#e31a1c", "urban": "#999999",
    "agriculture": "#ffcc33", "forest": "#33a02c", "barren": "#a67c52",
}

# Approximate named zones within the study grid for zonal statistics
# (illustrative administrative proxy, not surveyed boundaries)
ZONES = {
    "Kathmandu Metro (core)":     (0.30, 0.55, 0.30, 0.55),
    "Lalitpur":                   (0.45, 0.70, 0.20, 0.45),
    "Bhaktapur":                  (0.15, 0.35, 0.45, 0.65),
    "Bagmati corridor (upstream)": (0.55, 0.85, 0.00, 0.30),
    "Koshi confluence (downstream)": (0.00, 0.30, 0.55, 0.90),
}
# tuple = (row_frac_min, row_frac_max, col_frac_min, col_frac_max)


def extent_lonlat(transform):
    lon0 = transform["top_left_lon"]
    lat0 = transform["top_left_lat"]
    lon1 = lon0 + transform["width"] * transform["px_size_x"]
    lat1 = lat0 - transform["height"] * transform["px_size_y"]
    return [lon0, lon1, lat1, lat0]  # imshow extent: left,right,bottom,top


def plot_flood_extent_map(classified_map, transform, path):
    H, W = classified_map.shape
    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASSES])
    ext = extent_lonlat(transform)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.imshow(classified_map, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1,
              extent=ext, origin="upper")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Flood Extent & Land-Cover Classification\nKathmandu Valley / Koshi-Bagmati Basin (case-study subset)",
                 fontsize=11)
    handles = [Patch(facecolor=CLASS_COLORS[c], label=c.capitalize()) for c in CLASSES]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    # north arrow + scale bar (approx, 1 deg lon ~ 98.7 km at this latitude)
    ax.annotate("N", xy=(0.94, 0.92), xytext=(0.94, 0.86), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    bar_deg = 0.05  # ~5.5 km
    x0 = ext[0] + 0.03 * (ext[1] - ext[0])
    y0 = ext[2] + 0.05 * (ext[3] - ext[2])
    ax.plot([x0, x0 + bar_deg], [y0, y0], color="black", lw=2)
    ax.text(x0 + bar_deg / 2, y0 + 0.01 * (ext[3] - ext[2]), "~5.5 km",
            ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_comparison_panel(pre_png, post_png, change_png, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, img_path, title in zip(
        axes, [pre_png, post_png, change_png],
        ["Pre-Flood (baseline)", "Post-Flood (peak inundation)", "MNDWI Change (flood signal)"]
    ):
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle("Multi-Temporal Flood Change Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def zonal_statistics(classified_map, class_id, transform):
    H, W = classified_map.shape
    flooded_mask = (classified_map == class_id["flooded"])
    water_mask = (classified_map == class_id["water"])
    rows = []
    for name, (r0f, r1f, c0f, c1f) in ZONES.items():
        r0, r1 = int(r0f * H), int(r1f * H)
        c0, c1 = int(c0f * W), int(c1f * W)
        sub_total = (r1 - r0) * (c1 - c0)
        sub_flooded = flooded_mask[r0:r1, c0:c1].sum()
        sub_water = water_mask[r0:r1, c0:c1].sum()
        px_area_km2 = (transform["px_size_x"] * 111.32) * (transform["px_size_y"] * 110.57)
        rows.append({
            "zone": name,
            "flooded_px": int(sub_flooded),
            "flooded_km2": round(sub_flooded * px_area_km2, 2),
            "flooded_pct_of_zone": round(100 * sub_flooded / sub_total, 2),
            "existing_water_km2": round(sub_water * px_area_km2, 2),
        })
    return rows


def plot_zonal_chart(rows, path):
    zones = [r["zone"] for r in rows]
    pct = [r["flooded_pct_of_zone"] for r in rows]
    order = np.argsort(pct)[::-1]
    zones = [zones[i] for i in order]
    pct = [pct[i] for i in order]

    plt.figure(figsize=(8, 4.5))
    bars = plt.barh(zones, pct, color="#c0392b")
    plt.xlabel("Flooded area (% of zone)")
    plt.title("Flood Impact by Zone")
    for b, v in zip(bars, pct):
        plt.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def export_geojson(flood_mask, transform, path):
    contours = measure.find_contours(flood_mask.astype(float), level=0.5)
    features = []
    for c in contours:
        if len(c) < 4:
            continue
        coords = []
        for (row, col) in c[::2]:  # thin vertices a bit
            lon, lat = pixel_to_lonlat(row, col, transform)
            coords.append([round(lon, 6), round(lat, 6)])
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "properties": {"class": "flood_extent"},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    geojson = {
        "type": "FeatureCollection",
        "name": "flood_extent_kathmandu_koshi_bagmati",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(geojson, f)
    return len(features)


def main():
    classified_map = np.load("outputs/classified/classified_map.npy")
    with open("outputs/classified/class_legend.json") as f:
        class_id = json.load(f)
    with open("data/transform.json") as f:
        transform = json.load(f)
    flood_pred_mask = np.load("outputs/classified/flood_pred_mask.npy")

    plot_flood_extent_map(classified_map, transform, "outputs/maps/flood_extent_classification_map.png")
    plot_comparison_panel(
        "outputs/maps/preview_pre_flood_rgb.png",
        "outputs/maps/preview_post_flood_rgb.png",
        "outputs/figures/d_mndwi.png",
        "outputs/maps/multitemporal_comparison_panel.png",
    )

    rows = zonal_statistics(classified_map, class_id, transform)
    plot_zonal_chart(rows, "outputs/figures/zonal_flood_impact.png")
    with open("outputs/classified/zonal_statistics.json", "w") as f:
        json.dump(rows, f, indent=2)

    n_features = export_geojson(flood_pred_mask, transform, "outputs/vectors/flood_extent.geojson")

    print("Zonal flood statistics:")
    for r in rows:
        print(f"  {r['zone']:32s} {r['flooded_pct_of_zone']:5.1f}% flooded  "
              f"({r['flooded_km2']} km^2)")
    print(f"\nExported {n_features} flood-extent polygon(s) -> outputs/vectors/flood_extent.geojson")
    print("Saved maps -> outputs/maps/flood_extent_classification_map.png, "
          "multitemporal_comparison_panel.png")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
