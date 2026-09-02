"""
02_spectral_indices.py

Computes standard remote-sensing spectral indices for both dates and
derives change-detection layers used for flood delineation:
  NDVI  = (NIR - Red)  / (NIR + Red)        vegetation vigor
  NDWI  = (Green - NIR)/ (Green + NIR)      open water (McFeeters 1996)
  MNDWI = (Green - SWIR1)/(Green + SWIR1)   water incl. turbid/urban water
                                             (Xu 2006) - primary flood index
  dMNDWI = MNDWI_post - MNDWI_pre           water-gain change layer
  dNDVI  = NDVI_post - NDVI_pre             vegetation-loss change layer
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_raster, save_raster, normalized_diff, stretch_to_uint8, BAND_ORDER
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, G, R, NIR, SWIR1 = range(5)


def compute_indices(bands):
    ndvi = normalized_diff(bands[NIR], bands[R])
    ndwi = normalized_diff(bands[G], bands[NIR])
    mndwi = normalized_diff(bands[G], bands[SWIR1])
    return ndvi, ndwi, mndwi


def save_index_map(path, index_arr, cmap, title, vmin=-1, vmax=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(index_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=11)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    pre, meta = load_raster("data/pre_flood/pre_flood_bands.tif")
    post, _ = load_raster("data/post_flood/post_flood_bands.tif")

    ndvi_pre, ndwi_pre, mndwi_pre = compute_indices(pre)
    ndvi_post, ndwi_post, mndwi_post = compute_indices(post)

    d_mndwi = mndwi_post - mndwi_pre
    d_ndvi = ndvi_post - ndvi_pre

    stack_pre = np.stack([ndvi_pre, ndwi_pre, mndwi_pre]).astype(np.float32)
    stack_post = np.stack([ndvi_post, ndwi_post, mndwi_post]).astype(np.float32)
    save_raster("outputs/indices/indices_pre_flood.tif", stack_pre, meta,
                band_names=["NDVI", "NDWI", "MNDWI"], description="Pre-flood indices")
    save_raster("outputs/indices/indices_post_flood.tif", stack_post, meta,
                band_names=["NDVI", "NDWI", "MNDWI"], description="Post-flood indices")
    save_raster("outputs/indices/change_layers.tif",
                np.stack([d_mndwi, d_ndvi]).astype(np.float32), meta,
                band_names=["dMNDWI", "dNDVI"], description="Change-detection layers")

    save_index_map("outputs/figures/ndvi_pre.png", ndvi_pre, "RdYlGn", "NDVI - Pre-Flood")
    save_index_map("outputs/figures/ndvi_post.png", ndvi_post, "RdYlGn", "NDVI - Post-Flood")
    save_index_map("outputs/figures/mndwi_pre.png", mndwi_pre, "Blues", "MNDWI - Pre-Flood", vmin=-0.6, vmax=0.6)
    save_index_map("outputs/figures/mndwi_post.png", mndwi_post, "Blues", "MNDWI - Post-Flood", vmin=-0.6, vmax=0.6)
    save_index_map("outputs/figures/d_mndwi.png", d_mndwi, "seismic", "MNDWI Change (Post - Pre)\nPositive = new/increased water", vmin=-0.5, vmax=0.5)
    save_index_map("outputs/figures/d_ndvi.png", d_ndvi, "PiYG", "NDVI Change (Post - Pre)\nNegative = vegetation loss / inundation", vmin=-0.5, vmax=0.5)

    print("NDVI pre/post mean: %.3f / %.3f" % (ndvi_pre.mean(), ndvi_post.mean()))
    print("MNDWI pre/post mean: %.3f / %.3f" % (mndwi_pre.mean(), mndwi_post.mean()))
    print("Saved index rasters -> outputs/indices/, figures -> outputs/figures/")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
