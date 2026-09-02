# Geospatial Assessment of Flood-Affected Areas in Nepal
### Multi-Temporal Satellite Imagery & Machine Learning · Case Study: Kathmandu Valley / Koshi-Bagmati Basin

An end-to-end flood-mapping pipeline: multi-temporal imagery → spectral indices →
bi-temporal change detection → Random Forest classification → GIS mapping &
vector export → auto-generated report.

**⚠️ Data note:** This build environment has no internet access, so real
Sentinel-2/Landsat scenes could not be downloaded. The pipeline runs on a
physically-informed **synthetic** 5-band dataset (DEM + rule-based land cover +
a stochastic monsoon flood simulator, calibrated to realistic Sentinel-2 L2A
reflectance ranges) so the full workflow can be demonstrated and validated.
Every stage after `01_generate_synthetic_imagery.py` consumes plain 5-band
arrays and is agnostic to where they came from — see **"Using real satellite
imagery"** below to point it at an actual scene.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `src/01_generate_synthetic_imagery.py` | Pre/post-flood 5-band rasters (`data/`) |
| 2 | `src/02_spectral_indices.py` | NDVI, NDWI, MNDWI, change layers (`outputs/indices/`) |
| 3 | `src/03_flood_classification_ml.py` | Random Forest land-cover/flood classification (`outputs/classified/`) |
| 4 | `src/04_visualization_gis.py` | Georeferenced maps, zonal stats, GeoJSON (`outputs/maps/`, `outputs/vectors/`) |
| 5 | `src/05_generate_report.js` | `outputs/Flood_Assessment_Report_Nepal.docx` |

Run in order:

```bash
python3 src/01_generate_synthetic_imagery.py
python3 src/02_spectral_indices.py
python3 src/03_flood_classification_ml.py
python3 src/04_visualization_gis.py
node   src/05_generate_report.js
```

## Method summary

- **Imagery**: 5 bands (Blue, Green, Red, NIR, SWIR1), pre-flood (pre-monsoon)
  vs. post-flood (post-monsoon peak) composites.
- **Indices**: NDVI (vegetation), NDWI (open water), MNDWI (water incl.
  turbid/urban water — the primary flood index).
- **Change detection**: ΔMNDWI (water gain) and ΔNDVI (vegetation loss)
  between dates.
- **ML classification**: Random Forest (300 trees) on a 10-feature pixel
  vector (5 bands + NDVI + MNDWI + ΔMNDWI + ΔNDVI + elevation) into six
  classes: water, flooded, urban, agriculture, forest, barren.
- **GIS outputs**: georeferenced classification map (WGS84 lon/lat, scale
  bar, north arrow), zonal flood-impact statistics by basin sub-area, and a
  GeoJSON polygon export of the flood extent for QGIS/ArcGIS.

## Results at a glance (synthetic demo run)

- Held-out classification accuracy: **99.7%** (weighted F1 0.997)
- Estimated flooded + open-water extent: **~113 km²** (~7.7% of the scene)
- Highest zonal impact: Kathmandu Metro core (20.0%) and Lalitpur (18.3%)

Full methodology, figures, and tables: `outputs/Flood_Assessment_Report_Nepal.docx`.

## Project structure

```
flood-assessment-nepal/
├── src/                        pipeline scripts (run in numeric order)
├── data/                       generated pre/post-flood rasters + DEM/labels
├── outputs/
│   ├── indices/                NDVI/NDWI/MNDWI + change-layer GeoTIFFs
│   ├── classified/              classification raster, accuracy report, area/zonal stats
│   ├── maps/                    georeferenced flood map, comparison panel, previews
│   ├── figures/                 index maps, confusion matrix, feature importance, charts
│   ├── vectors/                 flood_extent.geojson
│   └── Flood_Assessment_Report_Nepal.docx
└── README.md
```

## Using real satellite imagery

To run this on an actual flood event:

1. **Get imagery**: Sentinel-2 L2A (10 m, optical) via [Copernicus Data Space
   Ecosystem](https://dataspace.copernicus.eu/) or Google Earth Engine;
   Sentinel-1 SAR is strongly recommended alongside it since monsoon flood
   scenes are frequently cloud-covered and SAR sees through cloud.
2. **Replace Step 1**: swap `01_generate_synthetic_imagery.py` for a loader
   that reads your two GeoTIFFs (`rasterio.open(path).read()`) into the same
   `(5, H, W)` band order (Blue, Green, Red, NIR, SWIR1) and calls
   `utils.save_raster()` with the real affine transform/CRS from the source
   file's metadata instead of the synthetic `affine_transform()`.
3. **Everything downstream is unchanged** — indices, ML classification, GIS
   mapping, and the report generator all operate on the saved band arrays,
   not on how they were produced.
4. **Swap reference labels**: replace the rule-based `build_labels()` in
   Step 3 with real training points (field GPS, Copernicus Emergency
   Management Service flood maps, or hand-digitized reference polygons).
5. **Real zone boundaries**: replace the illustrative `ZONES` grid in Step 4
   with actual ward/municipality GeoJSON boundaries for Kathmandu
   Metropolitan City, Lalitpur, Bhaktapur, and the relevant Koshi basin
   districts.

## Requirements

See `requirements.txt`. Core stack: numpy, scipy, scikit-learn, scikit-image,
matplotlib, pandas, Pillow, tifffile (Python); `docx` (Node, for the report).
For production use with real satellite data, add `rasterio` and `geopandas`
for proper CRS/reprojection handling (this environment could not install
them without internet access, so lightweight numpy-based equivalents are
used here instead — see `src/utils.py`).

## Multitemporal_comparison_panel

<img width="2100" height="750" alt="multitemporal_comparison_panel" src="https://github.com/user-attachments/assets/e198925f-b5e9-42ea-80aa-35d344e19b6f" />

## Flood Extent & Land-Cover Classification

<img width="1350" height="1260" alt="flood_extent_classification_map" src="https://github.com/user-attachments/assets/89eabc57-9b58-4ecf-9d16-f233a37fa98c" />

## Nepal flood 2025-2026

"C:\Users\Abhishek\Downloads\Flood_Assessment_Report_Nepal.docx"
