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

## Geospatial Assessment of Flood-Affected Areas in Nepal
Using Multi-Temporal Satellite Imagery & Machine Learning (2024–2026)
Case Study: Kathmandu Valley / Koshi-Bagmati Basin
1. Executive Summary
This study assesses flood extent and land-cover change across a Kathmandu Valley / Koshi-Bagmati Basin case-study area using a bi-temporal (pre-flood vs. post-flood) multispectral remote-sensing workflow combined with a supervised machine-learning classifier. Spectral water indices (NDWI, MNDWI) and vegetation indices (NDVI) were computed for both dates, differenced to isolate the flood signal, and fed into a Random Forest classifier that mapped the scene into six land-cover/flood classes.
The classifier reached an overall held-out accuracy of 0.9969 (weighted F1-score 0.9969) against reference labels. The model estimates a combined flooded + open-water extent of approximately 113.1 km² (7.66% of the study scene), concentrated along the Bagmati river corridor through the Kathmandu Metro core and Lalitpur, which show the highest proportional inundation of the five zones assessed.
2. Study Area & Data
Study area: Kathmandu Valley and an adjoining reach of the Koshi-Bagmati basin, approximately bounded by 85.15°–85.55°E and 27.55°–27.85°N (WGS84). The area spans dense urban core (Kathmandu, Lalitpur, Bhaktapur), peri-urban agriculture, forested hillslopes, and the Bagmati river network with a tributary confluence.
Data note: This working environment had no internet access, so real Sentinel-2/Landsat scenes could not be downloaded from Copernicus Data Space Ecosystem, USGS EarthExplorer, or Google Earth Engine. The pipeline below was therefore demonstrated on a physically-informed synthetic 5-band (Blue, Green, Red, NIR, SWIR1) dataset generated from a fractal-noise DEM, a rule-based land-cover template, and a stochastic monsoon flood-inundation simulator calibrated to Sentinel-2 L2A surface-reflectance ranges. Every downstream step — indices, change detection, ML classification, GIS mapping, vectorization — operates on the 5-band array unmodified, so swapping in a real Sentinel-2/Landsat GeoTIFF pair reproduces the same outputs on real imagery. See README.md, section 'Using real satellite imagery.'
Imagery bands and acquisition timing simulated:
●	Pre-flood: pre-monsoon baseline composite
●	Post-flood: post-monsoon peak-inundation composite
●	Bands: Blue, Green, Red, NIR, SWIR1 (Sentinel-2-like)
●	Grid: 512×512 px over the study bounding box (≈95 m/px effective)
3. Methodology
3.1 Pre-processing
Bands were assembled per acquisition date and clipped to the study extent. In an operational deployment this stage would additionally include atmospheric correction (Sen2Cor/L2A), cloud/shadow masking, and co-registration of the two dates.
3.2 Spectral index computation
●	NDVI = (NIR − Red) / (NIR + Red) — vegetation vigor
●	NDWI = (Green − NIR) / (Green + NIR) — open water (McFeeters, 1996)
●	MNDWI = (Green − SWIR1) / (Green + SWIR1) — water incl. turbid/urban water (Xu, 2006); primary flood index used here
3.3 Change detection
Bi-temporal differencing produced ΔMNDWI (water gain) and ΔNDVI (vegetation loss) layers, which sharply isolate the flood-affected corridor from background land-cover noise (Figure 2).
3.4 Machine-learning classification
A Random Forest classifier (300 trees, max depth 18, class-balanced) was trained on a stratified pixel sample using a 10-feature vector per pixel: 5 post-flood reflectance bands, post-flood NDVI and MNDWI, the two change layers (ΔMNDWI, ΔNDVI), and relative elevation. Training used a 70/30 train/test split; classes were {water, flooded, urban, agriculture, forest, barren}.
3.5 GIS mapping & vectorization
The full-scene classification was rendered as a georeferenced map (WGS84 lon/lat), zonal flood statistics were tabulated across five basin sub-zones, and the binary flood-extent mask was vectorized to polygons (marching-squares contouring) and exported as GeoJSON for use in downstream GIS software (QGIS/ArcGIS).
4. Results
4.1 Multi-temporal change detection
   <img width="906" height="323" alt="image" src="https://github.com/user-attachments/assets/d58f7655-a190-45fe-bf99-4975d7a4e688" />
   
Figure 1. Pre-flood, post-flood, and MNDWI change (flood signal) composites.
4.2 Flood extent & land-cover classification map
<img width="672" height="627" alt="image" src="https://github.com/user-attachments/assets/1ffe8f6c-5808-45a9-8b3d-bedb928580a5" />

Figure 2. Georeferenced land-cover/flood classification map with scale bar and north arrow.
4.3 Classification accuracy
Held-out test accuracy: 0.9969  |  Weighted F1-score: 0.9969
<img width="594" height="503" alt="image" src="https://github.com/user-attachments/assets/9bafe4f2-3834-4418-906b-8be9b9e2ac16" />

Figure 3. Confusion matrix, held-out test pixels.
<img width="672" height="466" alt="image" src="https://github.com/user-attachments/assets/a3624b9e-750f-4e89-ad57-8c97c541491e" />

Figure 4. Random Forest feature importance — MNDWI and its bi-temporal change dominate flood discrimination, as expected.
4.4 Mapped area by class
Land-cover / flood class	Pixels	Area (km²)	% of scene
water	5,405	30.45	2.06%
flooded	14,669	82.65	5.60%
urban	1,122	6.32	0.43%
agriculture	229	1.29	0.09%
forest	238,795	1345.48	91.09%
barren	1,924	10.84	0.73%

4.5 Zonal flood impact
<img width="719" height="405" alt="image" src="https://github.com/user-attachments/assets/e99defa9-33a7-4d0b-ad46-d63cd19c6e82" />

Figure 5. Flooded area as a percentage of each zone.
Zone	Flooded area (km²)	Existing water (km²)	% of zone flooded
Kathmandu Metro (core)	18.48	2.23	20.0%
Lalitpur	16.93	3.63	18.3%
Bhaktapur	2.69	0.37	4.5%
Bagmati corridor (upstream)	6.33	5.42	4.8%
Koshi confluence (downstream)	6.84	3.81	4.4%
5. Limitations
●	Imagery is synthetic (see Data note, Section 2) — absolute figures are illustrative of the workflow, not a real-world flood assessment.
●	Zone boundaries are an illustrative grid proxy, not surveyed administrative/ward boundaries.
●	No atmospheric correction, cloud masking, or multi-date co-registration step was needed here since the synthetic dates are already aligned and cloud-free; real imagery requires these steps.
●	Training/reference labels came from the same rule-based logic used to generate the scene; on real imagery, labels should come from field GPS points, high-resolution reference imagery, or validated historical flood maps.
6. Recommended Next Steps (for deployment on real imagery)
●	Pull Sentinel-2 L2A (10 m) or Sentinel-1 SAR (cloud-penetrating, critical for monsoon flood mapping) via Copernicus Data Space Ecosystem or Google Earth Engine.
●	Add Sentinel-1 VV/VH backscatter change detection to complement optical MNDWI, since monsoon flood scenes are frequently cloud-obscured.
●	Replace the rule-based reference labels with field-validated or crowd-sourced (e.g. Copernicus EMS) flood-extent ground truth.
●	Extend zonal statistics to real ward/municipality boundaries (Kathmandu Metropolitan City, Lalitpur, Bhaktapur, and Koshi basin districts) via official administrative GeoJSON/shapefiles.
●	Wire the GeoJSON output into a QGIS/ArcGIS project or a web map (Leaflet/Mapbox) for stakeholder-facing disaster-response dashboards.
7. Project Deliverables
●	src/ — full Python pipeline (imagery → indices → ML classification → GIS mapping/vectorization)
●	outputs/maps/ — georeferenced flood-extent map, comparison panel
●	outputs/figures/ — index maps, confusion matrix, feature importance, zonal chart
●	outputs/classified/ — classified raster, area/zonal statistics (CSV/JSON), accuracy report
●	outputs/vectors/flood_extent.geojson — flood-extent polygons for GIS software
●	README.md — how to run the pipeline and how to swap in real satellite imagery

