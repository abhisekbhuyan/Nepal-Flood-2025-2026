const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, AlignmentType, PageOrientation,
  Header, Footer, PageNumber, NumberFormat, LevelFormat, convertInchesToTwip
} = require("docx");

const ROOT = __dirname + "/..";
const areaRows = fs.readFileSync(`${ROOT}/outputs/classified/class_area_summary.csv`, "utf8")
  .trim().split("\n").slice(1).map(l => {
    const [cls, px, km2, pct] = l.split(",");
    return { cls, px: Number(px), km2: Number(km2), pct: Number(pct) };
  });
const zonal = JSON.parse(fs.readFileSync(`${ROOT}/outputs/classified/zonal_statistics.json`, "utf8"));
const reportTxt = fs.readFileSync(`${ROOT}/outputs/classified/classification_report.txt`, "utf8");
const accLine = reportTxt.split("\n")[0];
const f1Line = reportTxt.split("\n")[1];

function img(path, width, height) {
  return new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } });
}

function h(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 280, after: 140 } });
}

function p(runsOrText, opts = {}) {
  const children = typeof runsOrText === "string" ? [new TextRun(runsOrText)] : runsOrText;
  return new Paragraph({ children, spacing: { after: 160 }, ...opts });
}

function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}

function figure(path, caption, widthPx = 560) {
  // read PNG dims via a quick header parse to keep aspect ratio
  const buf = fs.readFileSync(path);
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  const w = widthPx;
  const hgt = Math.round(widthPx * (height / width));
  return [
    new Paragraph({ children: [img(path, w, hgt)], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 } }),
    new Paragraph({
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "555555" })],
      alignment: AlignmentType.CENTER, spacing: { after: 240 },
    }),
  ];
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1f3864" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
    verticalAlign: "center",
  });
}

function areaTable() {
  const widths = [2600, 1800, 2200, 2200];
  const header = new TableRow({ children: [
    cell("Land-cover / flood class", { header: true, width: widths[0] }),
    cell("Pixels", { header: true, width: widths[1] }),
    cell("Area (km²)", { header: true, width: widths[2] }),
    cell("% of scene", { header: true, width: widths[3] }),
  ]});
  const rows = areaRows.map(r => new TableRow({ children: [
    cell(r.cls, { width: widths[0] }),
    cell(r.px.toLocaleString(), { width: widths[1] }),
    cell(r.km2.toFixed(2), { width: widths[2] }),
    cell(r.pct.toFixed(2) + "%", { width: widths[3] }),
  ]}));
  return new Table({ width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA }, columnWidths: widths, rows: [header, ...rows] });
}

function zonalTable() {
  const widths = [3200, 2000, 2000, 1600];
  const header = new TableRow({ children: [
    cell("Zone", { header: true, width: widths[0] }),
    cell("Flooded area (km²)", { header: true, width: widths[1] }),
    cell("Existing water (km²)", { header: true, width: widths[2] }),
    cell("% of zone flooded", { header: true, width: widths[3] }),
  ]});
  const rows = zonal.map(z => new TableRow({ children: [
    cell(z.zone, { width: widths[0] }),
    cell(z.flooded_km2.toFixed(2), { width: widths[1] }),
    cell(z.existing_water_km2.toFixed(2), { width: widths[2] }),
    cell(z.flooded_pct_of_zone.toFixed(1) + "%", { width: widths[3] }),
  ]}));
  return new Table({ width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA }, columnWidths: widths, rows: [header, ...rows] });
}

const totalFlooded = areaRows.find(r => r.cls === "flooded").km2 + areaRows.find(r => r.cls === "water").km2;
const totalFloodedPct = (areaRows.find(r => r.cls === "flooded").pct + areaRows.find(r => r.cls === "water").pct).toFixed(2);

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } },
    },
    headers: {
      default: new Header({ children: [ new Paragraph({
        children: [new TextRun({ text: "Geospatial Flood Assessment — Kathmandu Valley / Koshi-Bagmati Basin", size: 16, color: "777777" })],
        alignment: AlignmentType.CENTER,
      })]}),
    },
    footers: {
      default: new Footer({ children: [ new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun({ text: "Page ", size: 16, color: "777777" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "777777" }),
        ],
      })]}),
    },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Geospatial Assessment of Flood-Affected Areas in Nepal", bold: true, size: 40, color: "1f3864" })],
        alignment: AlignmentType.CENTER, spacing: { after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Using Multi-Temporal Satellite Imagery & Machine Learning (2024–2026)", size: 26, color: "444444" })],
        alignment: AlignmentType.CENTER, spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Case Study: Kathmandu Valley / Koshi-Bagmati Basin", italics: true, size: 24, color: "444444" })],
        alignment: AlignmentType.CENTER, spacing: { after: 320 },
      }),

      h("1. Executive Summary", HeadingLevel.HEADING_1),
      p(`This study assesses flood extent and land-cover change across a Kathmandu Valley / Koshi-Bagmati Basin case-study area using a bi-temporal (pre-flood vs. post-flood) multispectral remote-sensing workflow combined with a supervised machine-learning classifier. Spectral water indices (NDWI, MNDWI) and vegetation indices (NDVI) were computed for both dates, differenced to isolate the flood signal, and fed into a Random Forest classifier that mapped the scene into six land-cover/flood classes.`),
      p(`The classifier reached ${accLine.replace("Overall accuracy: ", "an overall held-out accuracy of ")} (${f1Line.replace("Weighted F1: ", "weighted F1-score ")}) against reference labels. The model estimates a combined flooded + open-water extent of approximately ${totalFlooded.toFixed(1)} km² (${totalFloodedPct}% of the study scene), concentrated along the Bagmati river corridor through the Kathmandu Metro core and Lalitpur, which show the highest proportional inundation of the five zones assessed.`),

      h("2. Study Area & Data", HeadingLevel.HEADING_1),
      p("Study area: Kathmandu Valley and an adjoining reach of the Koshi-Bagmati basin, approximately bounded by 85.15°–85.55°E and 27.55°–27.85°N (WGS84). The area spans dense urban core (Kathmandu, Lalitpur, Bhaktapur), peri-urban agriculture, forested hillslopes, and the Bagmati river network with a tributary confluence."),
      p([
        new TextRun({ text: "Data note: ", bold: true }),
        new TextRun("This working environment had no internet access, so real Sentinel-2/Landsat scenes could not be downloaded from Copernicus Data Space Ecosystem, USGS EarthExplorer, or Google Earth Engine. The pipeline below was therefore demonstrated on a physically-informed synthetic 5-band (Blue, Green, Red, NIR, SWIR1) dataset generated from a fractal-noise DEM, a rule-based land-cover template, and a stochastic monsoon flood-inundation simulator calibrated to Sentinel-2 L2A surface-reflectance ranges. Every downstream step — indices, change detection, ML classification, GIS mapping, vectorization — operates on the 5-band array unmodified, so swapping in a real Sentinel-2/Landsat GeoTIFF pair reproduces the same outputs on real imagery. See README.md, section 'Using real satellite imagery.'"),
      ]),
      p("Imagery bands and acquisition timing simulated:"),
      bullet("Pre-flood: pre-monsoon baseline composite"),
      bullet("Post-flood: post-monsoon peak-inundation composite"),
      bullet("Bands: Blue, Green, Red, NIR, SWIR1 (Sentinel-2-like)"),
      bullet("Grid: 512×512 px over the study bounding box (≈95 m/px effective)"),

      h("3. Methodology", HeadingLevel.HEADING_1),
      h("3.1 Pre-processing", HeadingLevel.HEADING_2),
      p("Bands were assembled per acquisition date and clipped to the study extent. In an operational deployment this stage would additionally include atmospheric correction (Sen2Cor/L2A), cloud/shadow masking, and co-registration of the two dates."),
      h("3.2 Spectral index computation", HeadingLevel.HEADING_2),
      bullet("NDVI = (NIR − Red) / (NIR + Red) — vegetation vigor"),
      bullet("NDWI = (Green − NIR) / (Green + NIR) — open water (McFeeters, 1996)"),
      bullet("MNDWI = (Green − SWIR1) / (Green + SWIR1) — water incl. turbid/urban water (Xu, 2006); primary flood index used here"),
      h("3.3 Change detection", HeadingLevel.HEADING_2),
      p("Bi-temporal differencing produced ΔMNDWI (water gain) and ΔNDVI (vegetation loss) layers, which sharply isolate the flood-affected corridor from background land-cover noise (Figure 2)."),
      h("3.4 Machine-learning classification", HeadingLevel.HEADING_2),
      p("A Random Forest classifier (300 trees, max depth 18, class-balanced) was trained on a stratified pixel sample using a 10-feature vector per pixel: 5 post-flood reflectance bands, post-flood NDVI and MNDWI, the two change layers (ΔMNDWI, ΔNDVI), and relative elevation. Training used a 70/30 train/test split; classes were {water, flooded, urban, agriculture, forest, barren}."),
      h("3.5 GIS mapping & vectorization", HeadingLevel.HEADING_2),
      p("The full-scene classification was rendered as a georeferenced map (WGS84 lon/lat), zonal flood statistics were tabulated across five basin sub-zones, and the binary flood-extent mask was vectorized to polygons (marching-squares contouring) and exported as GeoJSON for use in downstream GIS software (QGIS/ArcGIS)."),

      h("4. Results", HeadingLevel.HEADING_1),
      h("4.1 Multi-temporal change detection", HeadingLevel.HEADING_2),
      ...figure(`${ROOT}/outputs/maps/multitemporal_comparison_panel.png`, "Figure 1. Pre-flood, post-flood, and MNDWI change (flood signal) composites.", 580),
      h("4.2 Flood extent & land-cover classification map", HeadingLevel.HEADING_2),
      ...figure(`${ROOT}/outputs/maps/flood_extent_classification_map.png`, "Figure 2. Georeferenced land-cover/flood classification map with scale bar and north arrow.", 430),
      h("4.3 Classification accuracy", HeadingLevel.HEADING_2),
      p(`Held-out test accuracy: ${accLine.split(": ")[1]}  |  Weighted F1-score: ${f1Line.split(": ")[1]}`),
      ...figure(`${ROOT}/outputs/figures/confusion_matrix.png`, "Figure 3. Confusion matrix, held-out test pixels.", 380),
      ...figure(`${ROOT}/outputs/figures/feature_importance.png`, "Figure 4. Random Forest feature importance — MNDWI and its bi-temporal change dominate flood discrimination, as expected.", 430),
      h("4.4 Mapped area by class", HeadingLevel.HEADING_2),
      areaTable(),
      new Paragraph({ text: "", spacing: { after: 200 } }),
      h("4.5 Zonal flood impact", HeadingLevel.HEADING_2),
      ...figure(`${ROOT}/outputs/figures/zonal_flood_impact.png`, "Figure 5. Flooded area as a percentage of each zone.", 460),
      zonalTable(),

      h("5. Limitations", HeadingLevel.HEADING_1),
      bullet("Imagery is synthetic (see Data note, Section 2) — absolute figures are illustrative of the workflow, not a real-world flood assessment."),
      bullet("Zone boundaries are an illustrative grid proxy, not surveyed administrative/ward boundaries."),
      bullet("No atmospheric correction, cloud masking, or multi-date co-registration step was needed here since the synthetic dates are already aligned and cloud-free; real imagery requires these steps."),
      bullet("Training/reference labels came from the same rule-based logic used to generate the scene; on real imagery, labels should come from field GPS points, high-resolution reference imagery, or validated historical flood maps."),

      h("6. Recommended Next Steps (for deployment on real imagery)", HeadingLevel.HEADING_1),
      bullet("Pull Sentinel-2 L2A (10 m) or Sentinel-1 SAR (cloud-penetrating, critical for monsoon flood mapping) via Copernicus Data Space Ecosystem or Google Earth Engine."),
      bullet("Add Sentinel-1 VV/VH backscatter change detection to complement optical MNDWI, since monsoon flood scenes are frequently cloud-obscured."),
      bullet("Replace the rule-based reference labels with field-validated or crowd-sourced (e.g. Copernicus EMS) flood-extent ground truth."),
      bullet("Extend zonal statistics to real ward/municipality boundaries (Kathmandu Metropolitan City, Lalitpur, Bhaktapur, and Koshi basin districts) via official administrative GeoJSON/shapefiles."),
      bullet("Wire the GeoJSON output into a QGIS/ArcGIS project or a web map (Leaflet/Mapbox) for stakeholder-facing disaster-response dashboards."),

      h("7. Project Deliverables", HeadingLevel.HEADING_1),
      bullet("src/ — full Python pipeline (imagery → indices → ML classification → GIS mapping/vectorization)"),
      bullet("outputs/maps/ — georeferenced flood-extent map, comparison panel"),
      bullet("outputs/figures/ — index maps, confusion matrix, feature importance, zonal chart"),
      bullet("outputs/classified/ — classified raster, area/zonal statistics (CSV/JSON), accuracy report"),
      bullet("outputs/vectors/flood_extent.geojson — flood-extent polygons for GIS software"),
      bullet("README.md — how to run the pipeline and how to swap in real satellite imagery"),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(`${ROOT}/outputs/Flood_Assessment_Report_Nepal.docx`, buf);
  console.log("Wrote outputs/Flood_Assessment_Report_Nepal.docx");
});
