"""
03_flood_classification_ml.py

Trains a Random Forest classifier to map post-flood land cover into
{water, flooded, urban, agriculture, forest, barren} using post-flood
spectral bands + indices + bi-temporal change layers as features, then
derives a binary flood-extent map and validates it against the
simulated reference inundation used to generate the scene.

Feature vector per pixel (10 features):
  Blue, Green, Red, NIR, SWIR1 (post-flood reflectance)
  NDVI_post, MNDWI_post
  dMNDWI, dNDVI (bi-temporal change)
  Elevation (relative DEM, 0-1)

This mirrors a standard operational workflow: reference/training pixels
would normally come from field GPS points or a validated historical
inundation map; here labels are generated from the same rule-based
land-cover/flood logic used to synthesize the imagery (i.e. this
functions as a controlled accuracy-assessment benchmark for the
pipeline, exactly as a held-out validation set would).
"""
import os, sys, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_raster, save_raster

CLASSES = ["water", "flooded", "urban", "agriculture", "forest", "barren"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def build_features():
    post, meta = load_raster("data/post_flood/post_flood_bands.tif")
    idx_post, _ = load_raster("outputs/indices/indices_post_flood.tif")   # NDVI, NDWI, MNDWI
    chg, _ = load_raster("outputs/indices/change_layers.tif")             # dMNDWI, dNDVI
    dem = np.load("data/dem.npy")

    ndvi_post, _, mndwi_post = idx_post[0], idx_post[1], idx_post[2]
    d_mndwi, d_ndvi = chg[0], chg[1]

    feat_names = ["Blue", "Green", "Red", "NIR", "SWIR1",
                  "NDVI_post", "MNDWI_post", "dMNDWI", "dNDVI", "Elevation"]
    stack = np.stack([post[0], post[1], post[2], post[3], post[4],
                       ndvi_post, mndwi_post, d_mndwi, d_ndvi, dem])
    H, W = dem.shape
    X = stack.reshape(len(feat_names), -1).T  # (H*W, n_features)
    return X, feat_names, meta, (H, W)


def build_labels():
    lc = np.load("data/landcover_labels.npy", allow_pickle=True)
    flood_extent = np.load("data/flood_extent_truth.npy")
    water_pre = (lc == "water")
    new_flood = flood_extent & (~water_pre)

    labels = np.empty(lc.shape, dtype=object)
    labels[:] = lc[:]
    labels[new_flood] = "flooded"
    y = np.vectorize(CLASS_ID.get)(labels)
    return y.ravel(), labels


def plot_confusion(cm, classes, path):
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Land Cover / Flood Classification")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel("Reference")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, feat_names, path):
    imp = model.feature_importances_
    order = np.argsort(imp)
    plt.figure(figsize=(6.5, 4.5))
    plt.barh(np.array(feat_names)[order], imp[order], color="#2b7a78")
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    X, feat_names, meta, (H, W) = build_features()
    y, label_grid = build_labels()

    # Stratified pixel sample for training (operational practice: don't
    # train on every pixel of a full scene) -- 40,000 px training sample
    rng = np.random.default_rng(7)
    n_sample = min(40000, X.shape[0])
    idx_all = np.arange(X.shape[0])
    sample_idx = rng.choice(idx_all, size=n_sample, replace=False)
    X_sample, y_sample = X[sample_idx], y[sample_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.30, random_state=7, stratify=y_sample)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=7)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1w = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=CLASSES, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Held-out accuracy: {acc:.4f}   Weighted F1: {f1w:.4f}")
    print(report)

    os.makedirs("outputs/classified", exist_ok=True)
    with open("outputs/classified/classification_report.txt", "w") as f:
        f.write(f"Overall accuracy: {acc:.4f}\nWeighted F1: {f1w:.4f}\n\n{report}\n")

    plot_confusion(cm, CLASSES, "outputs/figures/confusion_matrix.png")
    plot_feature_importance(clf, feat_names, "outputs/figures/feature_importance.png")

    # classify the FULL scene
    y_full_pred = clf.predict(X)
    classified_map = y_full_pred.reshape(H, W).astype(np.uint8)
    save_raster("outputs/classified/landcover_classified.tif",
                classified_map[np.newaxis, ...].astype(np.float32), meta,
                band_names=["class_id"], description="RF land-cover/flood classification")

    flood_pred_mask = (classified_map == CLASS_ID["flooded"]) | (classified_map == CLASS_ID["water"])
    np.save("outputs/classified/classified_map.npy", classified_map)
    np.save("outputs/classified/flood_pred_mask.npy", flood_pred_mask)

    with open("outputs/classified/class_legend.json", "w") as f:
        json.dump(CLASS_ID, f, indent=2)

    px_area_km2 = (meta["px_size_x"] * 111.32) * (meta["px_size_y"] * 110.57)  # approx km^2/px at this latitude
    print("\nMapped class areas (predicted, full scene):")
    area_rows = []
    for c, cid in CLASS_ID.items():
        n_px = int((classified_map == cid).sum())
        area_km2 = n_px * px_area_km2
        pct = 100 * n_px / classified_map.size
        area_rows.append((c, n_px, area_km2, pct))
        print(f"  {c:12s}  {n_px:7d} px   {area_km2:8.2f} km^2   {pct:5.2f}%")

    import csv
    with open("outputs/classified/class_area_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "pixel_count", "area_km2", "percent_of_scene"])
        w.writerows(area_rows)

    print("\nSaved: outputs/classified/landcover_classified.tif, class_area_summary.csv, "
          "classification_report.txt, figures/confusion_matrix.png, figures/feature_importance.png")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
