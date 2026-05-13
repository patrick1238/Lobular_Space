from pathlib import Path
import re
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

from readlif.reader import LifFile
from cellpose import models
from skimage.color import label2rgb
from skimage.measure import regionprops_table, regionprops
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter

from tqdm import tqdm


# =========================================================
# SILENCE WARNINGS
# =========================================================
logging.getLogger("cellpose").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# =========================================================
# INPUT
# =========================================================
input_dir = Path(input("Enter path to LIF folder: ").strip())
if not input_dir.exists():
    raise FileNotFoundError(input_dir)
if not input_dir.is_dir():
    raise NotADirectoryError(input_dir)

stain = input("Enter stain (e.g. CD20): ").strip()
diameter_um = float(input("Enter Cellpose diameter in µm: ").strip())

output_dir = Path("./Celldetection")
output_dir.mkdir(exist_ok=True, parents=True)

overlay_dir = output_dir / "overlays"
csv_dir = output_dir / "per_image_csv"

overlay_dir.mkdir(exist_ok=True, parents=True)
csv_dir.mkdir(exist_ok=True, parents=True)


# =========================================================
# SETTINGS
# =========================================================
px_original = 0.164
scale_factor = 0.8
total_scale = scale_factor ** 2

px = px_original / total_scale
px2 = px ** 2
diameter_px = diameter_um / px

use_gpu = True
pretrained_model = "cpsam"
flow_threshold = 0.6
niter = None
min_size = max(5, diameter_px * 0.5)

print("\nSettings")
print(f"Input folder: {input_dir}")
print(f"Output folder: {output_dir}")
print(f"Stain: {stain}")
print(f"Resolution: {px:.4f} µm/px")
print(f"Cellpose diameter: {diameter_um:.2f} µm ({diameter_px:.1f} px)")
print(f"Model: {pretrained_model}")
print(f"GPU: {use_gpu}")
print(f"Flow threshold: {flow_threshold}")
print(f"niter: {niter}")
print(f"min_size: {min_size:.1f} px")


# =========================================================
# HELPERS
# =========================================================
def is_lif_file(path: Path) -> bool:
    return path.suffix.lower() == ".lif"


def parse_metadata(path: Path):
    parts = path.name.split("_")[0].split("-")
    parts = parts + ["", ""]
    caseid = parts[0]
    imageid = parts[1]
    return caseid, imageid


def get_channel_from_filename(path: Path, stain_name: str) -> int:
    stem = path.stem

    if "_" in stem:
        channel_part = stem.split("_", 1)[1]
    else:
        channel_part = stem

    channel_part = channel_part.split("_")[0]

    channels = [
        p.lower()
        for p in re.split(r"[-\s]+", channel_part)
        if p
    ]

    stain_norm = stain_name.lower()

    if stain_norm not in channels:
        raise ValueError(
            f"Stain '{stain_name}' not found in channel list {channels}"
        )

    return channels.index(stain_norm)


def load_lif_channel_maxproj_resized(path: Path):
    channel = get_channel_from_filename(path, stain)

    lif = LifFile(str(path))
    lif_images = list(lif.get_iter_image())
    lif_image = lif_images[-1]

    planes = lif_image.get_iter_z(c=channel)

    max_proj = np.array(next(planes))
    for plane in planes:
        np.maximum(max_proj, plane, out=max_proj)

    if max_proj.dtype != np.uint8:
        max_val = max_proj.max()
        if max_val > 0:
            max_proj = (255 * max_proj / max_val).astype(np.uint8)
        else:
            max_proj = max_proj.astype(np.uint8)

    pil_img = Image.fromarray(max_proj)

    size_1 = (
        int(pil_img.width * scale_factor),
        int(pil_img.height * scale_factor),
    )
    size_2 = (
        int(size_1[0] * scale_factor),
        int(size_1[1] * scale_factor),
    )

    resized = pil_img.resize(size_1).resize(size_2)

    return np.array(resized), channel


# =========================================================
# PREPROCESSING
# =========================================================
def remove_background_multiotsu(img: np.ndarray) -> np.ndarray:
    thresholds = threshold_multiotsu(img, classes=4)
    mask = img > thresholds[0]
    return (img * mask).astype(np.uint8)


# =========================================================
# CELLPOSE
# =========================================================
def run_cellpose(model, img: np.ndarray) -> np.ndarray:
    masks, _, _ = model.eval(
        img,
        diameter=diameter_px,
        flow_threshold=flow_threshold,
        niter=niter,
        min_size=min_size,
    )
    return masks


# =========================================================
# FILTER MASKS
# =========================================================
def filter_masks(masks: np.ndarray) -> np.ndarray:
    radius_um = diameter_um / 2
    single_cell_area_um2 = np.pi * radius_um**2

    max_allowed_area_um2 = 4 * single_cell_area_um2
    max_allowed_area_px2 = max_allowed_area_um2 / px2

    max_aspect_ratio = 10.0
    min_solidity = 0.1

    filtered = np.zeros_like(masks, dtype=np.uint32)
    new_label = 1

    for region in regionprops(masks):
        area_ok = region.area <= max_allowed_area_px2

        if region.minor_axis_length > 0:
            aspect_ratio = region.major_axis_length / region.minor_axis_length
        else:
            aspect_ratio = np.inf

        band_like = (
            aspect_ratio > max_aspect_ratio
            and region.solidity < min_solidity
        )

        if area_ok and not band_like:
            filtered[masks == region.label] = new_label
            new_label += 1

    return filtered


# =========================================================
# OUTPUT
# =========================================================
def make_overlay(img: np.ndarray, masks: np.ndarray, outpath: Path):
    overlay = label2rgb(masks, image=img, bg_label=0)

    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_mask(masks: np.ndarray, outpath: Path):
    tifffile.imwrite(outpath, masks.astype(np.uint32))


# =========================================================
# MEASUREMENTS
# =========================================================
def measure(
    masks: np.ndarray,
    img: np.ndarray,
    path: Path,
    caseid: str,
    imageid: str,
) -> pd.DataFrame:

    props = regionprops_table(
        masks,
        intensity_image=img,
        properties=[
            "label",
            "area",
            "centroid",
            "eccentricity",
            "axis_major_length",
            "axis_minor_length",
            "orientation",
            "perimeter",
            "solidity",
            "mean_intensity",
        ],
    )

    df = pd.DataFrame(props)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "path",
            "caseid",
            "imageid",
            "stain",
            "cellid",
            "x_px",
            "y_px",
            "x_um",
            "y_um",
            "area_um2",
            "perimeter_um",
            "axis_major_length_um",
            "axis_minor_length_um",
            "eccentricity",
            "circularity",
            "solidity",
            "orientation",
            "mean_intensity",
            "intensity_norm",
        ])

    df.rename(columns={
        "label": "cellid",
        "centroid-0": "y_px",
        "centroid-1": "x_px",
        "area": "area_px2",
        "perimeter": "perimeter_px",
        "axis_major_length": "axis_major_length_px",
        "axis_minor_length": "axis_minor_length_px",
    }, inplace=True)

    df["path"] = str(path)
    df["caseid"] = caseid
    df["imageid"] = imageid
    df["stain"] = stain

    df["x_um"] = df["x_px"] * px
    df["y_um"] = df["y_px"] * px
    df["area_um2"] = df["area_px2"] * px2
    df["perimeter_um"] = df["perimeter_px"] * px
    df["axis_major_length_um"] = df["axis_major_length_px"] * px
    df["axis_minor_length_um"] = df["axis_minor_length_px"] * px

    df["circularity"] = np.where(
        df["perimeter_px"] > 0,
        4 * np.pi * df["area_px2"] / (df["perimeter_px"] ** 2),
        np.nan,
    )

    df["integrated_intensity"] = df["mean_intensity"] * df["area_px2"]

    df["intensity_norm"] = np.where(
        df["area_um2"] > 0,
        df["integrated_intensity"] / df["area_um2"],
        np.nan,
    )

    preferred_cols = [
        "path",
        "caseid",
        "imageid",
        "stain",
        "cellid",
        "x_px",
        "y_px",
        "x_um",
        "y_um",
        "area_um2",
        "perimeter_um",
        "axis_major_length_um",
        "axis_minor_length_um",
        "eccentricity",
        "circularity",
        "solidity",
        "orientation",
        "mean_intensity",
        "intensity_norm",
    ]

    return df[preferred_cols]


# =========================================================
# MAIN
# =========================================================
files = sorted([path for path in input_dir.iterdir() if is_lif_file(path)])

if len(files) == 0:
    raise FileNotFoundError(f"No LIF files found in {input_dir}")

model = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=pretrained_model,
)

print(f"Cellpose device: {model.device}")
print(f"Found {len(files)} LIF file(s)")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

start_time = time.time()
all_tables = []

pbar = tqdm(files, desc="Processing", unit="image")

for path in pbar:
    image_start = time.time()

    try:
        caseid, imageid = parse_metadata(path)

        img, channel = load_lif_channel_maxproj_resized(path)
        img = remove_background_multiotsu(img)

        masks_raw = run_cellpose(model, img)
        masks_filtered = filter_masks(masks_raw)

        overlay_path = overlay_dir / f"{path.stem}_{stain}_ch{channel}_overlay.png"
        csv_path = csv_dir / f"{path.stem}_{stain}_measurements.csv"

        make_overlay(img, masks_filtered, overlay_path)

        df = measure(
            masks=masks_filtered,
            img=img,
            path=path,
            caseid=caseid,
            imageid=imageid,
        )

        df.to_csv(csv_path, index=False)

        if len(df) > 0:
            all_tables.append(df)

        image_time_min = (time.time() - image_start) / 60
        pbar.set_postfix({
            "last_min": f"{image_time_min:.1f}",
            "cells": len(df),
        })

    except Exception as error:
        error_log = output_dir / "errors.log"
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(f"{path.name}\t{repr(error)}\n")
        continue

if len(all_tables) > 0:
    df_all = pd.concat(all_tables, ignore_index=True)
    combined_csv = output_dir / f"measurements_{stain}_combined.csv"
    df_all.to_csv(combined_csv, index=False)

end_time = time.time()

print(f"\nDone in {(end_time - start_time) / 60:.2f} min")
print(f"Output folder: {output_dir}")