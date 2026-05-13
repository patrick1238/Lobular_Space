from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from readlif.reader import LifFile
from cellpose import models
from skimage.color import label2rgb
from skimage.measure import regionprops_table
import cv2
from scipy.ndimage import gaussian_filter


# =========================================================
# INPUT VIA CONSOLE
# =========================================================
input_dir_str = input("Bitte Pfad zum LIF-Ordner eingeben: ").strip()
input_dir = Path(input_dir_str)

if not input_dir.exists():
    raise FileNotFoundError(f"Ordner existiert nicht: {input_dir}")
if not input_dir.is_dir():
    raise NotADirectoryError(f"Pfad ist kein Ordner: {input_dir}")

stain = input("Bitte Stain eingeben, z.B. CD8: ").strip()
diameter_um = float(input("Bitte Cellpose-Diameter in µm eingeben: ").strip())

output_dir = Path("./Celldetection")
output_dir.mkdir(exist_ok=True, parents=True)


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
min_size = max(5, diameter_px / 2)
max_files = None


print(f"\nNeue Auflösung nach Resize: {px:.4f} µm/Pixel")
print(f"Cellpose diameter: {diameter_um:.2f} µm = {diameter_px:.2f} Pixel")


# =========================================================
# HELPERS
# =========================================================
def is_lif_file(path: Path) -> bool:
    return path.name.lower().endswith(".lif")


def parse_metadata(input_file: Path):
    filename = input_file.name
    base0 = filename.split("_")[0]
    parts = base0.split("-")

    caseid = parts[0] if len(parts) > 0 else ""
    imageid = parts[1] if len(parts) > 1 else ""

    return caseid, imageid


def get_channel_from_filename(input_file: Path, stain: str) -> int:
    """
    Sucht den Stain im Dateinamen und nimmt seine Position als Channel-Index.

    Beispiel:
    Datei: 123-456_DAPI-CD8-CD3.lif
    stain = CD8
    Channel = 1

    Datei: 123-456_CD3_CD8_DAPI.lif
    stain = CD8
    Channel = 1
    """
    stem = input_file.stem
    stain_norm = stain.lower()

    parts = re.split(r"[_\-\s]+", stem)
    parts_norm = [p.lower() for p in parts if p]

    matches = [i for i, p in enumerate(parts_norm) if p == stain_norm]

    if len(matches) == 0:
        raise ValueError(
            f"Stain '{stain}' wurde nicht im Dateinamen gefunden: {input_file.name}"
        )

    # Falls caseid-imageid vorne stehen, diese entfernen wir heuristisch:
    # wir nehmen die Position innerhalb der Stain-Liste nach dem ersten nicht-numerischen Block.
    stain_like_parts = [
        p for p in parts_norm
        if not p.isdigit()
    ]

    if stain_norm not in stain_like_parts:
        raise ValueError(
            f"Stain '{stain}' konnte nicht eindeutig im Dateinamen interpretiert werden."
        )

    return stain_like_parts.index(stain_norm)


def load_lif_channel_maxproj_resized(input_file: Path, stain: str):
    channel = get_channel_from_filename(input_file, stain) - 1

    lif = LifFile(str(input_file))
    images = list(lif.get_iter_image())
    img = images[-1]

    print(f"Gewählte Serie: {img.name}")
    print(f"Berechneter Channel für {stain}: {channel}")

    planes = img.get_iter_z(c=channel)

    max_proj_image = np.array(next(planes))

    for plane in planes:
        np.maximum(max_proj_image, plane, out=max_proj_image)

    if max_proj_image.dtype != np.uint8:
        max_val = max_proj_image.max()
        if max_val > 0:
            max_proj_image = (
                255 * (max_proj_image / max_val)
            ).astype(np.uint8)
        else:
            max_proj_image = max_proj_image.astype(np.uint8)

    pil_img = Image.fromarray(max_proj_image, mode="L")

    new_size_1 = (
        int(pil_img.width * scale_factor),
        int(pil_img.height * scale_factor),
    )
    new_size_2 = (
        int(new_size_1[0] * scale_factor),
        int(new_size_1[1] * scale_factor),
    )

    resized_img = pil_img.resize(
        new_size_1,
        resample=Image.BILINEAR
    ).resize(
        new_size_2,
        resample=Image.BILINEAR
    )

    return np.array(resized_img), channel


def make_overlay(img_2d: np.ndarray, masks_2d: np.ndarray, outpath: Path):
    overlay = label2rgb(masks_2d, image=img_2d, bg_label=0)

    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def run_cellpose(model, img_2d: np.ndarray) -> np.ndarray:
    # Sharpening
    blur = gaussian_filter(img_2d, sigma=1)
    img_2d = img_2d + (img_2d - blur)
    img_2d = np.clip(img_2d, 0, 255).astype(np.uint8)
    img_2d = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(img_2d)
    masks, _, _ = model.eval(
        img_2d,
        diameter=diameter_px,
        flow_threshold=flow_threshold,
        niter=niter,
        min_size=min_size,
    )
    return masks


def measure_masks_2d(
    masks: np.ndarray,
    intensity_image: np.ndarray,
    input_file: Path,
    caseid: str,
    imageid: str,
    stain: str,
    px: float,
) -> pd.DataFrame:

    props = regionprops_table(
        masks,
        intensity_image=intensity_image,
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
            "x_µm",
            "y_µm",
            "area_µm2",
            "perimeter_µm",
            "axis_major_length_µm",
            "axis_minor_length_µm",
            "eccentricity",
            "circularity",
            "solidity",
            "orientation",
            "mean_intensity",
            "integrated_intensity_per_µm2",
        ])

    df = df.rename(columns={
        "label": "cellid",
        "centroid-0": "y_px",
        "centroid-1": "x_px",
        "area": "area_px2",
        "perimeter": "perimeter_px",
        "axis_major_length": "axis_major_length_px",
        "axis_minor_length": "axis_minor_length_px",
    })

    df["path"] = str(input_file)
    df["caseid"] = caseid
    df["imageid"] = imageid
    df["stain"] = stain

    df["x_µm"] = df["x_px"] * px
    df["y_µm"] = df["y_px"] * px

    df["area_µm2"] = df["area_px2"] * px ** 2
    df["perimeter_µm"] = df["perimeter_px"] * px
    df["axis_major_length_µm"] = df["axis_major_length_px"] * px
    df["axis_minor_length_µm"] = df["axis_minor_length_px"] * px

    df["circularity"] = np.where(
        df["perimeter_px"] > 0,
        4 * np.pi * df["area_px2"] / (df["perimeter_px"] ** 2),
        np.nan
    )

    df["integrated_intensity"] = df["mean_intensity"] * df["area_px2"]

    df["integrated_intensity_per_µm2"] = np.where(
        df["area_µm2"] > 0,
        df["integrated_intensity"] / df["area_µm2"],
        np.nan
    )

    preferred_cols = [
        "path",
        "caseid",
        "imageid",
        "stain",
        "cellid",
        "x_px",
        "y_px",
        "x_µm",
        "y_µm",
        "area_µm2",
        "perimeter_µm",
        "axis_major_length_µm",
        "axis_minor_length_µm",
        "eccentricity",
        "circularity",
        "solidity",
        "orientation",
        "mean_intensity",
        "integrated_intensity_per_µm2",
    ]

    return df[preferred_cols]


def process_one_lif(model, input_file: Path, output_dir: Path) -> pd.DataFrame:
    caseid, imageid = parse_metadata(input_file)

    print(f"\nVerarbeite: {input_file.name}")

    img, channel = load_lif_channel_maxproj_resized(input_file, stain)

    print(f"Finale Bildgröße nach Resize: {img.shape}")

    masks = run_cellpose(model, img)

    out_overlay = output_dir / f"{input_file.stem}_{stain}_ch{channel}_cellpose_overlay.png"
    make_overlay(img, masks, out_overlay)

    df = measure_masks_2d(
        masks=masks,
        intensity_image=img,
        input_file=input_file,
        caseid=caseid,
        imageid=imageid,
        stain=stain,
        px=px,
    )

    if len(df) > 0:
        df["cell_uid"] = [
            f"{caseid}_{imageid}_{stain}_{cid}" for cid in df["cellid"]
        ]

    print(f"Detected cells: {len(df)}")
    return df


# =========================================================
# MAIN
# =========================================================
all_files = sorted([
    p for p in input_dir.iterdir()
    if p.is_file() and is_lif_file(p)
])

if max_files is not None:
    all_files = all_files[:max_files]

if len(all_files) == 0:
    raise FileNotFoundError(f"Keine LIF-Dateien gefunden in {input_dir}")

print(f"{len(all_files)} LIF-Datei(en) gefunden.")

model = models.CellposeModel(
    gpu=use_gpu,
    pretrained_model=pretrained_model,
)

print(f"Cellpose device: {model.device}")

all_tables = []

for input_file in all_files:
    try:
        df_one = process_one_lif(model, input_file, output_dir)
        all_tables.append(df_one)
    except Exception as e:
        print(f"ERROR bei {input_file.name}: {e}")

if len(all_tables) == 0:
    raise RuntimeError("Keine Datei konnte erfolgreich verarbeitet werden.")

df_all = pd.concat(all_tables, ignore_index=True)

front_cols = ["path", "caseid", "imageid", "stain", "cellid", "cell_uid"]
other_cols = [c for c in df_all.columns if c not in front_cols]
df_all = df_all[[c for c in front_cols if c in df_all.columns] + other_cols]

out_csv = output_dir / f"cellpose_lif_measurements_{stain}.csv"
df_all.to_csv(out_csv, index=False)

print("\nFertig.")
print(f"CSV gespeichert: {out_csv}")
print(f"Output-Ordner: {output_dir}")
print(df_all.head())