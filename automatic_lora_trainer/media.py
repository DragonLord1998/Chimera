import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .paths import dirs, ensure_case, image_paths


def placeholder_sheet(text):
    img = Image.new("RGB", (900, 620), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((24, 24, 876, 596), radius=18, outline=(40, 40, 40), width=3)
    draw.text((330, 285), text, fill=(80, 80, 80))
    return img


def contact_sheet(files, selected=None, max_images=80, cols=8, thumb=96, gap=8):
    files = [str(Path(p)) for p in files[:max_images]]
    selected = {str(Path(p)) for p in (selected or [])}
    if not files:
        return placeholder_sheet("No images yet")

    rows = int(math.ceil(len(files) / cols))
    width = cols * thumb + (cols + 1) * gap
    height = rows * thumb + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)

    for idx, path in enumerate(files):
        row = idx // cols
        col = idx % cols
        x = gap + col * (thumb + gap)
        y = gap + row * (thumb + gap)
        try:
            with Image.open(path) as img:
                tile = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (thumb, thumb), Image.Resampling.LANCZOS)
        except Exception:
            tile = Image.new("RGB", (thumb, thumb), (235, 235, 235))
        sheet.paste(tile, (x, y))
        is_selected = str(Path(path)) in selected
        outline = (255, 55, 55) if is_selected else (30, 30, 30)
        draw.rounded_rectangle((x, y, x + thumb, y + thumb), radius=8, outline=outline, width=5 if is_selected else 2)
        if is_selected:
            draw.rectangle((x, y, x + thumb, y + 12), fill=(255, 80, 80))
    return sheet


def selected_from_qc(case_name, top_n=None):
    paths = dirs(case_name)
    csv_path = paths["case"] / "qc_scores.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    df = df[df["passed"] == True].sort_values(["identity_score", "sharpness"], ascending=[False, False])
    if top_n:
        df = df.head(int(top_n))
    return [str(Path(p)) for p in df["file"].tolist()]


def refs_sheet(case_name):
    case_name = ensure_case(case_name)
    return contact_sheet(image_paths(dirs(case_name)["refs"]), max_images=8, cols=4, thumb=130)


def candidates_sheet(case_name):
    case_name = ensure_case(case_name)
    return contact_sheet(image_paths(dirs(case_name)["candidates"]))


def qc_sheet(case_name, top_n=100):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    return contact_sheet(image_paths(paths["candidates"]), selected=selected_from_qc(case_name, top_n=top_n))


def train_sheet(case_name):
    case_name = ensure_case(case_name)
    return contact_sheet(image_paths(dirs(case_name)["train"]))


def latest_sample_image(case_name):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    generated = image_paths(paths["output"])
    if generated:
        return generated[-1]
    curated = image_paths(paths["train"])
    if curated:
        return curated[0]
    refs = image_paths(paths["refs"])
    if refs:
        return refs[0]
    return None


def dashboard_sample(case_name):
    sample = latest_sample_image(case_name)
    if sample:
        with Image.open(sample) as img:
            return ImageOps.exif_transpose(img).convert("RGB").copy()
    return placeholder_sheet("No sample yet")


def strict_step_from_path(path):
    match = re.search(r"(?:step|steps|global[_-]?step|checkpoint[_-]?step)[^\d]{0,12}(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def training_artifact_steps(case_name):
    paths = dirs(case_name)
    sample_steps = set()
    checkpoint_steps = set()
    sample_files = []
    checkpoint_files = []
    for path in paths["output"].rglob("*"):
        if not path.is_file():
            continue
        step = strict_step_from_path(path)
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            sample_files.append(str(path))
            if step is not None:
                sample_steps.add(step)
        elif suffix in {".safetensors", ".pt", ".ckpt"}:
            checkpoint_files.append(str(path))
            if step is not None:
                checkpoint_steps.add(step)
    return {
        "sample_steps": sorted(sample_steps),
        "checkpoint_steps": sorted(checkpoint_steps),
        "sample_files": sorted(sample_files),
        "checkpoint_files": sorted(checkpoint_files),
    }

