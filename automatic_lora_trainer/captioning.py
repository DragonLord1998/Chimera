from pathlib import Path

import pandas as pd

from .face_qc import face_embedding
from .paths import dirs, ensure_case, image_paths


def caption_for_image(path, trigger, base_caption):
    face_area = 0.0
    try:
        _, _, face_area = face_embedding(path)
    except Exception:
        pass
    if face_area > 0.14:
        framing = "close-up portrait photo"
    elif face_area > 0.045:
        framing = "portrait photo"
    elif face_area > 0.015:
        framing = "half body photo"
    else:
        framing = "full body photo"
    pieces = [f"{trigger} person", framing]
    if base_caption.strip():
        pieces.append(base_caption.strip().strip(","))
    return ", ".join(dict.fromkeys([p.strip() for p in pieces if p.strip()]))


def caption_curated(case_name, trigger, base_caption):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    imgs = image_paths(paths["train"])
    if not imgs:
        return "No curated training images found. Run auto-select first."
    for img in imgs:
        path = Path(img)
        path.with_suffix(".txt").write_text(caption_for_image(path, trigger, base_caption) + "\n")
    return f"Wrote {len(imgs)} caption files in {paths['train']}."


def preview_captions(case_name):
    case_name = ensure_case(case_name)
    rows = []
    for img in image_paths(dirs(case_name)["train"])[:80]:
        txt = Path(img).with_suffix(".txt")
        rows.append({"image": Path(img).name, "caption": txt.read_text(errors="replace").strip() if txt.exists() else ""})
    return pd.DataFrame(rows)

