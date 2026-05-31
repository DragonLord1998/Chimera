import json
from pathlib import Path

import pandas as pd

from .face_qc import face_embedding
from .paths import dirs, ensure_case, image_paths


def split_tags(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace("|", ",").split(",")
    return [str(item).strip().strip(".") for item in raw if str(item).strip()]


def read_train_metadata(path):
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(errors="replace"))
    except Exception:
        return {}


def framing_for_image(path):
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
        return "half body photo"
    return "full body photo"


def is_framing_tag(tag):
    text = tag.lower()
    return any(token in text for token in ("portrait", "profile", "body photo", "front-facing", "three-quarter", "close-up"))


def caption_for_image(path, trigger, base_caption, metadata=None):
    metadata = metadata or {}
    metadata_tags = split_tags(metadata.get("caption_tags"))
    metadata_framing = [tag for tag in metadata_tags if is_framing_tag(tag)]
    metadata_context = [tag for tag in metadata_tags if not is_framing_tag(tag)]
    pieces = [f"{trigger} person"]
    pieces.extend(metadata_framing or [framing_for_image(path)])
    pieces.extend(metadata_context)
    if base_caption.strip():
        pieces.extend(split_tags(base_caption))

    deduped = []
    seen = set()
    for piece in pieces:
        cleaned = piece.strip().strip(",")
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return ", ".join(deduped)


def caption_curated(case_name, trigger, base_caption):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    imgs = image_paths(paths["train"])
    if not imgs:
        return "No curated training images found. Run auto-select first."
    for img in imgs:
        path = Path(img)
        path.with_suffix(".txt").write_text(caption_for_image(path, trigger, base_caption, read_train_metadata(path)) + "\n")
    return f"Wrote {len(imgs)} caption files in {paths['train']}."


def preview_captions(case_name):
    case_name = ensure_case(case_name)
    rows = []
    for img in image_paths(dirs(case_name)["train"])[:80]:
        txt = Path(img).with_suffix(".txt")
        rows.append({"image": Path(img).name, "caption": txt.read_text(errors="replace").strip() if txt.exists() else ""})
    return pd.DataFrame(rows)
