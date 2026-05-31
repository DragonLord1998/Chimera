import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:
    import cv2
except Exception:
    cv2 = None

from .generation import read_candidate_metadata
from .media import qc_sheet, train_sheet
from .paths import dirs, ensure_case, image_paths
from .settings import FACE_MODEL, WORK_ROOT

FACE_APP = None


class FaceQcUnavailable(RuntimeError):
    pass


def laplacian_sharpness(path):
    if cv2 is None:
        return 0.0
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return 0.0
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())


def get_face_app():
    global FACE_APP
    if FACE_APP is not None:
        return FACE_APP
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise FaceQcUnavailable(
            "Face QC dependencies are not installed. Install insightface and onnxruntime-gpu "
            "or onnxruntime, or launch Colab with INSTALL_FACE_QC=1."
        ) from exc

    if FACE_MODEL == "auraface":
        from huggingface_hub import snapshot_download

        model_root = WORK_ROOT / "models"
        snapshot_download("fal/AuraFace-v1", local_dir=str(model_root / "auraface"))
        FACE_APP = FaceAnalysis(name="auraface", root=str(model_root.parent), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        FACE_APP = FaceAnalysis(name=FACE_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_APP


def load_rgb(path):
    with Image.open(path) as img:
        return np.array(ImageOps.exif_transpose(img).convert("RGB"))


def face_embedding(path):
    app = get_face_app()
    rgb = load_rgb(path)
    bgr = rgb[:, :, ::-1]
    faces = app.get(bgr)
    if not faces:
        return None, [], 0.0
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    face = faces[0]
    emb = np.array(face.normed_embedding, dtype=np.float32)
    emb = emb / max(float(np.linalg.norm(emb)), 1e-8)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = face.bbox
    face_area = max(0.0, float((x2 - x1) * (y2 - y1)) / float(w * h))
    return emb, faces, face_area


def cosine(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def score_image_folder(case_name, source_key, csv_name, label, identity_threshold, min_face_area):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    refs = image_paths(paths["refs"])
    candidates = image_paths(paths[source_key])
    if not refs:
        return "No reference images found.", None
    if not candidates:
        return f"No {label} images found.", None

    ref_embs = []
    ref_notes = []
    for ref in refs:
        emb, faces, face_area = face_embedding(ref)
        ref_notes.append(f"{Path(ref).name}: faces={len(faces)}, face_area={face_area:.4f}")
        if emb is not None:
            ref_embs.append(emb)
    if not ref_embs:
        return "No usable faces found in reference images.\n" + "\n".join(ref_notes), None

    ref_vec = np.mean(np.stack(ref_embs), axis=0)
    ref_vec = ref_vec / max(float(np.linalg.norm(ref_vec)), 1e-8)
    rows = []
    for path in candidates:
        pathp = Path(path)
        try:
            emb, faces, face_area = face_embedding(pathp)
            identity = cosine(ref_vec, emb) if emb is not None else -1.0
            face_count = len(faces)
            sharpness = laplacian_sharpness(pathp)
            with Image.open(pathp) as img:
                width, height = img.size
            passed = emb is not None and face_count == 1 and identity >= float(identity_threshold) and face_area >= float(min_face_area) and sharpness >= 35.0
            reason = "pass" if passed else "reject"
            if emb is None:
                reason = "no_face"
            elif face_count != 1:
                reason = "face_count"
            elif identity < float(identity_threshold):
                reason = "identity"
            elif face_area < float(min_face_area):
                reason = "tiny_face"
            elif sharpness < 35.0:
                reason = "blur"
            rows.append(
                {
                    "file": str(pathp),
                    "name": pathp.name,
                    "identity_score": round(identity, 5),
                    "face_count": face_count,
                    "face_area": round(face_area, 6),
                    "sharpness": round(sharpness, 2),
                    "width": width,
                    "height": height,
                    "passed": passed,
                    "reason": reason,
                }
            )
        except Exception as exc:
            rows.append({"file": str(pathp), "name": pathp.name, "identity_score": -1, "face_count": 0, "face_area": 0, "sharpness": 0, "width": 0, "height": 0, "passed": False, "reason": f"error: {exc}"})

    df = pd.DataFrame(rows).sort_values(["passed", "identity_score", "sharpness"], ascending=[False, False, False])
    out = paths["case"] / csv_name
    df.to_csv(out, index=False)
    summary = f"Scored {len(df)} {label}. Passed {int(df['passed'].sum())}. CSV: {out}\n" + "\n".join(ref_notes)
    return summary, df


def score_candidates(case_name, identity_threshold, min_face_area):
    return score_image_folder(case_name, "candidates", "qc_scores.csv", "candidates", identity_threshold, min_face_area)


def auto_select_from_csv(case_name, csv_name, dest_key, top_n, metadata=None):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    metadata = metadata or {}
    csv_path = paths["case"] / csv_name
    if not csv_path.exists():
        return "Run QC scoring first.", image_paths(paths[dest_key])
    df = pd.read_csv(csv_path)
    df = df[df["passed"] == True].sort_values(["identity_score", "sharpness"], ascending=[False, False]).head(int(top_n))
    for existing in paths[dest_key].glob("*"):
        if existing.is_file():
            existing.unlink()
    copied = 0
    for _, row in df.iterrows():
        src = Path(row["file"])
        if not src.exists():
            continue
        dest = paths[dest_key] / f"{copied + 1:04d}.png"
        with Image.open(src) as img:
            ImageOps.exif_transpose(img).convert("RGB").save(dest)
        source_metadata = metadata.get(str(src)) or metadata.get(src.name) or metadata.get(str(row.get("name", "")))
        if source_metadata:
            dest.with_suffix(".json").write_text(
                json.dumps({**source_metadata, "selected_source_file": str(src), "selected_name": dest.name}, indent=2, sort_keys=True)
            )
        copied += 1
    return f"Selected {copied} training images into {paths[dest_key]}.", image_paths(paths[dest_key])


def auto_select(case_name, top_n):
    return auto_select_from_csv(case_name, "qc_scores.csv", "train", top_n, read_candidate_metadata(case_name))


def score_select_pipeline(case_name, identity_threshold, min_face_area, top_n):
    score_status, df = score_candidates(case_name, identity_threshold, min_face_area)
    if df is None:
        return score_status, df, qc_sheet(case_name, top_n=top_n), train_sheet(case_name)
    select_status, _ = auto_select(case_name, top_n)
    return f"{score_status}\n{select_status}", df, qc_sheet(case_name, top_n=top_n), train_sheet(case_name)


def score_final_candidates(case_name, identity_threshold, min_face_area):
    return score_image_folder(case_name, "production_candidates", "final_qc_scores.csv", "Z-Image production candidates", identity_threshold, min_face_area)


def auto_select_final(case_name, top_n):
    return auto_select_from_csv(case_name, "final_qc_scores.csv", "final", top_n, read_candidate_metadata(case_name))


def score_select_final_pipeline(case_name, identity_threshold, min_face_area, top_n):
    score_status, df = score_final_candidates(case_name, identity_threshold, min_face_area)
    if df is None:
        return score_status, df, image_paths(dirs(case_name)["final"])
    select_status, selected = auto_select_final(case_name, top_n)
    return f"{score_status}\n{select_status}", df, selected
