import datetime as dt
from pathlib import Path

from .settings import IMAGE_EXTS, WORK_ROOT


def now_slug():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_slug(value):
    value = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    value = value.strip("_")
    return value[:80] or "case"


def case_dir(case_name):
    return WORK_ROOT / "cases" / case_name


def dirs(case_name):
    root = case_dir(case_name)
    result = {
        "case": root,
        "refs": root / "refs",
        "candidates": root / "candidates",
        "identity_lora": root / "identity_lora",
        "production_candidates": root / "production_candidates",
        "curated": root / "curated",
        "train": root / "curated" / "train",
        "final": root / "curated" / "final",
        "rejected": root / "rejected",
        "logs": root / "logs",
        "output": root / "output",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def ensure_case(case_name):
    case_name = case_name or "default"
    dirs(case_name)
    return case_name


def list_cases():
    base = WORK_ROOT / "cases"
    base.mkdir(parents=True, exist_ok=True)
    cases = [p.name for p in base.iterdir() if p.is_dir()]
    ordered = sorted([case for case in cases if case != "default"], reverse=True)
    if "default" in cases:
        ordered.append("default")
    return ordered or ["default"]


def create_case(label):
    name = f"{now_slug()}_{clean_slug(label or 'character')}"
    dirs(name)
    return name


def image_paths(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    return [str(p) for p in sorted(paths)]
