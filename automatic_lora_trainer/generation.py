import datetime as dt
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .media import candidates_sheet, qc_sheet, refs_sheet
from .paths import dirs, ensure_case, image_paths
from .settings import IMAGE_EXTS
from .state import RUNNING, env_for_case, log_path, read_log


def save_refs(case_name, files, consent):
    case_name = ensure_case(case_name)
    if not consent:
        return "Consent/rights checkbox must be enabled before saving reference images.", image_paths(dirs(case_name)["refs"])
    if not files:
        return "Upload at least one reference image.", image_paths(dirs(case_name)["refs"])
    paths = dirs(case_name)
    saved = 0
    for item in files:
        src = Path(getattr(item, "name", str(item)))
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dest = paths["refs"] / f"ref_{saved + 1:02d}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        saved += 1
    return f"Saved {saved} reference image(s) into {paths['refs']}.", image_paths(paths["refs"])


def save_refs_pipeline(case_name, files, consent):
    status, _ = save_refs(case_name, files, consent)
    return status, refs_sheet(case_name)


def start_background(case_name, name, command, trigger="zphchar", count=200):
    case_name = ensure_case(case_name)
    if not command.strip():
        return f"No {name} command provided.", read_log(case_name, name)
    key = (case_name, name)
    existing = RUNNING.get(key)
    if existing and existing.poll() is None:
        return f"{name} is already running with PID {existing.pid}.", read_log(case_name, name)

    lp = log_path(case_name, name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    env = env_for_case(case_name, trigger=trigger, count=count)
    with lp.open("ab", buffering=0) as handle:
        handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Starting {name}\n".encode())
        handle.write((command + "\n\n").encode())
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(dirs(case_name)["case"]),
            env=env,
        )
    RUNNING[key] = proc
    return f"Started {name} with PID {proc.pid}.", read_log(case_name, name)


def smoke_generate(case_name, count):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    refs = image_paths(paths["refs"])
    if not refs:
        return "No refs found. Upload refs first.", image_paths(paths["candidates"])
    made = 0
    for idx in range(int(count)):
        src = Path(random.choice(refs))
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.15))
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.4)))
            canvas = Image.new("RGB", (1024, 1024), (245, 245, 245))
            canvas.paste(img, ((1024 - img.width) // 2, (1024 - img.height) // 2))
            canvas.save(paths["candidates"] / f"smoke_{idx + 1:04d}.png")
            made += 1
    return f"Created {made} smoke-test candidates. Replace with real FLUX/ComfyUI generation for production.", image_paths(paths["candidates"])


def smoke_generate_pipeline(case_name, count):
    status, _ = smoke_generate(case_name, count)
    return status, candidates_sheet(case_name), qc_sheet(case_name, top_n=100)


def import_candidates(case_name, source_folder, copy_limit):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    source = Path(source_folder).expanduser()
    if not source.exists():
        return f"Source folder not found: {source}", image_paths(paths["candidates"])
    candidates = image_paths(source)
    if copy_limit:
        candidates = candidates[: int(copy_limit)]
    copied = 0
    for src in candidates:
        srcp = Path(src)
        dest = paths["candidates"] / f"import_{copied + 1:04d}{srcp.suffix.lower()}"
        shutil.copy2(srcp, dest)
        copied += 1
    return f"Imported {copied} image(s) from {source}.", image_paths(paths["candidates"])


def import_candidates_pipeline(case_name, source_folder, copy_limit):
    status, _ = import_candidates(case_name, source_folder, copy_limit)
    return status, candidates_sheet(case_name), qc_sheet(case_name, top_n=100)

