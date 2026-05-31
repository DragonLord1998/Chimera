import datetime as dt
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .media import candidates_sheet, qc_sheet, refs_sheet
from .paths import dirs, ensure_case, image_paths
from .settings import FLUX2_PULID_COMMAND, IMAGE_EXTS
from .state import RUNNING, env_for_case, log_path, read_log


class GenerationUnavailable(RuntimeError):
    pass


PROMPT_BUCKETS = [
    {
        "bucket": "front_portrait",
        "prompt": "[trigger] person, front-facing portrait photo, neutral background, soft studio lighting",
        "caption_tags": ["front-facing portrait photo", "neutral background", "soft studio lighting"],
    },
    {
        "bucket": "three_quarter",
        "prompt": "[trigger] person, three-quarter portrait photo, natural daylight, simple background",
        "caption_tags": ["three-quarter portrait photo", "natural daylight", "simple background"],
    },
    {
        "bucket": "side_profile",
        "prompt": "[trigger] person, side profile portrait photo, clean background, soft light",
        "caption_tags": ["side profile portrait photo", "clean background", "soft light"],
    },
    {
        "bucket": "half_body",
        "prompt": "[trigger] person, half body photo, relaxed pose, indoor natural light",
        "caption_tags": ["half body photo", "relaxed pose", "indoor natural light"],
    },
    {
        "bucket": "outdoor",
        "prompt": "[trigger] person, portrait photo, outdoor daylight, shallow depth of field",
        "caption_tags": ["portrait photo", "outdoor daylight", "shallow depth of field"],
    },
    {
        "bucket": "editorial",
        "prompt": "[trigger] person, fashion editorial portrait photo, controlled studio lighting",
        "caption_tags": ["fashion editorial portrait photo", "controlled studio lighting"],
    },
]


def metadata_path(case_name):
    return dirs(case_name)["case"] / "candidate_metadata.jsonl"


def prompt_manifest_path(case_name):
    return dirs(case_name)["case"] / "flux2_pulid_prompt_manifest.json"


def read_candidate_metadata(case_name):
    case_name = ensure_case(case_name)
    path = metadata_path(case_name)
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key in (record.get("file"), record.get("name"), record.get("output_name")):
            if key:
                records[str(key)] = record
    return records


def write_candidate_metadata(case_name, rows):
    case_name = ensure_case(case_name)
    path = metadata_path(case_name)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def build_flux2_pulid_manifest(case_name, trigger, count):
    case_name = ensure_case(case_name)
    rows = []
    for idx in range(int(count)):
        bucket = PROMPT_BUCKETS[idx % len(PROMPT_BUCKETS)]
        output_name = f"pulid_{idx + 1:04d}.png"
        prompt = bucket["prompt"].replace("[trigger]", trigger)
        rows.append(
            {
                "index": idx + 1,
                "backend": "flux2_pulid",
                "bucket": bucket["bucket"],
                "output_name": output_name,
                "prompt": prompt,
                "caption_tags": bucket["caption_tags"],
                "seed": 42000 + idx,
            }
        )
    path = prompt_manifest_path(case_name)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return path, rows


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


def flux2_pulid_generate(case_name, count, trigger="zphchar"):
    case_name = ensure_case(case_name)
    if not FLUX2_PULID_COMMAND.strip():
        raise GenerationUnavailable(
            "Flux2-PuLID generation is not configured. Set FLUX2_PULID_COMMAND so Chimera can create "
            "identity-preserving synthetic candidates from REF_DIR into CANDIDATE_DIR using PROMPT_MANIFEST."
        )

    paths = dirs(case_name)
    refs = image_paths(paths["refs"])
    if not refs:
        return "No refs found. Upload refs first.", image_paths(paths["candidates"])

    manifest, rows = build_flux2_pulid_manifest(case_name, trigger, count)
    env = env_for_case(case_name, trigger=trigger, count=count)
    env["PROMPT_MANIFEST"] = str(manifest)
    env["GENERATION_BACKEND"] = "flux2_pulid"

    lp = log_path(case_name, "generate")
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("ab", buffering=0) as handle:
        handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Starting Flux2-PuLID generation\n".encode())
        handle.write((FLUX2_PULID_COMMAND + "\n\n").encode())
        proc = subprocess.run(
            FLUX2_PULID_COMMAND,
            shell=True,
            executable="/bin/bash",
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(paths["case"]),
            env=env,
            check=False,
        )
        handle.write(f"\n[{dt.datetime.now().isoformat()}] Flux2-PuLID command exited with code {proc.returncode}\n".encode())

    if proc.returncode != 0:
        raise GenerationUnavailable(f"Flux2-PuLID command failed with exit code {proc.returncode}. See {lp}.")

    candidates = image_paths(paths["candidates"])
    by_name = {Path(path).name: str(path) for path in candidates}
    generated_rows = []
    for row in rows:
        output_name = row["output_name"]
        if output_name in by_name:
            generated_rows.append({**row, "name": output_name, "file": by_name[output_name]})
    if generated_rows:
        write_candidate_metadata(case_name, generated_rows)
    if not candidates:
        raise GenerationUnavailable(
            f"Flux2-PuLID command finished but no images were found in {paths['candidates']}."
        )
    return f"Generated {len(candidates)} Flux2-PuLID candidate image(s). Manifest: {manifest}", candidates


def smoke_generate(case_name, count):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    refs = image_paths(paths["refs"])
    if not refs:
        return "No refs found. Upload refs first.", image_paths(paths["candidates"])
    made = 0
    metadata_rows = []
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
            dest = paths["candidates"] / f"smoke_{idx + 1:04d}.png"
            canvas.save(dest)
            metadata_rows.append(
                {
                    "backend": "local_reference_augmentation",
                    "bucket": "dev_smoke",
                    "caption_tags": ["portrait photo", "neutral background"],
                    "file": str(dest),
                    "name": dest.name,
                    "output_name": dest.name,
                    "prompt": "reference-preserving local augmentation smoke test",
                    "source_ref": str(src),
                }
            )
            made += 1
    if metadata_rows:
        write_candidate_metadata(case_name, metadata_rows)
    return f"Generated {made} candidate image(s) with the built-in reference augmentation generator.", image_paths(paths["candidates"])


def generate_candidates(case_name, count, trigger="zphchar", backend="flux2_pulid", allow_smoke_fallback=False):
    if backend == "local_smoke":
        return smoke_generate(case_name, count)
    try:
        return flux2_pulid_generate(case_name, count, trigger)
    except GenerationUnavailable:
        if not allow_smoke_fallback:
            raise
        return smoke_generate(case_name, count)


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
