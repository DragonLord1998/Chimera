import datetime as dt
import json
import subprocess
from pathlib import Path

from .generation import PROMPT_BUCKETS, write_candidate_metadata
from .paths import dirs, ensure_case, image_paths
from .settings import ZIMAGE_EXPANSION_COMMAND, ZIMAGE_IDENTITY_TRAIN_COMMAND
from .state import env_for_case, log_path


class ZImageUnavailable(RuntimeError):
    pass


def zimage_manifest_path(case_name):
    return dirs(case_name)["case"] / "zimage_expansion_manifest.json"


def zimage_metadata_path(case_name):
    return dirs(case_name)["case"] / "zimage_candidate_metadata.jsonl"


def identity_lora_artifacts(case_name):
    paths = dirs(case_name)
    suffixes = {".safetensors", ".pt", ".ckpt"}
    return sorted(str(path) for path in paths["identity_lora"].rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def build_zimage_manifest(case_name, trigger, count):
    case_name = ensure_case(case_name)
    rows = []
    for idx in range(int(count)):
        bucket = PROMPT_BUCKETS[idx % len(PROMPT_BUCKETS)]
        output_name = f"zimage_{idx + 1:04d}.png"
        prompt = bucket["prompt"].replace("[trigger]", trigger)
        rows.append(
            {
                "index": idx + 1,
                "backend": "zimage_turbo_identity_lora",
                "bucket": bucket["bucket"],
                "output_name": output_name,
                "prompt": prompt,
                "caption_tags": bucket["caption_tags"],
                "seed": 52000 + idx,
            }
        )
    path = zimage_manifest_path(case_name)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    return path, rows


def run_stage_command(case_name, log_name, command, trigger, count, extra_env=None):
    case_name = ensure_case(case_name)
    if not command.strip():
        raise ZImageUnavailable(f"{log_name} is not configured.")
    paths = dirs(case_name)
    env = env_for_case(case_name, trigger=trigger, count=count)
    env.update(extra_env or {})
    lp = log_path(case_name, log_name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("ab", buffering=0) as handle:
        handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Starting {log_name}\n".encode())
        handle.write((command + "\n\n").encode())
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(paths["case"]),
            env=env,
            check=False,
        )
        handle.write(f"\n[{dt.datetime.now().isoformat()}] {log_name} exited with code {proc.returncode}\n".encode())
    if proc.returncode != 0:
        raise ZImageUnavailable(f"{log_name} failed with exit code {proc.returncode}. See {lp}.")
    return lp


def train_zimage_identity_lora(case_name, trigger="zphchar", count=1, training_env=None):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    if not ZIMAGE_IDENTITY_TRAIN_COMMAND.strip():
        raise ZImageUnavailable("ZIMAGE_IDENTITY_TRAIN_COMMAND is not set.")
    train_images = image_paths(paths["train"])
    if not train_images:
        raise ZImageUnavailable("No strict-QC seed training images found. Run Flux2-PuLID generation and strict QC first.")
    run_stage_command(
        case_name,
        "zimage_identity_lora",
        ZIMAGE_IDENTITY_TRAIN_COMMAND,
        trigger,
        max(1, int(count or len(train_images))),
        training_env,
    )
    artifacts = identity_lora_artifacts(case_name)
    if not artifacts:
        raise ZImageUnavailable(
            f"Z-Image identity LoRA command finished but no LoRA artifact was found in {paths['identity_lora']}."
        )
    return f"Z-Image identity LoRA ready: {Path(artifacts[-1]).name}", artifacts


def expand_with_zimage_identity_lora(case_name, trigger="zphchar", count=5000):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    if not ZIMAGE_EXPANSION_COMMAND.strip():
        raise ZImageUnavailable("ZIMAGE_EXPANSION_COMMAND is not set.")
    artifacts = identity_lora_artifacts(case_name)
    if not artifacts:
        raise ZImageUnavailable("No Z-Image identity LoRA artifact found. Train the identity LoRA stage first.")
    manifest, rows = build_zimage_manifest(case_name, trigger, count)
    run_stage_command(
        case_name,
        "zimage_expansion",
        ZIMAGE_EXPANSION_COMMAND,
        trigger,
        count,
        {
            "ZIMAGE_PROMPT_MANIFEST": str(manifest),
            "ZIMAGE_IDENTITY_LORA": artifacts[-1],
        },
    )
    candidates = image_paths(paths["production_candidates"])
    by_name = {Path(path).name: str(path) for path in candidates}
    metadata_rows = []
    for row in rows:
        output_name = row["output_name"]
        if output_name in by_name:
            metadata_rows.append({**row, "name": output_name, "file": by_name[output_name], "identity_lora": artifacts[-1]})
    if metadata_rows:
        zimage_metadata_path(case_name).write_text("\n".join(json.dumps(row, sort_keys=True) for row in metadata_rows) + "\n")
        write_candidate_metadata(case_name, metadata_rows)
    if not candidates:
        raise ZImageUnavailable(
            f"Z-Image expansion command finished but no images were found in {paths['production_candidates']}."
        )
    return f"Generated {len(candidates)} Z-Image production candidate image(s). Manifest: {manifest}", candidates
