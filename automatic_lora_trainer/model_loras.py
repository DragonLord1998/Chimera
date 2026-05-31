import datetime as dt
import subprocess
from pathlib import Path

from .paths import dirs, ensure_case, image_paths
from .settings import MODEL_LORA_COMMAND
from .state import env_for_case, log_path
from .zimage import identity_lora_artifacts


DEFAULT_MODEL_TARGETS = ["flux", "zimage_base", "wan", "ltx"]


class ModelLoraUnavailable(RuntimeError):
    pass


def model_lora_artifacts(case_name):
    paths = dirs(case_name)
    suffixes = {".safetensors", ".pt", ".ckpt"}
    return sorted(str(path) for path in paths["model_loras"].rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def normalize_targets(targets):
    if not targets:
        return DEFAULT_MODEL_TARGETS
    if isinstance(targets, str):
        raw = targets.replace("|", ",").split(",")
    else:
        raw = targets
    cleaned = []
    seen = set()
    for target in raw:
        value = str(target).strip().lower().replace(" ", "_").replace("-", "_")
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned or DEFAULT_MODEL_TARGETS


def run_model_lora_generation(case_name, trigger="zphchar", targets=None, training_env=None):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    if not MODEL_LORA_COMMAND.strip():
        raise ModelLoraUnavailable("MODEL_LORA_COMMAND is not set.")
    final_images = image_paths(paths["final"])
    if not final_images:
        raise ModelLoraUnavailable("No final curated dataset found. Run very-strict final QC first.")

    target_list = normalize_targets(targets)
    env = env_for_case(case_name, trigger=trigger, count=len(final_images))
    env.update(training_env or {})
    env["MODEL_TARGETS"] = ",".join(target_list)
    identity_artifacts = identity_lora_artifacts(case_name)
    if identity_artifacts:
        env["ZIMAGE_IDENTITY_LORA"] = identity_artifacts[-1]

    lp = log_path(case_name, "model_loras")
    lp.parent.mkdir(parents=True, exist_ok=True)
    with lp.open("ab", buffering=0) as handle:
        handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Starting final model LoRA generation\n".encode())
        handle.write((MODEL_LORA_COMMAND + "\n\n").encode())
        proc = subprocess.run(
            MODEL_LORA_COMMAND,
            shell=True,
            executable="/bin/bash",
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(paths["case"]),
            env=env,
            check=False,
        )
        handle.write(f"\n[{dt.datetime.now().isoformat()}] model_loras exited with code {proc.returncode}\n".encode())

    if proc.returncode != 0:
        raise ModelLoraUnavailable(f"MODEL_LORA_COMMAND failed with exit code {proc.returncode}. See {lp}.")

    artifacts = model_lora_artifacts(case_name)
    if not artifacts:
        raise ModelLoraUnavailable(
            f"MODEL_LORA_COMMAND finished but no LoRA artifact was found in {paths['model_loras']}."
        )
    return f"Generated {len(artifacts)} final model LoRA artifact(s) for {', '.join(target_list)}.", artifacts
