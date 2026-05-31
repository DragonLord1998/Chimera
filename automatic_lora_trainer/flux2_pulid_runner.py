import json
import os
import time
import uuid
from pathlib import Path

import requests


COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
POLL_SECONDS = float(os.environ.get("COMFY_POLL_SECONDS", "2"))
TIMEOUT_SECONDS = float(os.environ.get("COMFY_TIMEOUT_SECONDS", "900"))


def require_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def load_workflow_template():
    path_value = os.environ.get("FLUX2_PULID_WORKFLOW", "").strip()
    if not path_value:
        raise RuntimeError("FLUX2_PULID_WORKFLOW is required and must point to a ComfyUI API workflow JSON file.")
    path = Path(path_value)
    if not path.exists():
        raise RuntimeError(f"FLUX2_PULID_WORKFLOW not found: {path}")
    return json.loads(path.read_text(errors="replace"))


def walk_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_nodes(child)


def set_first_matching_input(workflow, names, value):
    names = {name.lower() for name in names}
    matches = 0
    for node in walk_nodes(workflow):
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key in list(inputs):
            if key.lower() in names:
                inputs[key] = value
                matches += 1
    return matches


def set_prompt_inputs(workflow, prompt, ref_image, output_prefix, seed):
    prompt_hits = set_first_matching_input(workflow, {"text", "prompt", "positive", "positive_prompt"}, prompt)
    ref_hits = set_first_matching_input(workflow, {"image", "reference_image", "pulid_image", "face_image"}, ref_image)
    prefix_hits = set_first_matching_input(workflow, {"filename_prefix", "prefix"}, output_prefix)
    seed_hits = set_first_matching_input(workflow, {"seed", "noise_seed"}, int(seed))
    return {"prompt": prompt_hits, "reference": ref_hits, "prefix": prefix_hits, "seed": seed_hits}


def upload_image(path):
    with Path(path).open("rb") as handle:
        response = requests.post(
            f"{COMFY_URL}/upload/image",
            files={"image": (Path(path).name, handle, "image/png")},
            data={"overwrite": "true"},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    return payload.get("name") or Path(path).name


def queue_prompt(workflow):
    client_id = str(uuid.uuid4())
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["prompt_id"]


def wait_for_prompt(prompt_id):
    start = time.time()
    while time.time() - start < TIMEOUT_SECONDS:
        response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=120)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")


def download_outputs(history, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    for node in history.get("outputs", {}).values():
        for image in node.get("images", []):
            params = {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
            response = requests.get(f"{COMFY_URL}/view", params=params, timeout=120)
            response.raise_for_status()
            target = destination / image["filename"]
            target.write_bytes(response.content)
            saved.append(target)
    return saved


def main():
    ref_dir = Path(require_env("REF_DIR"))
    candidate_dir = Path(require_env("CANDIDATE_DIR"))
    manifest_path = Path(require_env("PROMPT_MANIFEST"))

    refs = sorted(path for path in ref_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not refs:
        raise RuntimeError(f"No reference images found in {ref_dir}")
    reference_name = upload_image(refs[0])
    manifest = json.loads(manifest_path.read_text(errors="replace"))

    total_saved = 0
    for row in manifest:
        workflow = load_workflow_template()
        output_name = row["output_name"]
        output_prefix = Path(output_name).stem
        replacements = set_prompt_inputs(workflow, row["prompt"], reference_name, output_prefix, row.get("seed", 42))
        if replacements["prompt"] == 0:
            raise RuntimeError("Workflow template has no prompt/text input to replace.")
        if replacements["reference"] == 0:
            raise RuntimeError("Workflow template has no reference image input to replace.")
        prompt_id = queue_prompt(workflow)
        history = wait_for_prompt(prompt_id)
        saved = download_outputs(history, candidate_dir)
        if not saved:
            raise RuntimeError(f"ComfyUI prompt {prompt_id} finished with no image outputs.")
        latest = max(saved, key=lambda path: path.stat().st_mtime)
        final_path = candidate_dir / output_name
        if latest != final_path:
            latest.replace(final_path)
        print(json.dumps({"output": str(final_path), "prompt_id": prompt_id, "bucket": row.get("bucket")}))
        total_saved += 1

    print(f"Flux2-PuLID ComfyUI generation complete: {total_saved} image(s)")


if __name__ == "__main__":
    main()
