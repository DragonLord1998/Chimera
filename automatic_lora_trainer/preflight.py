import json
import os
from pathlib import Path

import requests

from .settings import AI_TOOLKIT_DIR, FLUX2_PULID_COMMAND, WORK_ROOT, ZIMAGE_EXPANSION_COMMAND, ZIMAGE_IDENTITY_TRAIN_COMMAND


def check_item(key, label, ok, detail, required=True):
    return {
        "key": key,
        "label": label,
        "ok": bool(ok),
        "required": bool(required),
        "level": "ok" if ok else ("error" if required else "warn"),
        "detail": detail,
    }


def workflow_check():
    workflow = os.environ.get("FLUX2_PULID_WORKFLOW", "").strip()
    if not workflow:
        return check_item(
            "flux2_pulid_workflow",
            "Flux2-PuLID workflow",
            False,
            "FLUX2_PULID_WORKFLOW is not set.",
        )
    path = Path(workflow)
    if not path.exists():
        return check_item("flux2_pulid_workflow", "Flux2-PuLID workflow", False, f"Workflow not found: {path}")
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        return check_item("flux2_pulid_workflow", "Flux2-PuLID workflow", False, f"Workflow JSON is invalid: {exc}")
    if not isinstance(payload, dict) or not payload:
        return check_item("flux2_pulid_workflow", "Flux2-PuLID workflow", False, f"Workflow JSON has no ComfyUI API nodes: {path}")
    if payload.get("_comment") and len(payload) == 1:
        return check_item("flux2_pulid_workflow", "Flux2-PuLID workflow", False, f"Placeholder workflow must be replaced: {path}")
    return check_item("flux2_pulid_workflow", "Flux2-PuLID workflow", True, str(path))


def comfy_check():
    url = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
    try:
        response = requests.get(f"{url}/system_stats", timeout=2)
        response.raise_for_status()
    except Exception as exc:
        return check_item("comfyui", "ComfyUI API", False, f"{url} is not reachable: {exc}")
    return check_item("comfyui", "ComfyUI API", True, url)


def face_qc_check():
    try:
        import insightface  # noqa: F401
    except Exception as exc:
        return check_item("face_qc", "Face QC dependencies", False, f"insightface import failed: {exc}")
    return check_item("face_qc", "Face QC dependencies", True, "insightface import is available")


def ai_toolkit_check():
    run_py = AI_TOOLKIT_DIR / "run.py"
    return check_item(
        "ai_toolkit",
        "ai-toolkit trainer",
        run_py.exists(),
        str(run_py) if run_py.exists() else f"Missing {run_py}",
    )


def work_root_check():
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        probe = WORK_ROOT / ".chimera_write_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return check_item("work_root", "Work root", False, f"{WORK_ROOT} is not writable: {exc}")
    return check_item("work_root", "Work root", True, str(WORK_ROOT))


def generation_command_check():
    return check_item(
        "flux2_pulid_command",
        "Flux2-PuLID command",
        bool(FLUX2_PULID_COMMAND.strip()),
        FLUX2_PULID_COMMAND.strip() or "FLUX2_PULID_COMMAND is not set.",
    )


def zimage_identity_command_check():
    return check_item(
        "zimage_identity_train_command",
        "Z-Image identity LoRA command",
        bool(ZIMAGE_IDENTITY_TRAIN_COMMAND.strip()),
        ZIMAGE_IDENTITY_TRAIN_COMMAND.strip() or "ZIMAGE_IDENTITY_TRAIN_COMMAND is not set.",
    )


def zimage_expansion_command_check():
    return check_item(
        "zimage_expansion_command",
        "Z-Image expansion command",
        bool(ZIMAGE_EXPANSION_COMMAND.strip()),
        ZIMAGE_EXPANSION_COMMAND.strip() or "ZIMAGE_EXPANSION_COMMAND is not set.",
    )


def runtime_preflight():
    items = [
        work_root_check(),
        generation_command_check(),
        workflow_check(),
        comfy_check(),
        face_qc_check(),
        ai_toolkit_check(),
    ]
    full_items = items + [
        zimage_identity_command_check(),
        zimage_expansion_command_check(),
    ]
    blocking = [item for item in items if item["required"] and not item["ok"]]
    full_blocking = [item for item in full_items if item["required"] and not item["ok"]]
    return {
        "ready": not blocking,
        "full_ready": not full_blocking,
        "items": items,
        "full_items": full_items,
        "blocking": blocking,
        "full_blocking": full_blocking,
    }
