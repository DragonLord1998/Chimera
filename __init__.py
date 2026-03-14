"""
Chimera — ComfyUI custom node package.

Takes a single character image and produces a trained Z-Image LoRA,
designed to run on RunPod with an NVIDIA A40 (48 GB VRAM).
"""

try:
    from .nodes import Chimera

    NODE_CLASS_MAPPINGS = {
        "Chimera": Chimera,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "Chimera": "Chimera",
    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

except Exception as exc:
    print(
        f"[Chimera] WARNING: Failed to import node — dependencies may not be installed yet. "
        f"Error: {exc}"
    )

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
