"""
ModelManager — auto-downloads all required models from HuggingFace.

Supports both single-file downloads (hf_hub_download) and full-repo
snapshot downloads (snapshot_download). Resumes partial downloads
automatically after pod restart.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    # --- Flux 2 DEV (full diffusers-format repo from Comfy-Org, public) ---
    "flux2": {
        "repo_id": "Comfy-Org/flux2-dev",
        "subdir": "flux2",
        "description": "Flux 2 DEV (diffusers format, includes transformer + VAE + text encoder)",
        "snapshot": True,
        "ignore_patterns": ["split_files/**"],
        "ready_subdir": "transformer",
        "size_hint": "~25 GB",
    },
    # --- Florence 2 (captioning) ---
    "florence2": {
        "repo_id": "microsoft/Florence-2-large",
        "subdir": "florence2",
        "description": "Florence 2 Large captioning model",
        "snapshot": True,
        "size_hint": "~2 GB",
    },
    # --- Z-Image (LoRA training base) ---
    # Transformer from ostris De-Turbo + text encoder from official Tongyi repo.
    # Both download to the same subdir so AI Toolkit sees a complete model.
    "zimage_base": {
        "repo_id": "ostris/Z-Image-De-Turbo",
        "subdir": "z_image",
        "description": "Z-Image De-Turbo transformer for LoRA training",
        "snapshot": True,
        "size_hint": "~12 GB",
    },
    "zimage_text_enc": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "subdir": "z_image",
        "description": "Z-Image text encoder (Qwen3-based, from official Z-Image repo)",
        "snapshot": True,
        "allow_patterns": ["text_encoder/**"],
        "ready_subdir": "text_encoder",
        "size_hint": "~8 GB",
    },
}

_RETRY_COUNT = 3
_RETRY_BASE_DELAY = 5  # seconds


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


class ModelManager:
    """Download and locate ComfyUI models required by Chimera.

    Parameters
    ----------
    base_path:
        Root directory where model subdirectories are created.
        Defaults to ``/workspace/models/``.
    hf_token:
        Optional HuggingFace access token for gated repositories.
    """

    def __init__(
        self,
        base_path: str = "/workspace/models/",
        hf_token: Optional[str] = None,
    ) -> None:
        self.base_path = os.path.abspath(base_path)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_all_models(self) -> None:
        """Check every registered model and download any that are missing."""
        print("[ModelManager] Checking all required models...")
        missing: list[str] = [
            key for key in MODEL_REGISTRY if not self.is_model_ready(key)
        ]

        if not missing:
            print("[ModelManager] All models already present. Nothing to download.")
            return

        print(f"[ModelManager] {len(missing)} model(s) to download: {', '.join(missing)}")
        for key in missing:
            self._download_with_retry(key)

        print("[ModelManager] All models ready.")

    def get_model_path(self, model_key: str) -> str:
        """Return the absolute path to a model file or snapshot directory.

        Parameters
        ----------
        model_key:
            A key from ``MODEL_REGISTRY``.

        Returns
        -------
        str
            Full path to the model file (single-file) or directory (snapshot).

        Raises
        ------
        KeyError
            If ``model_key`` is not in the registry.
        """
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"[ModelManager] Unknown model key: '{model_key}'")

        spec = MODEL_REGISTRY[model_key]
        subdir_path = os.path.join(self.base_path, spec["subdir"])

        if spec.get("snapshot"):
            return subdir_path

        # Single-file downloads with repo_subfolder end up nested.
        if spec.get("repo_subfolder"):
            return os.path.join(subdir_path, spec["repo_subfolder"], spec["filename"])

        return os.path.join(subdir_path, spec["filename"])

    def is_model_ready(self, model_key: str) -> bool:
        """Return True if the model file or snapshot directory already exists.

        Parameters
        ----------
        model_key:
            A key from ``MODEL_REGISTRY``.
        """
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"[ModelManager] Unknown model key: '{model_key}'")

        path = self.get_model_path(model_key)
        spec = MODEL_REGISTRY[model_key]

        if spec.get("snapshot"):
            # For partial snapshots (e.g. text_encoder only), check the specific subdir.
            if spec.get("ready_subdir"):
                check = os.path.join(path, spec["ready_subdir"])
                return os.path.isdir(check) and bool(os.listdir(check))
            # Consider ready if the directory exists and is non-empty
            return os.path.isdir(path) and bool(os.listdir(path))

        return os.path.isfile(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_with_retry(self, model_key: str) -> None:
        """Attempt to download a model, retrying up to ``_RETRY_COUNT`` times."""
        spec = MODEL_REGISTRY[model_key]
        label = spec["description"]
        size = spec.get("size_hint", "unknown size")

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                print(
                    f"[ModelManager] Downloading {label}... ({size})"
                    + (f" [attempt {attempt}/{_RETRY_COUNT}]" if attempt > 1 else "")
                )
                self._download(model_key, spec)
                print(f"[ModelManager] {label} — done.")
                return
            except (EntryNotFoundError, RepositoryNotFoundError) as exc:
                # These are fatal — no point retrying.
                print(f"[ModelManager] ERROR: {label} not found on HuggingFace: {exc}")
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt < _RETRY_COUNT:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"[ModelManager] WARNING: Download attempt {attempt} failed for "
                        f"'{model_key}': {exc}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[ModelManager] ERROR: All {_RETRY_COUNT} attempts failed for "
                        f"'{model_key}': {exc}"
                    )
                    raise

    def _download(self, model_key: str, spec: dict) -> None:
        """Dispatch to the appropriate HuggingFace download function."""
        subdir_path = os.path.join(self.base_path, spec["subdir"])
        os.makedirs(subdir_path, exist_ok=True)

        common_kwargs: dict = {
            "repo_id": spec["repo_id"],
            "local_dir": subdir_path,
        }
        if self.hf_token:
            common_kwargs["token"] = self.hf_token

        if spec.get("snapshot"):
            base_ignore = ["*.msgpack", "*.h5", "flax_model*", "tf_model*"]
            extra_ignore = spec.get("ignore_patterns", [])
            snapshot_kwargs = {
                **common_kwargs,
                "resume_download": True,
                "ignore_patterns": base_ignore + extra_ignore,
            }
            if spec.get("allow_patterns"):
                snapshot_kwargs["allow_patterns"] = spec["allow_patterns"]
            snapshot_download(**snapshot_kwargs)
        else:
            dl_kwargs = {
                **common_kwargs,
                "filename": spec["filename"],
                "resume_download": True,
            }
            if spec.get("repo_subfolder"):
                dl_kwargs["subfolder"] = spec["repo_subfolder"]
            hf_hub_download(**dl_kwargs)
