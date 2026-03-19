"""
ModelManager — auto-downloads all required models from HuggingFace.

Supports both single-file downloads (hf_hub_download) and full-repo
snapshot downloads (snapshot_download). Resumes partial downloads
automatically after pod restart.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    # --- Flux 2 DEV is NOT pre-downloaded here. ---
    # It uses a gated BFL repo (black-forest-labs/FLUX.2-dev) and is loaded
    # directly via Flux2Pipeline.from_pretrained() which handles download
    # and caching through HuggingFace Hub. HF token is required.
    #
    # --- Florence 2 (captioning) ---
    "florence2": {
        "repo_id": "florence-community/Florence-2-large",
        "subdir": "florence2",
        "description": "Florence 2 Large captioning model (community, native transformers 5.x)",
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
    "zimage_tokenizer": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "subdir": "z_image",
        "description": "Z-Image Qwen3 tokenizer (from official Z-Image repo)",
        "snapshot": True,
        "allow_patterns": ["tokenizer/**"],
        "ready_subdir": "tokenizer",
        "size_hint": "~7 MB",
    },
    "zimage_vae": {
        "repo_id": "Tongyi-MAI/Z-Image-Turbo",
        "subdir": "z_image",
        "description": "Z-Image VAE (AutoencoderKL, from official Z-Image repo)",
        "snapshot": True,
        "allow_patterns": ["vae/**"],
        "ready_subdir": "vae",
        "size_hint": "~335 MB",
    },
    # --- FLUX.1-Krea-dev (alternative training base) ---
    "flux_krea": {
        "repo_id": "black-forest-labs/FLUX.1-Krea-dev",
        "subdir": "flux_krea",
        "description": "FLUX.1-Krea-dev (BFL + Krea AI aesthetic fine-tune)",
        "snapshot": True,
        "size_hint": "~24 GB",
        "gated": True,
    },
    # --- FLUX.2 Klein 9B KV (alternative synthesizer) ---
    # Step-distilled 9B model with KV-cache for fast multi-reference generation.
    # Downloaded on demand when user selects Klein as synthesizer.
    "flux2_klein_kv": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9b-kv",
        "subdir": "flux2_klein_kv",
        "description": "FLUX.2 Klein 9B KV (step-distilled, KV-cache for multi-ref)",
        "snapshot": True,
        "size_hint": "~29 GB",
        "gated": True,
    },
    # --- SRPO Base Model (photorealism enhancement) ---
    # Tencent's Semantic Relative Preference Optimization full fine-tune of FLUX.1-dev.
    # This is a complete transformer replacement (NOT a LoRA) — architecturally identical
    # to FLUX.1-dev but with different weights trained for photorealism.
    # Used during dataset enhancement as the img2img pipeline base.
    # Downloaded on demand when user enables Enhanced Mode.
    "srpo_base": {
        "repo_id": "rockerBOO/flux.1-dev-SRPO",
        "subdir": "srpo_base",
        "filename": "flux.1-dev-SRPO-bf16.safetensors",
        "description": "FLUX.1-dev SRPO full transformer (rockerBOO, BF16, 23.8 GB)",
        "snapshot": False,
        "size_hint": "~23.8 GB",
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
        progress_callback=None,
    ) -> None:
        self.base_path = os.path.abspath(base_path)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.progress_callback = progress_callback

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

        For sharded models (those with a ``.safetensors.index.json``), also
        verifies that every referenced shard file is present on disk.

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
            if not os.path.isdir(path) or not os.listdir(path):
                return False
            # Verify sharded model completeness: if an index.json exists,
            # check that all referenced shard files are present.
            if not self._verify_shards(path):
                return False
            return True

        return os.path.isfile(path)

    def _verify_shards(self, model_dir: str) -> bool:
        """Walk model_dir (recursively) looking for ``*.index.json`` files.

        For each index found, verify every shard it references exists on disk.
        Returns False if any shard is missing.
        """
        import json

        for dirpath, _dirs, files in os.walk(model_dir):
            for fname in files:
                if not fname.endswith(".index.json"):
                    continue
                index_path = os.path.join(dirpath, fname)
                try:
                    with open(index_path, "r") as fh:
                        index_data = json.load(fh)
                    weight_map = index_data.get("weight_map", {})
                    shard_files = set(weight_map.values())
                    for shard in shard_files:
                        shard_path = os.path.join(dirpath, shard)
                        if not os.path.isfile(shard_path):
                            print(
                                f"[ModelManager] Missing shard: {shard_path} "
                                f"(referenced by {index_path})"
                            )
                            return False
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"[ModelManager] WARNING: Could not verify {index_path}: {exc}")
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size_hint(hint: str) -> int:
        """Convert a human size hint like ``'~12 GB'`` to bytes."""
        if not hint:
            return 0
        m = re.match(r"~?([\d.]+)\s*(GB|MB|KB)", hint, re.IGNORECASE)
        if not m:
            return 0
        val = float(m.group(1))
        unit = m.group(2).upper()
        multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
        return int(val * multipliers.get(unit, 1))

    @staticmethod
    def _path_size(path: str) -> int:
        """Recursively compute total bytes under *path* (follows symlinks)."""
        if not os.path.exists(path):
            return 0
        if os.path.isfile(path):
            try:
                return os.path.getsize(path)
            except OSError:
                return 0
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def _download_with_retry(self, model_key: str) -> None:
        """Attempt to download a model, retrying up to ``_RETRY_COUNT`` times."""
        spec = MODEL_REGISTRY[model_key]
        label = spec["description"]
        size = spec.get("size_hint", "unknown size")

        # Determine path to monitor for progress
        subdir_path = os.path.join(self.base_path, spec["subdir"])
        if spec.get("ready_subdir"):
            monitor_path = os.path.join(subdir_path, spec["ready_subdir"])
        elif not spec.get("snapshot") and spec.get("filename"):
            monitor_path = subdir_path  # single-file: monitor parent dir
        else:
            monitor_path = subdir_path

        expected_bytes = self._parse_size_hint(size)
        initial_size = self._path_size(monitor_path)

        # Background thread for progress monitoring
        stop_event = threading.Event()
        monitor_thread = None

        if self.progress_callback and expected_bytes > 0:
            self.progress_callback(model_key, label, 0, size, "downloading")

            # Poll faster for small models, slower for large ones to avoid I/O pressure
            poll_interval = 2.0 if expected_bytes < 1024**3 else 5.0

            def _monitor():
                while not stop_event.wait(poll_interval):
                    current = self._path_size(monitor_path) - initial_size
                    pct = min(current / expected_bytes * 100, 99) if expected_bytes else 0
                    if pct < 0:
                        pct = 0
                    self.progress_callback(model_key, label, pct, size, "downloading")

            monitor_thread = threading.Thread(target=_monitor, daemon=True)
            monitor_thread.start()

        for attempt in range(1, _RETRY_COUNT + 1):
            try:
                print(
                    f"[ModelManager] Downloading {label}... ({size})"
                    + (f" [attempt {attempt}/{_RETRY_COUNT}]" if attempt > 1 else "")
                )
                self._download(model_key, spec)
                print(f"[ModelManager] {label} — done.")

                stop_event.set()
                if monitor_thread:
                    monitor_thread.join(timeout=3)
                if self.progress_callback:
                    self.progress_callback(model_key, label, 100, size, "complete")
                return
            except (EntryNotFoundError, RepositoryNotFoundError) as exc:
                stop_event.set()
                if monitor_thread:
                    monitor_thread.join(timeout=3)
                if self.progress_callback:
                    self.progress_callback(model_key, label, 0, size, "error")
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
                    stop_event.set()
                    if monitor_thread:
                        monitor_thread.join(timeout=3)
                    if self.progress_callback:
                        self.progress_callback(model_key, label, 0, size, "error")
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
