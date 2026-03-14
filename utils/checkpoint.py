"""
CheckpointManager — resume-from-crash support for Chimera jobs
running on RunPod (or any machine that may restart mid-job).

Checkpoint files are JSON documents written to:
    {job_dir}/checkpoints/{stage_name}.json

Each file records at minimum a UTC timestamp and an optional metadata dict
so that consumers can resume work without re-running completed stages.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Optional


# Sentinel value returned when no progress has been recorded yet.
_NO_PROGRESS = 0


class CheckpointManager:
    """Manage per-stage checkpoint markers for a single job directory.

    Parameters
    ----------
    job_dir:
        Root directory for this job, e.g.
        ``/workspace/character_jobs/chr7x_20260313/``.
        A ``checkpoints/`` subdirectory is created automatically.
    """

    def __init__(self, job_dir: str) -> None:
        self.job_dir = os.path.abspath(job_dir)
        self.checkpoint_dir = os.path.join(self.job_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage completion
    # ------------------------------------------------------------------

    def is_stage_complete(self, stage: str) -> bool:
        """Return True if a checkpoint marker file exists for *stage*."""
        return os.path.isfile(self._checkpoint_path(stage))

    def mark_stage_complete(
        self,
        stage: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Write a checkpoint marker file for *stage*.

        Parameters
        ----------
        stage:
            Stage identifier, e.g. ``"stage0_models_ready"`` or
            ``"stage1_complete"``.
        metadata:
            Optional dict of extra data to persist alongside the timestamp.
        """
        payload: dict = {
            "completed_at": _utc_now(),
            "stage": stage,
        }
        if metadata:
            payload["metadata"] = metadata

        path = self._checkpoint_path(stage)
        _write_json(path, payload)
        print(f"[CheckpointManager] Stage '{stage}' marked complete → {path}")

    def get_stage_metadata(self, stage: str) -> dict:
        """Return the metadata dict stored for *stage*, or ``{}`` if absent.

        Parameters
        ----------
        stage:
            Stage identifier matching a previous :meth:`mark_stage_complete` call.
        """
        path = self._checkpoint_path(stage)
        if not os.path.isfile(path):
            return {}
        payload = _read_json(path)
        return payload.get("metadata", {})

    # ------------------------------------------------------------------
    # Intra-stage progress (used by Stage 2 image generation)
    # ------------------------------------------------------------------

    def get_resume_point(self, stage: str) -> int:
        """Return how many items were completed within *stage* last run.

        Reads the ``current`` field written by :meth:`update_progress`.
        Returns 0 if no progress file exists yet.
        """
        path = self._progress_path(stage)
        if not os.path.isfile(path):
            return _NO_PROGRESS
        payload = _read_json(path)
        return int(payload.get("current", _NO_PROGRESS))

    def update_progress(self, stage: str, current: int, total: int) -> None:
        """Persist intra-stage progress so work can resume mid-stage.

        Parameters
        ----------
        stage:
            Stage identifier, e.g. ``"stage2_generation"``.
        current:
            Number of items completed so far (0-indexed count).
        total:
            Total number of items in this stage.
        """
        payload: dict = {
            "stage": stage,
            "current": current,
            "total": total,
            "updated_at": _utc_now(),
        }
        path = self._progress_path(stage)
        _write_json(path, payload)

    # ------------------------------------------------------------------
    # Job directory scaffolding
    # ------------------------------------------------------------------

    def create_job_dir(self) -> str:
        """Create standard subdirectories under *job_dir* and return *job_dir*.

        Subdirectories created:
        - ``stage1/``   — raw/intermediate Stage 1 outputs
        - ``dataset/``  — curated training dataset images + captions
        - ``output/``   — final LoRA weights and artefacts
        """
        subdirs = ["stage1", "dataset", "output"]
        for name in subdirs:
            path = os.path.join(self.job_dir, name)
            os.makedirs(path, exist_ok=True)
        print(f"[CheckpointManager] Job directory scaffolded at {self.job_dir}")
        return self.job_dir

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove the checkpoints directory after successful job completion.

        The job output directories (stage1/, dataset/, output/) are left
        intact — only the checkpoint bookkeeping files are removed.
        """
        if os.path.isdir(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            print(f"[CheckpointManager] Checkpoints removed: {self.checkpoint_dir}")
        else:
            print("[CheckpointManager] No checkpoint directory found; nothing to clean.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, stage: str) -> str:
        """Return the path for a stage completion marker file."""
        safe = _safe_filename(stage)
        return os.path.join(self.checkpoint_dir, f"{safe}.json")

    def _progress_path(self, stage: str) -> str:
        """Return the path for a stage intra-progress file."""
        safe = _safe_filename(stage)
        return os.path.join(self.checkpoint_dir, f"{safe}_progress.json")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_filename(stage: str) -> str:
    """Sanitise *stage* so it is safe to use as a filename component."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in stage)


def _write_json(path: str, payload: dict) -> None:
    """Atomically write *payload* as JSON to *path* via a temp file."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Clean up temp file on failure before re-raising.
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_json(path: str) -> dict:
    """Read and return the JSON object at *path*, or ``{}`` on parse error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[CheckpointManager] WARNING: Could not read checkpoint file {path}: {exc}")
        return {}
