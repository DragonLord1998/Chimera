"""
Chimera — Flask web server.

Routes:
  GET  /                         → serves index.html
  POST /api/start                → start pipeline in background, returns {job_id}
  GET  /api/stream/<job_id>      → SSE stream of progress events
  GET  /api/images/<job_id>/...  → serves generated images from job dir
  GET  /api/download/<job_id>    → downloads the final .safetensors file
"""

from __future__ import annotations

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

# Mock torchaudio if broken — AI Toolkit imports it but LoRA training doesn't need it.
# RunPod images often have a version compiled against a different torch, causing OSError.
import importlib.machinery
import sys
import types
try:
    import torchaudio  # noqa: F401
except (OSError, ImportError):
    _mock = types.ModuleType("torchaudio")
    _mock.__path__ = []
    _mock.__version__ = "0.0.0"
    _mock.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _mock
    for _sub in ("transforms", "functional", "_extension", "_extension.utils"):
        _sub_mod = types.ModuleType(f"torchaudio.{_sub}")
        _sub_mod.__spec__ = importlib.machinery.ModuleSpec(f"torchaudio.{_sub}", None)
        sys.modules[f"torchaudio.{_sub}"] = _sub_mod

# Mock removed huggingface_hub classes — AI Toolkit still imports them but they were
# dropped in huggingface_hub >= 1.0 (Repository, HfFolder, etc).
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "Repository"):
    class _MockRepository:
        def __init__(self, *a, **kw):
            raise NotImplementedError("Repository was removed in huggingface_hub >= 1.0")
    _hfhub.Repository = _MockRepository
if not hasattr(_hfhub, "HfFolder"):
    class _MockHfFolder:
        @classmethod
        def get_token(cls):
            return _hfhub.get_token() if hasattr(_hfhub, "get_token") else None
        @classmethod
        def save_token(cls, token):
            pass
    _hfhub.HfFolder = _MockHfFolder
# Also patch in huggingface_hub.utils where some code imports it from
try:
    import huggingface_hub.utils as _hfutils
    if not hasattr(_hfutils, "HfFolder"):
        _hfutils.HfFolder = _hfhub.HfFolder
except ImportError:
    pass

# Patch transformers 5.x: find_pruneable_heads_and_indices moved out of pytorch_utils.
# Florence 2's custom code and AI Toolkit still import from the old location.
try:
    from transformers import pytorch_utils as _pt_utils
    if not hasattr(_pt_utils, "find_pruneable_heads_and_indices"):
        # Try importing from the new location
        try:
            from transformers.utils.modeling_utils import find_pruneable_heads_and_indices
            _pt_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        except ImportError:
            # Provide a minimal fallback implementation
            import torch as _torch

            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = _torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head -= sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                index = _torch.arange(len(mask.view(-1)))[mask.view(-1).bool()]
                return heads, index

            _pt_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
except Exception:
    pass

# Patch transformers 5.x: load_backbone moved out of backbone_utils.
# Florence 2's custom modeling code still imports from the old location.
try:
    from transformers.utils import backbone_utils as _bb_utils
    if not hasattr(_bb_utils, "load_backbone"):
        try:
            from transformers import load_backbone
            _bb_utils.load_backbone = load_backbone
        except ImportError:
            def _load_backbone(config):
                from transformers import AutoBackbone
                return AutoBackbone.from_config(config)
            _bb_utils.load_backbone = _load_backbone
except Exception:
    pass

# Patch transformers 5.x: TokenizersBackend missing additional_special_tokens.
# Florence 2's custom processing code accesses this attribute on the tokenizer.
try:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend as _TokBackend
    if not hasattr(_TokBackend, "additional_special_tokens"):
        _TokBackend.additional_special_tokens = property(
            lambda self: getattr(self, "_additional_special_tokens", []),
            lambda self, v: setattr(self, "_additional_special_tokens", v),
        )
except ImportError:
    pass

# Patch AutoTokenizer.from_pretrained to auto-fallback to slow tokenizer.
# transformers 5.x TokenizersBackend fails to instantiate many tokenizers
# (T5, CLIP, etc.) that AI Toolkit and Florence 2 need.
try:
    from transformers import AutoTokenizer as _AutoTok
    _orig_from_pretrained = _AutoTok.from_pretrained.__func__

    @classmethod
    def _safe_from_pretrained(cls, *args, **kwargs):
        try:
            return _orig_from_pretrained(cls, *args, **kwargs)
        except (ValueError, OSError) as e:
            if "backend tokenizer" in str(e) or "slow tokenizer" in str(e):
                kwargs["use_fast"] = False
                return _orig_from_pretrained(cls, *args, **kwargs)
            raise

    _AutoTok.from_pretrained = _safe_from_pretrained
except Exception:
    pass

import datetime
import gc
import json
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="static")

# ---------------------------------------------------------------------------
# Global job store
# ---------------------------------------------------------------------------

# job_id -> { "history": [...], "subscribers": [Queue, ...],
#              "lora_path": str | None, "job_dir": str, "status": str,
#              "params": dict }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

JOBS_DIR = os.environ.get("JOBS_DIR", "/workspace/character_jobs")
MODELS_DIR = os.environ.get("MODELS_DIR", "/workspace/models")
TOOLKIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai-toolkit")
SEEDVR2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SeedVR2-CLI")


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


# ---------------------------------------------------------------------------
# API: start pipeline
# ---------------------------------------------------------------------------


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    # Accept multipart/form-data
    image_file = request.files.get("image")
    has_image = image_file is not None and image_file.filename

    # Collect params (all optional with defaults)
    params = {
        "trigger_word": request.form.get("trigger_word", "chrx").strip() or "chrx",
        "gemini_key": request.form.get("gemini_key", "").strip(),
        "hf_token": request.form.get("hf_token", "").strip() or None,
        "num_images": int(request.form.get("num_images", 25)),
        "lora_rank": int(request.form.get("lora_rank", 32)),
        "lora_steps": int(request.form.get("lora_steps", 1500)),
        "learning_rate": float(request.form.get("learning_rate", 1e-4)),
        "inference_steps": int(request.form.get("inference_steps", 50)),
        "base_model": request.form.get("base_model", "zimage").strip(),
        "sample_prompts": None,
    }

    # Parse sample prompts — one per line, replace TRIGGER with trigger word.
    raw_prompts = request.form.get("sample_prompts", "").strip()
    if raw_prompts:
        lines = [l.strip() for l in raw_prompts.splitlines() if l.strip()]
        params["sample_prompts"] = [
            l.replace("TRIGGER", params["trigger_word"]) for l in lines
        ]

    # Views zip is optional — if provided, skip Gemini generation
    views_zip = request.files.get("views_zip")
    has_views_zip = views_zip is not None and views_zip.filename

    # Dataset zip is optional — if provided, skip synthesis + captioning
    dataset_zip = request.files.get("dataset_zip")
    has_dataset_zip = dataset_zip is not None and dataset_zip.filename

    if not has_image and not has_views_zip and not has_dataset_zip:
        return jsonify({"error": "Upload a character image, views zip, or dataset zip"}), 400

    if not has_views_zip and not has_dataset_zip and not params["gemini_key"]:
        return jsonify({"error": "Gemini API key is required (or upload a views/dataset zip)"}), 400

    # Create job
    job_id = str(uuid.uuid4())[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Save uploaded image (optional when views zip is provided)
    image_path = None
    if has_image:
        ext = os.path.splitext(image_file.filename)[1] or ".png"
        image_path = os.path.join(job_dir, f"input{ext}")
        image_file.save(image_path)

    # Save views zip if provided
    views_zip_path = None
    if has_views_zip:
        views_zip_path = os.path.join(job_dir, "views.zip")
        views_zip.save(views_zip_path)

    # Save dataset zip if provided
    dataset_zip_path = None
    if has_dataset_zip:
        dataset_zip_path = os.path.join(job_dir, "dataset.zip")
        dataset_zip.save(dataset_zip_path)

    # Register job with event history and subscriber list
    with _jobs_lock:
        _jobs[job_id] = {
            "history": [],
            "subscribers": [],
            "lora_path": None,
            "job_dir": job_dir,
            "status": "running",
            "params": {
                "num_images": params["num_images"],
                "lora_steps": params["lora_steps"],
            },
        }

    # Launch pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, image_path, job_dir, params, views_zip_path, dataset_zip_path),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    thread.start()

    return jsonify({"job_id": job_id})


# ---------------------------------------------------------------------------
# API: SSE stream
# ---------------------------------------------------------------------------


@app.route("/api/stream/<job_id>")
def stream(job_id: str):
    def generate():
        with _jobs_lock:
            job = _jobs.get(job_id)
            if not job:
                yield _sse("error", {"message": f"Unknown job: {job_id}"})
                return

            # Snapshot history and subscribe atomically — no event gap
            history = list(job["history"])
            sub_q: queue.Queue = queue.Queue()
            job["subscribers"].append(sub_q)

        # Replay all past events so the UI rebuilds its state
        for event in history:
            yield _sse(event["type"], event["data"])

        # If job already finished, no need to wait for live events
        if job["status"] in ("complete", "error"):
            with _jobs_lock:
                if sub_q in job["subscribers"]:
                    job["subscribers"].remove(sub_q)
            return

        # Stream live events
        try:
            while True:
                try:
                    event = sub_q.get(timeout=30)
                    if event is None:  # sentinel — pipeline finished
                        break
                    yield _sse(event["type"], event["data"])
                except queue.Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            with _jobs_lock:
                if sub_q in job.get("subscribers", []):
                    job["subscribers"].remove(sub_q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.route("/api/jobs/active")
def active_job():
    """Return the most recent running job, if any."""
    with _jobs_lock:
        for job_id, job in reversed(list(_jobs.items())):
            if job["status"] == "running":
                return jsonify({
                    "job_id": job_id,
                    "status": "running",
                    "params": job.get("params", {}),
                })
    return jsonify({"job_id": None})


# ---------------------------------------------------------------------------
# API: serve generated images
# ---------------------------------------------------------------------------


@app.route("/api/images/<job_id>/<path:subpath>")
def serve_image(job_id: str, subpath: str):
    job_base = os.path.join(JOBS_DIR, job_id)
    return send_from_directory(job_base, subpath)


# ---------------------------------------------------------------------------
# API: download final LoRA
# ---------------------------------------------------------------------------


@app.route("/api/download/<job_id>")
def download_lora(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404

    lora_path = job.get("lora_path")
    if not lora_path or not os.path.isfile(lora_path):
        # Try to discover it from disk
        job_dir = job.get("job_dir", "")
        lora_path = _find_lora(job_dir)

    if not lora_path or not os.path.isfile(lora_path):
        return jsonify({"error": "LoRA file not found"}), 404

    return send_file(lora_path, as_attachment=True,
                     download_name=os.path.basename(lora_path))


# ---------------------------------------------------------------------------
# API: download views as zip
# ---------------------------------------------------------------------------


@app.route("/api/download-views/<job_id>")
def download_views(job_id: str):
    import zipfile
    import io

    stage1_dir = os.path.join(JOBS_DIR, job_id, "stage1")
    if not os.path.isdir(stage1_dir):
        return jsonify({"error": "No views found for this job"}), 404

    images = [f for f in os.listdir(stage1_dir)
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    if not images:
        return jsonify({"error": "No view images found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(images):
            zf.write(os.path.join(stage1_dir, fname), fname)
    buf.seek(0)

    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=chimera_views_{job_id}.zip"},
    )


# ---------------------------------------------------------------------------
# API: download dataset as zip
# ---------------------------------------------------------------------------


@app.route("/api/download-dataset/<job_id>")
def download_dataset(job_id: str):
    import zipfile
    import io

    dataset_dir = os.path.join(JOBS_DIR, job_id, "dataset")
    if not os.path.isdir(dataset_dir):
        return jsonify({"error": "No dataset found for this job"}), 404

    files = [f for f in os.listdir(dataset_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".txt"))]
    if not files:
        return jsonify({"error": "No dataset files found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(files):
            zf.write(os.path.join(dataset_dir, fname), fname)
    buf.seek(0)

    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=chimera_dataset_{job_id}.zip"},
    )


# ---------------------------------------------------------------------------
# Pipeline background thread
# ---------------------------------------------------------------------------


def _run_pipeline(
    job_id: str,
    image_path: str | None,
    job_dir: str,
    params: dict,
    views_zip_path: str | None = None,
    dataset_zip_path: str | None = None,
) -> None:
    """Full pipeline: model check → multi-view → synthesize → caption → train."""

    def emit(event_type: str, data: dict, ephemeral: bool = False) -> None:
        event = {"type": event_type, "data": data}
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                if not ephemeral:
                    job["history"].append(event)
                for sub_q in job["subscribers"]:
                    sub_q.put(event)

    def stage_msg(stage: int, status: str) -> None:
        emit("stage", {"stage": stage, "status": status})

    try:
        from PIL import Image

        # Subdirectories
        stage1_dir = os.path.join(job_dir, "stage1")
        dataset_dir = os.path.join(job_dir, "dataset")
        output_dir = os.path.join(job_dir, "output")
        for d in (stage1_dir, dataset_dir, output_dir):
            os.makedirs(d, exist_ok=True)

        trigger = params["trigger_word"]
        output_name = f"{trigger}_lora"

        # ------------------------------------------------------------------
        # Stage 0 — Model download
        # ------------------------------------------------------------------
        stage_msg(0, "Checking and downloading required models...")
        from stages.model_manager import ModelManager

        mm = ModelManager(base_path=MODELS_DIR, hf_token=params["hf_token"])

        # Download models based on selected base model
        base_model = params.get("base_model", "zimage")
        if base_model == "flux_krea":
            # Flux Krea only needs its own model + Florence 2 for captioning
            for key in ("florence2", "flux_krea"):
                if not mm.is_model_ready(key):
                    mm._download_with_retry(key)
        else:
            mm.ensure_all_models()

        # ------------------------------------------------------------------
        # Fast path: dataset ZIP provided — skip stages 1 and 2
        # ------------------------------------------------------------------
        if dataset_zip_path and os.path.isfile(dataset_zip_path):
            import zipfile
            stage_msg(2, "Extracting uploaded dataset...")
            with zipfile.ZipFile(dataset_zip_path, "r") as zf:
                zf.extractall(dataset_dir)

            # Count extracted images for UI
            dataset_imgs = sorted([
                f for f in os.listdir(dataset_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ])
            for i, fname in enumerate(dataset_imgs):
                emit("synthetic", {
                    "index": i,
                    "url": f"/api/images/{job_id}/dataset/{fname}",
                })
            stage_msg(2, f"Dataset loaded from zip — {len(dataset_imgs)} images.")

            # Check if captions exist — if not, run Florence 2
            caption_files = [
                f for f in os.listdir(dataset_dir)
                if f.lower().endswith(".txt")
            ]
            if len(caption_files) < len(dataset_imgs):
                stage_msg(2, f"No captions found — running Florence 2 on {len(dataset_imgs)} images...")
                from stages.caption import CaptionGenerator

                cap = CaptionGenerator(model_path=mm.get_model_path("florence2"))
                cap.load_model()
                cap.caption_dataset(dataset_dir, trigger)
                cap.unload_model()
                del cap
                gc.collect()
                stage_msg(2, "Captioning complete.")
            else:
                stage_msg(2, f"Found {len(caption_files)} existing captions — skipping Florence 2.")

        else:
            # Full pipeline: multi-view → synthesis → captioning

            # ------------------------------------------------------------------
            # Stage 1 — Multi-view generation (Gemini) or extract from zip
            # ------------------------------------------------------------------
            view_paths: list[str] = []
            view_images: list = []
            view_names = ["left", "front", "right", "face", "back"]

            if views_zip_path and os.path.isfile(views_zip_path):
                import zipfile
                stage_msg(1, "Extracting uploaded multi-view images...")
                with zipfile.ZipFile(views_zip_path, "r") as zf:
                    zf.extractall(stage1_dir)

                # Find view images — accept left/front/right.png or any 3 images
                for vn in view_names:
                    candidate = os.path.join(stage1_dir, f"{vn}.png")
                    if not os.path.isfile(candidate):
                        candidate = None
                    if candidate and os.path.isfile(candidate):
                        view_paths.append(candidate)
                        view_images.append(Image.open(candidate).convert("RGB"))
                        emit("view", {
                            "position": vn,
                            "url": f"/api/images/{job_id}/stage1/{vn}.png",
                        })

                # If exact names not found, grab any images from the zip
                if len(view_paths) < 3:
                    view_paths = []
                    view_images = []
                    all_imgs = sorted([
                        f for f in os.listdir(stage1_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                    ])
                    for i, fname in enumerate(all_imgs[:5]):
                        vn = view_names[i] if i < len(view_names) else f"view{i}"
                        src = os.path.join(stage1_dir, fname)
                        dst = os.path.join(stage1_dir, f"{vn}.png")
                        img = Image.open(src).convert("RGB")
                        img.save(dst, format="PNG")
                        view_paths.append(dst)
                        view_images.append(img)
                        emit("view", {
                            "position": vn,
                            "url": f"/api/images/{job_id}/stage1/{vn}.png",
                        })

                if len(view_paths) < 3:
                    raise RuntimeError(
                        f"Views zip must contain at least 3 images, found {len(view_paths)}"
                    )
                stage_msg(1, "Multi-view images loaded from zip.")

            else:
                stage_msg(1, "Generating multi-view character images with Gemini...")
                from stages.multiview import MultiViewGenerator

                pil_input = Image.open(image_path).convert("RGB")
                mv = MultiViewGenerator(api_key=params["gemini_key"])

                resized = MultiViewGenerator._resize(pil_input)

                for view_name, prompt in MultiViewGenerator.VIEWS.items():
                    stage_msg(1, f"Generating {view_name} view...")
                    generated = mv._generate_view_with_retry(resized, prompt, view_name)
                    path = os.path.join(stage1_dir, f"{view_name}.png")
                    generated.save(path, format="PNG")
                    view_paths.append(path)
                    view_images.append(generated)
                    emit("view", {
                        "position": view_name,
                        "url": f"/api/images/{job_id}/stage1/{view_name}.png",
                    })

                pil_input.close()

            # ------------------------------------------------------------------
            # Stage 2 — Dataset synthesis (Flux 2)
            # ------------------------------------------------------------------
            stage_msg(2, "Loading Flux 2 DEV — this takes a moment...")
            from stages.synthesize import DatasetSynthesizer

            synth = DatasetSynthesizer(
                hf_token=params["hf_token"],
            )
            synth.load_model()
            stage_msg(2, f"Synthesizing {params['num_images']} training images...")

            num_images = params["num_images"]

            def synthesis_progress(current: int, total: int) -> None:
                filename = f"img_{current:03d}.png"
                emit("synthetic", {
                    "index": current - 1,
                    "url": f"/api/images/{job_id}/dataset/{filename}",
                })

            def synthesis_preview(image_index: int, step: int, total_steps: int, preview_img) -> None:
                """Send a latent preview as base64 JPEG via ephemeral SSE."""
                import base64
                import io as _io
                buf = _io.BytesIO()
                preview_img.save(buf, format="JPEG", quality=60)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                emit("diffusion_preview", {
                    "index": image_index,
                    "step": step,
                    "total_steps": total_steps,
                    "preview": f"data:image/jpeg;base64,{b64}",
                }, ephemeral=True)

            synth.synthesize_dataset(
                reference_images=view_images,
                output_dir=dataset_dir,
                num_images=num_images,
                start_from=0,
                progress_callback=synthesis_progress,
                num_inference_steps=params["inference_steps"],
                preview_callback=synthesis_preview,
            )
            synth.unload_model()
            del synth
            for img in view_images:
                img.close()
            gc.collect()

            # ------------------------------------------------------------------
            # Stage 2a — SeedVR2 Upscale (1024→2048)
            # ------------------------------------------------------------------
            if os.path.isdir(SEEDVR2_PATH):
                stage_msg(2, "Upscaling dataset to 2048px with SeedVR2 7B...")
                from stages.upscale import ImageUpscaler

                def upscale_progress(current: int, total: int) -> None:
                    stage_msg(2, f"Upscaling image {current}/{total}...")

                def upscale_image_done(index: int, orig_rel: str, upscaled_rel: str) -> None:
                    emit("upscaled", {
                        "index": index,
                        "original_url": f"/api/images/{job_id}/dataset/{orig_rel}",
                        "upscaled_url": f"/api/images/{job_id}/dataset/{upscaled_rel}",
                    })

                upscaler = ImageUpscaler(cli_dir=SEEDVR2_PATH)
                upscaler.upscale_dataset(
                    dataset_dir=dataset_dir,
                    target_resolution=2048,
                    progress_callback=upscale_progress,
                    image_callback=upscale_image_done,
                )
                del upscaler
                gc.collect()
            else:
                print("[Chimera] SeedVR2 CLI not found — skipping upscale stage.")

            # ------------------------------------------------------------------
            # Stage 2b — Captioning (Florence 2)
            # ------------------------------------------------------------------
            stage_msg(2, "Captioning dataset with Florence 2...")
            from stages.caption import CaptionGenerator

            cap = CaptionGenerator(model_path=mm.get_model_path("florence2"))
            cap.load_model()
            cap.caption_dataset(dataset_dir, trigger)
            cap.unload_model()
            del cap
            gc.collect()

        # ------------------------------------------------------------------
        # Stage 3 — LoRA training (AI Toolkit / Z-Image)
        # ------------------------------------------------------------------
        model_label = "FLUX.1-Krea-dev" if base_model == "flux_krea" else "Z-Image De-Turbo"
        model_key = "flux_krea" if base_model == "flux_krea" else "zimage_base"
        stage_msg(3, f"Starting LoRA training with {model_label}...")
        from stages.train import LoRATrainer

        trainer = LoRATrainer(
            model_path=mm.get_model_path(model_key),
            toolkit_path=TOOLKIT_PATH,
            base_model=base_model,
        )

        total_steps = params["lora_steps"]
        save_every = 250  # checkpoint / sample interval

        samples_dir = os.path.join(output_dir, output_name, "samples")
        checkpoint_dir_path = os.path.join(output_dir, output_name)
        _stop_watcher = threading.Event()

        num_sample_prompts = len(params.get("sample_prompts") or [
            "default1", "default2",
        ])

        # Shared state: stderr capture writes current step, watcher reads it
        _tqdm_step = [0]  # [current_step] — mutable for thread safety

        import re as _re
        _TQDM_RE = _re.compile(r'\b(\d+)/(\d+)\s*\[')

        class _StderrCapture:
            """Wraps stderr to parse tqdm progress from AI Toolkit in real time."""
            def __init__(self, real):
                self._real = real

            def write(self, s):
                self._real.write(s)
                m = _TQDM_RE.search(s)
                if m:
                    _tqdm_step[0] = int(m.group(1))
                return len(s)

            def flush(self):
                self._real.flush()

            def __getattr__(self, name):
                return getattr(self._real, name)

        def _watcher():
            """Poll tqdm progress, checkpoint files, and sample images."""
            seen_checkpoints: set[str] = set()
            seen_samples: set[str] = set()
            step_estimate = 0
            last_emitted_step = -1
            # Buffer: step_num -> {"urls": [...], "first_seen": timestamp}
            pending_samples: dict[int, dict] = {}

            while not _stop_watcher.is_set():
                time.sleep(2)

                # 1. Real-time step from tqdm stderr capture
                tqdm_current = _tqdm_step[0]
                if tqdm_current > step_estimate:
                    step_estimate = tqdm_current
                if step_estimate != last_emitted_step:
                    emit("progress", {"step": step_estimate, "total": total_steps})
                    last_emitted_step = step_estimate

                # 2. Checkpoint .safetensors files (for accurate step on save)
                if os.path.isdir(checkpoint_dir_path):
                    ckpts = sorted(
                        f for f in os.listdir(checkpoint_dir_path)
                        if f.endswith(".safetensors")
                    )
                    new_ckpts = [c for c in ckpts if c not in seen_checkpoints]
                    for ckpt_name in new_ckpts:
                        seen_checkpoints.add(ckpt_name)

                # 3. Sample images (AI Toolkit outputs .jpg)
                if os.path.isdir(samples_dir):
                    sample_files = sorted(
                        f for f in os.listdir(samples_dir)
                        if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))
                        and f not in seen_samples
                    )
                    for sf in sample_files:
                        seen_samples.add(sf)
                        url = f"/api/images/{job_id}/output/{output_name}/samples/{sf}"
                        # Parse step from filename: {timestamp}__{step:09d}_{idx}.ext
                        parsed_step = -1
                        try:
                            parts = sf.rsplit(".", 1)[0]
                            step_part = parts.split("__")[1].split("_")[0]
                            parsed_step = int(step_part)
                        except (IndexError, ValueError):
                            pass

                        if parsed_step not in pending_samples:
                            pending_samples[parsed_step] = {
                                "urls": [], "first_seen": time.time(),
                            }
                        pending_samples[parsed_step]["urls"].append(url)

                # 4. Emit checkpoint events for completed sample groups
                now = time.time()
                done_steps = []
                for s_step, info in sorted(pending_samples.items()):
                    have_all = len(info["urls"]) >= num_sample_prompts
                    timed_out = (now - info["first_seen"]) > 60
                    if have_all or timed_out:
                        emit("checkpoint", {
                            "step": s_step if s_step >= 0 else step_estimate,
                            "images": info["urls"],
                        })
                        done_steps.append(s_step)

                for s in done_steps:
                    del pending_samples[s]

        watcher_thread = threading.Thread(target=_watcher, daemon=True,
                                          name=f"watcher-{job_id}")
        watcher_thread.start()

        # Capture stderr to parse tqdm progress in real time
        _real_stderr = sys.stderr
        sys.stderr = _StderrCapture(_real_stderr)

        try:
            lora_path = trainer.train(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                output_name=output_name,
                trigger_word=trigger,
                rank=params["lora_rank"],
                learning_rate=params["learning_rate"],
                steps=total_steps,
                save_every=save_every,
                sample_every=save_every,
                sample_prompts=params["sample_prompts"],
            )
        finally:
            sys.stderr = _real_stderr

        _stop_watcher.set()
        watcher_thread.join(timeout=5)

        trainer.cleanup()
        del trainer
        gc.collect()

        # Store lora_path in job record
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["lora_path"] = lora_path

        # Final progress = 100%
        emit("progress", {"step": total_steps, "total": total_steps})

        emit("complete", {
            "lora_path": lora_path,
            "download_url": f"/api/download/{job_id}",
        })

        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "complete"

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[server] Pipeline error for job {job_id}:\n{tb}", file=sys.stderr)
        emit("error", {"message": str(exc)})

        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"

    finally:
        # Send sentinel to all subscribers so SSE generators stop
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                for sub_q in job["subscribers"]:
                    sub_q.put(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_lora(job_dir: str) -> str | None:
    """Walk job_dir/output recursively to find the final .safetensors file."""
    output_dir = os.path.join(job_dir, "output")
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(Path(output_dir).rglob("*.safetensors"))
    return str(candidates[-1]) if candidates else None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    debug = os.environ.get("DEBUG", "0") == "1"
    print(f"[server] Starting Chimera on http://0.0.0.0:{port}")
    # threaded=True is required for SSE + concurrent requests
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
