from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .captioning import caption_curated, preview_captions
from .dashboard import expected_interval_steps, training_progress
from .face_qc import FaceQcUnavailable, score_candidates, score_select_pipeline
from .generation import GenerationUnavailable, generate_candidates, import_candidates, smoke_generate, start_background
from .media import latest_sample_image, selected_from_qc, training_artifact_steps
from .paths import clean_slug, create_case, dirs, ensure_case, image_paths, list_cases
from .preflight import runtime_preflight
from .settings import AI_TOOLKIT_DIR, IMAGE_EXTS, WORK_ROOT
from .state import read_log, read_training_state
from .system import system_stats
from .training import TrainingUnavailable, build_ai_toolkit_config, prepare_pipeline_training, start_pipeline_training


FIXED_IDENTITY_THRESHOLD = 0.92
DEFAULT_PIPELINE_COUNT = 10
DEFAULT_PIPELINE_TOP_N = 10
DEFAULT_TRIGGER = "zphchar"
DEFAULT_BASE_CAPTION = ""
DEFAULT_GENERATION_BACKEND = "flux2_pulid"
DEFAULT_MODEL_NAME = "black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_RANK = 32
DEFAULT_STEPS = 1200
DEFAULT_LR = "1e-4"
DEFAULT_SAMPLE_EVERY = 200
DEFAULT_SAVE_EVERY = 250
DEFAULT_SAMPLE_PROMPTS = "\n".join(
    [
        "zphchar person, professional portrait, soft studio lighting",
        "zphchar person, close-up portrait, neutral background",
        "zphchar person, three-quarter portrait, natural daylight",
        "zphchar person, side profile portrait, clean background",
    ]
)


class CaseCreate(BaseModel):
    label: str = "character"


class SmokeRequest(BaseModel):
    count: int = DEFAULT_PIPELINE_COUNT


class ImportRequest(BaseModel):
    source_folder: str
    copy_limit: int = 200


class GenerationRequest(BaseModel):
    command: str
    trigger: str = DEFAULT_TRIGGER
    count: int = DEFAULT_PIPELINE_COUNT


class QcRequest(BaseModel):
    identity_threshold: float = FIXED_IDENTITY_THRESHOLD
    min_face_area: float = 0.01
    top_n: int = DEFAULT_PIPELINE_TOP_N


class CaptionRequest(BaseModel):
    trigger: str = DEFAULT_TRIGGER
    base_caption: str = DEFAULT_BASE_CAPTION


class TrainingPrepareRequest(BaseModel):
    trigger: str = DEFAULT_TRIGGER
    base_caption: str = DEFAULT_BASE_CAPTION
    model_name: str = DEFAULT_MODEL_NAME
    rank: int = DEFAULT_RANK
    steps: int = DEFAULT_STEPS
    lr: str = DEFAULT_LR
    sample_prompts: str = DEFAULT_SAMPLE_PROMPTS
    sample_every: int = DEFAULT_SAMPLE_EVERY
    save_every: int = DEFAULT_SAVE_EVERY


class TrainingStartRequest(BaseModel):
    config_path: str
    trigger: str = DEFAULT_TRIGGER
    steps: int = DEFAULT_STEPS
    sample_prompts: str = DEFAULT_SAMPLE_PROMPTS
    sample_every: int = DEFAULT_SAMPLE_EVERY
    save_every: int = DEFAULT_SAVE_EVERY


class PipelineRunRequest(BaseModel):
    trigger: str = DEFAULT_TRIGGER
    count: int = DEFAULT_PIPELINE_COUNT
    top_n: int = DEFAULT_PIPELINE_TOP_N
    min_face_area: float = 0.01
    base_caption: str = DEFAULT_BASE_CAPTION
    model_name: str = DEFAULT_MODEL_NAME
    rank: int = DEFAULT_RANK
    steps: int = DEFAULT_STEPS
    lr: str = DEFAULT_LR
    sample_prompts: str = DEFAULT_SAMPLE_PROMPTS
    sample_every: int = DEFAULT_SAMPLE_EVERY
    save_every: int = DEFAULT_SAVE_EVERY
    start_training: bool = True
    generation_backend: str = DEFAULT_GENERATION_BACKEND
    allow_smoke_fallback: bool = False


def safe_case_name(case_name: str) -> str:
    return ensure_case(case_name)


def safe_file(path_value: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    root = WORK_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="File is outside the Project Chimera work root.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return path


def file_url(path: str) -> str:
    return f"/api/file?path={quote(str(path), safe='')}"


def image_records(folder: Path):
    return [{"name": Path(path).name, "path": path, "url": file_url(path)} for path in image_paths(folder)]


def dataframe_records(df):
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        return df.fillna("").to_dict(orient="records")
    return df


def latest_config_path(case_name: str) -> str:
    path = AI_TOOLKIT_DIR / "config" / f"lora_factory_{clean_slug(case_name)}.yml"
    return str(path) if path.exists() else ""


def case_payload(case_name: str, top_n: int = 100):
    case_name = safe_case_name(case_name)
    paths = dirs(case_name)
    qc_path = paths["case"] / "qc_scores.csv"
    qc_records = []
    if qc_path.exists():
        qc_records = pd.read_csv(qc_path).fillna("").to_dict(orient="records")
    selected = set(selected_from_qc(case_name, top_n=top_n))
    artifacts = training_artifact_steps(case_name)
    sample = latest_sample_image(case_name)
    current, target, pct, running, _ = training_progress(case_name, 2000)
    return {
        "case": case_name,
        "work_root": str(WORK_ROOT),
        "config_path": latest_config_path(case_name),
        "refs": image_records(paths["refs"]),
        "candidates": [
            {**record, "selected": record["path"] in selected}
            for record in image_records(paths["candidates"])
        ],
        "train": image_records(paths["train"]),
        "qc": qc_records,
        "logs": {
            "generate": read_log(case_name, "generate"),
            "train": read_log(case_name, "train"),
        },
        "dashboard": {
            "current_step": current,
            "total_steps": target,
            "percent": pct,
            "running": running,
            "training_state": read_training_state(case_name),
            "stats": system_stats(case_name),
            "artifacts": artifacts,
            "latest_sample": file_url(sample) if sample else None,
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Project Chimera")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"ok": True, "work_root": str(WORK_ROOT)}

    @app.get("/api/preflight")
    def preflight():
        return runtime_preflight()

    @app.get("/api/cases")
    def get_cases():
        cases = list_cases()
        return {"cases": cases, "active": cases[0]}

    @app.post("/api/cases")
    def post_case(payload: CaseCreate):
        name = create_case(payload.label)
        return {"case": name, "cases": list_cases(), "state": case_payload(name)}

    @app.get("/api/cases/{case_name}/state")
    def get_case_state(case_name: str, top_n: int = 100):
        return case_payload(case_name, top_n=top_n)

    @app.post("/api/cases/{case_name}/references")
    async def post_references(
        case_name: str,
        consent: Annotated[bool, Form()],
        files: Annotated[list[UploadFile], File()],
    ):
        if not consent:
            raise HTTPException(status_code=400, detail="Consent/rights confirmation is required.")
        paths = dirs(safe_case_name(case_name))
        saved = 0
        for item in files:
            suffix = Path(item.filename or "").suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            dest = paths["refs"] / f"ref_{saved + 1:02d}{suffix}"
            dest.write_bytes(await item.read())
            saved += 1
        return {"status": f"Saved {saved} reference image(s).", "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/smoke")
    def post_smoke(case_name: str, payload: SmokeRequest):
        status, _ = smoke_generate(case_name, payload.count)
        return {"status": status, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/import")
    def post_import(case_name: str, payload: ImportRequest):
        status, _ = import_candidates(case_name, payload.source_folder, payload.copy_limit)
        return {"status": status, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/generation/start")
    def post_generation(case_name: str, payload: GenerationRequest):
        status, log = start_background(case_name, "generate", payload.command, payload.trigger, payload.count)
        return {"status": status, "log": log, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/qc/score")
    def post_qc_score(case_name: str, payload: QcRequest):
        try:
            status, df = score_candidates(case_name, FIXED_IDENTITY_THRESHOLD, payload.min_face_area)
        except FaceQcUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": status, "qc": dataframe_records(df), "state": case_payload(case_name, payload.top_n)}

    @app.post("/api/cases/{case_name}/qc/score-select")
    def post_qc_score_select(case_name: str, payload: QcRequest):
        try:
            status, df, _, _ = score_select_pipeline(case_name, FIXED_IDENTITY_THRESHOLD, payload.min_face_area, payload.top_n)
        except FaceQcUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": status, "qc": dataframe_records(df), "state": case_payload(case_name, payload.top_n)}

    @app.post("/api/cases/{case_name}/captions")
    def post_captions(case_name: str, payload: CaptionRequest):
        status = caption_curated(case_name, payload.trigger, payload.base_caption)
        return {"status": status, "captions": dataframe_records(preview_captions(case_name)), "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/training/config")
    def post_training_config(case_name: str, payload: TrainingPrepareRequest):
        config_path, preview = build_ai_toolkit_config(
            case_name,
            payload.trigger,
            payload.model_name,
            payload.rank,
            payload.steps,
            payload.lr,
            payload.sample_prompts,
            payload.sample_every,
            payload.save_every,
        )
        return {"status": f"Wrote ai-toolkit config: {config_path}", "config_path": config_path, "preview": preview, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/training/prepare")
    def post_training_prepare(case_name: str, payload: TrainingPrepareRequest):
        status, config_path, _, _, _, log = prepare_pipeline_training(
            case_name,
            payload.trigger,
            payload.base_caption,
            payload.model_name,
            payload.rank,
            payload.steps,
            payload.lr,
            payload.sample_prompts,
            payload.sample_every,
            payload.save_every,
        )
        return {"status": status, "config_path": config_path, "log": log, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/training/start")
    def post_training_start(case_name: str, payload: TrainingStartRequest):
        try:
            status, _, _, _, log = start_pipeline_training(
                case_name,
                payload.config_path,
                payload.trigger,
                payload.steps,
                payload.sample_prompts,
                payload.sample_every,
                payload.save_every,
            )
        except TrainingUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"status": status, "log": log, "state": case_payload(case_name)}

    @app.post("/api/cases/{case_name}/pipeline/run")
    def post_pipeline_run(case_name: str, payload: PipelineRunRequest):
        statuses = []
        count = max(1, int(payload.count or DEFAULT_PIPELINE_COUNT))
        top_n = max(1, int(payload.top_n or count))

        try:
            generation_status, _ = generate_candidates(
                case_name,
                count,
                payload.trigger,
                payload.generation_backend,
                payload.allow_smoke_fallback,
            )
        except GenerationUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        statuses.append(generation_status)

        try:
            qc_status, _, _, _ = score_select_pipeline(
                case_name,
                FIXED_IDENTITY_THRESHOLD,
                payload.min_face_area,
                top_n,
            )
        except FaceQcUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        statuses.append(qc_status)

        prepare_status, config_path, _, _, _, _ = prepare_pipeline_training(
            case_name,
            payload.trigger,
            payload.base_caption,
            payload.model_name,
            payload.rank,
            payload.steps,
            payload.lr,
            payload.sample_prompts,
            payload.sample_every,
            payload.save_every,
        )
        statuses.append(prepare_status)

        train_log = ""
        if payload.start_training:
            try:
                train_status, _, _, _, train_log = start_pipeline_training(
                    case_name,
                    config_path,
                    payload.trigger,
                    payload.steps,
                    payload.sample_prompts,
                    payload.sample_every,
                    payload.save_every,
                )
                statuses.append(train_status)
            except TrainingUnavailable as exc:
                statuses.append(f"Training setup error: {exc}")

        return {
            "status": "\n".join(statuses),
            "config_path": config_path,
            "fixed_identity_threshold": FIXED_IDENTITY_THRESHOLD,
            "log": train_log,
            "state": case_payload(case_name, top_n),
        }

    @app.get("/api/cases/{case_name}/dashboard")
    def get_dashboard(case_name: str, steps: int = 2000, sample_every: int = 250, save_every: int = 250):
        case_name = safe_case_name(case_name)
        current, target, pct, running, _ = training_progress(case_name, steps)
        sample = latest_sample_image(case_name)
        artifacts = training_artifact_steps(case_name)
        return {
            "current_step": current,
            "total_steps": target,
            "percent": pct,
            "running": running,
            "training_state": read_training_state(case_name),
            "stats": system_stats(case_name),
            "artifacts": artifacts,
            "expected_sample_steps": expected_interval_steps(steps, sample_every),
            "expected_checkpoint_steps": expected_interval_steps(steps, save_every),
            "latest_sample": file_url(sample) if sample else None,
            "train_log": read_log(case_name, "train"),
        }

    @app.get("/api/cases/{case_name}/logs/{name}")
    def get_log(case_name: str, name: str):
        return {"log": read_log(case_name, name)}

    @app.get("/api/file")
    def get_file(path: str):
        safe = safe_file(path)
        return FileResponse(safe)

    web_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")

    return app
