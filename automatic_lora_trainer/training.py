import datetime as dt
import re
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from .captioning import caption_curated
from .dashboard import refresh_dashboard
from .paths import clean_slug, dirs, ensure_case
from .settings import AI_TOOLKIT_DIR
from .state import RUNNING, env_for_case, log_path, read_log, write_training_state


class TrainingUnavailable(RuntimeError):
    pass


def build_ai_toolkit_config(case_name, trigger, model_name, rank, steps, lr, sample_prompts, sample_every=250, save_every=250):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    config_dir = AI_TOOLKIT_DIR / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"lora_factory_{clean_slug(case_name)}.yml"
    prompts = [p.strip() for p in sample_prompts.splitlines() if p.strip()] or [
        "[trigger] person, studio portrait, plain white background",
        "[trigger] person, side profile photo, natural lighting",
        "[trigger] person, full body photo, wearing a black jacket",
        "[trigger] person, cinematic close-up, night street",
    ]
    cfg = {
        "job": "extension",
        "config": {
            "name": f"{clean_slug(case_name)}_{trigger}",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(paths["output"]),
                    "device": "cuda:0",
                    "trigger_word": trigger,
                    "network": {"type": "lora", "linear": int(rank), "linear_alpha": int(rank)},
                    "save": {"dtype": "float16", "save_every": int(save_every), "max_step_saves_to_keep": 4, "push_to_hub": False},
                    "datasets": [
                        {
                            "folder_path": str(paths["train"]),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": int(steps),
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": float(lr),
                        "ema_config": {"use_ema": True, "ema_decay": 0.99},
                        "dtype": "bf16",
                    },
                    "model": {"name_or_path": model_name, "is_flux": True, "quantize": True},
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": int(sample_every),
                        "width": 1024,
                        "height": 1024,
                        "prompts": prompts,
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                    "meta": {"name": "[name]", "version": "1.0"},
                }
            ],
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(cfg_path), cfg_path.read_text()


def config_total_steps(config_path, fallback=2000):
    try:
        cfg = yaml.safe_load(Path(config_path).read_text())
        return int(cfg["config"]["process"][0]["train"]["steps"])
    except Exception:
        return int(float(fallback or 2000))


def parse_reported_step(line):
    text = line.replace("\r", "\n").splitlines()[-1].strip()
    if not text or not any(marker in text.lower() for marker in ("step", "steps", "it/s", "s/it", "loss", "%|")):
        return None
    for pattern in (
        r"(?:step|steps)[^\d]{0,20}(\d+)\s*/\s*(\d+)",
        r"(\d+)\s*/\s*(\d+).{0,50}(?:step|steps|it/s|s/it|loss)",
        r"(?:global_step|current_step|step)[^\d]{0,20}(\d+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            current = int(match.group(1))
            total = int(match.group(2)) if len(match.groups()) >= 2 and match.group(2) else None
            return current, total, text[-500:]
    return None


def monitor_training_process(case_name, proc, log_file, total_steps):
    write_training_state(case_name, status="running", pid=proc.pid, current_step=None, total_steps=int(float(total_steps or 1)), exit_code=None, last_step_line="")
    with log_file.open("ab", buffering=0) as handle:
        buffer = ""
        while True:
            chunk = proc.stdout.read(1)
            if chunk == "":
                break
            handle.write(chunk.encode(errors="replace"))
            buffer += chunk
            if chunk in ("\r", "\n"):
                parsed = parse_reported_step(buffer)
                if parsed:
                    current, parsed_total, source = parsed
                    write_training_state(case_name, status="running", pid=proc.pid, current_step=current, total_steps=parsed_total or int(float(total_steps or 1)), last_step_line=source)
                buffer = ""
        if buffer:
            parsed = parse_reported_step(buffer)
            if parsed:
                current, parsed_total, source = parsed
                write_training_state(case_name, status="running", pid=proc.pid, current_step=current, total_steps=parsed_total or int(float(total_steps or 1)), last_step_line=source)
        exit_code = proc.wait()
        handle.write(f"\n[{dt.datetime.now().isoformat()}] Training exited with code {exit_code}\n".encode())
    write_training_state(case_name, status="complete" if exit_code == 0 else "failed", pid=proc.pid, exit_code=exit_code)


def start_tracked_training(case_name, config_path, total_steps):
    case_name = ensure_case(case_name)
    run_py = AI_TOOLKIT_DIR / "run.py"
    if not run_py.exists():
        message = f"ai-toolkit run.py was not found at {run_py}. Install ai-toolkit or launch Chimera with INSTALL_AI_TOOLKIT=1."
        lp = log_path(case_name, "train")
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("ab", buffering=0) as handle:
            handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Training setup error\n{message}\n".encode())
        write_training_state(case_name, status="setup_error", pid=None, current_step=None, total_steps=int(float(total_steps or 1)), exit_code=None, last_step_line=message)
        raise TrainingUnavailable(message)
    key = (case_name, "train")
    existing = RUNNING.get(key)
    if existing and existing.poll() is None:
        return f"Training is already running with PID {existing.pid}.", read_log(case_name, "train")
    lp = log_path(case_name, "train")
    lp.parent.mkdir(parents=True, exist_ok=True)
    env = env_for_case(case_name)
    env["PYTHONUNBUFFERED"] = "1"
    with lp.open("ab", buffering=0) as handle:
        handle.write(f"\n\n[{dt.datetime.now().isoformat()}] Starting tracked ai-toolkit training\n".encode())
        handle.write(f"cd {AI_TOOLKIT_DIR} && {sys.executable} -u run.py {config_path}\n\n".encode())
    proc = subprocess.Popen([sys.executable, "-u", "run.py", config_path], cwd=str(AI_TOOLKIT_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    RUNNING[key] = proc
    write_training_state(case_name, status="running", pid=proc.pid, current_step=None, total_steps=int(float(total_steps or 1)), exit_code=None, last_step_line="")
    threading.Thread(target=monitor_training_process, args=(case_name, proc, lp, total_steps), daemon=True).start()
    return f"Started tracked ai-toolkit training with PID {proc.pid}.", read_log(case_name, "train")


def start_training(case_name, config_path, trigger, total_steps=None):
    case_name = ensure_case(case_name)
    config_path = config_path.strip()
    if not config_path:
        return "Generate an ai-toolkit config first.", read_log(case_name, "train")
    return start_tracked_training(case_name, config_path, int(float(total_steps or config_total_steps(config_path))))


def prepare_pipeline_training(case_name, trigger, base_caption, model_name, rank, steps, lr, sample_prompts, sample_every, save_every):
    case_name = ensure_case(case_name)
    caption_status = caption_curated(case_name, trigger, base_caption)
    cfg_path, _ = build_ai_toolkit_config(case_name, trigger, model_name, rank, steps, lr, sample_prompts, sample_every, save_every)
    dash, sample, cards, log = refresh_dashboard(case_name, steps, sample_prompts, sample_every, save_every)
    return f"{caption_status}\nWrote ai-toolkit config: {cfg_path}", cfg_path, dash, sample, cards, log


def start_pipeline_training(case_name, config_path, trigger, steps, sample_prompts, sample_every, save_every):
    status, log = start_training(case_name, config_path, trigger, steps)
    dash, sample, cards, _ = refresh_dashboard(case_name, steps, sample_prompts, sample_every, save_every)
    return status, dash, sample, cards, log
