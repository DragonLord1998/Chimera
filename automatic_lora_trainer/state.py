import datetime as dt
import json
import os

from .paths import dirs
from .settings import WORK_ROOT


RUNNING = {}


def log_path(case_name, name):
    return dirs(case_name)["logs"] / f"{name}.log"


def read_log(case_name, name, tail_chars=12000):
    path = log_path(case_name, name)
    if not path.exists():
        return ""
    return path.read_text(errors="replace")[-tail_chars:]


def training_state_path(case_name):
    return dirs(case_name)["case"] / "training_state.json"


def default_training_state(status="idle"):
    return {
        "status": status,
        "pid": None,
        "current_step": None,
        "total_steps": None,
        "last_step_line": "",
        "updated_at": None,
        "exit_code": None,
    }


def read_training_state(case_name):
    path = training_state_path(case_name)
    if not path.exists():
        return default_training_state()
    try:
        return json.loads(path.read_text())
    except Exception:
        return default_training_state("state_error")


def write_training_state(case_name, **updates):
    state = read_training_state(case_name)
    state.update(updates)
    state["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    training_state_path(case_name).write_text(json.dumps(state, indent=2, sort_keys=True))
    return state


def env_for_case(case_name, trigger="zphchar", count=200):
    paths = dirs(case_name)
    env = os.environ.copy()
    env.update(
        {
            "WORK_ROOT": str(WORK_ROOT),
            "CASE_NAME": case_name,
            "CASE_DIR": str(paths["case"]),
            "REF_DIR": str(paths["refs"]),
            "CANDIDATE_DIR": str(paths["candidates"]),
            "IDENTITY_LORA_DIR": str(paths["identity_lora"]),
            "MODEL_LORA_DIR": str(paths["model_loras"]),
            "PRODUCTION_CANDIDATE_DIR": str(paths["production_candidates"]),
            "CURATED_DIR": str(paths["curated"]),
            "TRAIN_DIR": str(paths["train"]),
            "FINAL_TRAIN_DIR": str(paths["final"]),
            "REJECT_DIR": str(paths["rejected"]),
            "OUTPUT_DIR": str(paths["output"]),
            "TRIGGER": trigger,
            "COUNT": str(count),
        }
    )
    return env
