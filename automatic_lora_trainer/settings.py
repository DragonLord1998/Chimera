import os
from pathlib import Path


WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/content/drive/MyDrive/GenAI/Project Chimera"))
APP_DIR = Path(os.environ.get("APP_DIR", "/content/lora_factory_server"))
AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", "/content/ai-toolkit"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7860"))
SHARE = os.environ.get("SHARE", "0") == "1"
FACE_MODEL = os.environ.get("FACE_MODEL", "auraface")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

