import os
from pathlib import Path


WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/content/drive/MyDrive/GenAI/Project Chimera"))
AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", "/content/ai-toolkit"))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "7860"))
FACE_MODEL = os.environ.get("FACE_MODEL", "auraface")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
