import os
from pathlib import Path


WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/content/drive/MyDrive/GenAI/Project Chimera"))
AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", "/content/ai-toolkit"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "7860"))
FACE_MODEL = os.environ.get("FACE_MODEL", "auraface")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
