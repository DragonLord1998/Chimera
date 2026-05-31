import os
from pathlib import Path


WORK_ROOT = Path(os.environ.get("WORK_ROOT", "/content/drive/MyDrive/GenAI/Project Chimera"))
AI_TOOLKIT_DIR = Path(os.environ.get("AI_TOOLKIT_DIR", "/content/ai-toolkit"))
FLUX2_PULID_COMMAND = os.environ.get("FLUX2_PULID_COMMAND", "")
ZIMAGE_IDENTITY_TRAIN_COMMAND = os.environ.get("ZIMAGE_IDENTITY_TRAIN_COMMAND", "")
ZIMAGE_EXPANSION_COMMAND = os.environ.get("ZIMAGE_EXPANSION_COMMAND", "")
MODEL_LORA_COMMAND = os.environ.get("MODEL_LORA_COMMAND", "")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "7860"))
FACE_MODEL = os.environ.get("FACE_MODEL", "auraface")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
