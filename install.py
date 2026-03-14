"""
ComfyUI auto-runs this file on startup to install dependencies for the
Chimera node.
"""

import os
import subprocess
import sys


def run(cmd, cwd=None):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[Chimera] WARNING: command failed: {' '.join(cmd)}")
        print(f"[Chimera] stdout: {result.stdout.strip()}")
        print(f"[Chimera] stderr: {result.stderr.strip()}")
    return result.returncode == 0


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(NODE_DIR, "requirements.txt")
AI_TOOLKIT_DIR = os.path.join(NODE_DIR, "ai-toolkit")
AI_TOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"
AI_TOOLKIT_REQUIREMENTS = os.path.join(AI_TOOLKIT_DIR, "requirements.txt")


def install():
    # Install node requirements
    print("[Chimera] Installing requirements...")
    ok = run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS, "--quiet"])
    if not ok:
        print("[Chimera] WARNING: Some requirements may not have installed correctly.")

    # Clone ai-toolkit if not already present
    if not os.path.isdir(AI_TOOLKIT_DIR):
        print("[Chimera] Cloning ostris/ai-toolkit...")
        ok = run(["git", "clone", "--depth", "1", AI_TOOLKIT_REPO, AI_TOOLKIT_DIR])
        if not ok:
            print("[Chimera] WARNING: Failed to clone ai-toolkit. Training will not be available.")
            return
    else:
        print("[Chimera] ai-toolkit already present, skipping clone.")

    # Install ai-toolkit requirements
    if os.path.isfile(AI_TOOLKIT_REQUIREMENTS):
        print("[Chimera] Installing ai-toolkit requirements...")
        ok = run(
            [sys.executable, "-m", "pip", "install", "-r", AI_TOOLKIT_REQUIREMENTS, "--quiet"],
            cwd=AI_TOOLKIT_DIR,
        )
        if not ok:
            print("[Chimera] WARNING: Some ai-toolkit requirements may not have installed correctly.")
    else:
        print("[Chimera] WARNING: ai-toolkit/requirements.txt not found, skipping.")

    print("[Chimera] Install complete.")


try:
    install()
except Exception as exc:
    print(f"[Chimera] WARNING: install.py encountered an unexpected error: {exc}")
    print("[Chimera] ComfyUI will continue, but the node may not function correctly.")
