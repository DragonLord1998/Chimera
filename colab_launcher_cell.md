# Colab Repo Launcher Cell

Paste this into one Colab notebook cell after you push this project to GitHub.

Replace `REPO_URL` with your repository clone URL, for example:

```text
https://github.com/DragonLord1998/Chimera.git
```

```python
from google.colab import drive
drive.mount("/content/drive")

import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/DragonLord1998/Chimera.git"
BRANCH = "project-chimera-react"
REPO_DIR = Path("/content/project-chimera")
FRESH_CLONE_EVERY_RUN = True
PORT = 7860

if FRESH_CLONE_EVERY_RUN and REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

if not REPO_DIR.exists():
    clone_cmd = ["git", "clone", "--depth", "1"]
    if BRANCH:
        clone_cmd += ["--branch", BRANCH]
    clone_cmd += [REPO_URL, str(REPO_DIR)]
    subprocess.run(clone_cmd, check=True)
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "--all", "--prune"], check=True)
    if BRANCH:
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)

script_path = REPO_DIR / "colab_lora_factory.sh"
if not script_path.exists():
    found = sorted(str(p.relative_to(REPO_DIR)) for p in REPO_DIR.glob("*"))
    raise FileNotFoundError(f"Missing colab_lora_factory.sh. Files in repo root: {found}")

os.chmod(script_path, 0o755)

env = os.environ.copy()
env["PORT"] = str(PORT)
env["HOST"] = "127.0.0.1"

print(f"Launching: {script_path}")
print("Watch the output below for the Colab proxy URL.")
subprocess.run(["bash", str(script_path)], check=True, env=env)
```

The app is served through FastAPI on localhost. The only access point is the Colab proxy URL printed by the startup output.
