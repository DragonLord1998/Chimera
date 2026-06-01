import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def write_command(path, source):
    path.write_text(source)
    path.chmod(0o755)
    return f"{sys.executable} {path}"


def png_bytes(color=(120, 80, 180)):
    buffer = io.BytesIO()
    Image.new("RGB", (768, 768), color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def install_env(root):
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)

    os.environ["WORK_ROOT"] = str(root / "work")
    os.environ["AI_TOOLKIT_DIR"] = str(root / "ai_toolkit")
    os.environ["FACE_MODEL"] = "buffalo_l"
    Path(os.environ["AI_TOOLKIT_DIR"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["AI_TOOLKIT_DIR"]) / "run.py").write_text("print('ai-toolkit shim')\n")

    os.environ["FLUX2_PULID_COMMAND"] = write_command(
        scripts / "flux2_pulid_shim.py",
        """
import json, os
from pathlib import Path
from PIL import Image

manifest = json.loads(Path(os.environ["PROMPT_MANIFEST"]).read_text())
out = Path(os.environ["CANDIDATE_DIR"])
out.mkdir(parents=True, exist_ok=True)
for index, row in enumerate(manifest):
    Image.new("RGB", (768, 768), (40 + index * 20, 80, 160)).save(out / row["output_name"])
""".strip(),
    )
    os.environ["ZIMAGE_IDENTITY_TRAIN_COMMAND"] = write_command(
        scripts / "zimage_identity_shim.py",
        """
import os
from pathlib import Path

out = Path(os.environ["IDENTITY_LORA_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "zimage_identity_lora.safetensors").write_bytes(b"identity-lora")
""".strip(),
    )
    os.environ["ZIMAGE_EXPANSION_COMMAND"] = write_command(
        scripts / "zimage_expansion_shim.py",
        """
import json, os
from pathlib import Path
from PIL import Image

manifest = json.loads(Path(os.environ["ZIMAGE_PROMPT_MANIFEST"]).read_text())
out = Path(os.environ["PRODUCTION_CANDIDATE_DIR"])
out.mkdir(parents=True, exist_ok=True)
for index, row in enumerate(manifest):
    Image.new("RGB", (768, 768), (90, 100 + index * 15, 170)).save(out / row["output_name"])
""".strip(),
    )
    os.environ["MODEL_LORA_COMMAND"] = write_command(
        scripts / "model_loras_shim.py",
        """
import os
from pathlib import Path

out = Path(os.environ["MODEL_LORA_DIR"])
out.mkdir(parents=True, exist_ok=True)
for target in os.environ["MODEL_TARGETS"].split(","):
    (out / f"{target}.safetensors").write_bytes(f"{target}-lora".encode())
""".strip(),
    )


def install_fake_qc(api_module):
    from automatic_lora_trainer.paths import dirs, image_paths

    def select_images(case_name, source_key, csv_name, dest_key, top_n):
        paths = dirs(case_name)
        rows = []
        for index, src in enumerate(image_paths(paths[source_key])[: int(top_n)], start=1):
            src_path = Path(src)
            rows.append(
                {
                    "file": str(src_path),
                    "name": src_path.name,
                    "identity_score": 0.99,
                    "face_count": 1,
                    "face_area": 0.2,
                    "sharpness": 100.0,
                    "width": 768,
                    "height": 768,
                    "passed": True,
                    "reason": "pass",
                }
            )
            dest = paths[dest_key] / f"{index:04d}.png"
            with Image.open(src_path) as img:
                img.save(dest)
        df = pd.DataFrame(rows)
        df.to_csv(paths["case"] / csv_name, index=False)
        return df

    def fake_seed_qc(case_name, identity_threshold, min_face_area, top_n):
        df = select_images(case_name, "candidates", "qc_scores.csv", "train", top_n)
        return f"Fake seed QC selected {len(df)}.", df, None, None

    def fake_final_qc(case_name, identity_threshold, min_face_area, top_n):
        df = select_images(case_name, "production_candidates", "final_qc_scores.csv", "final", top_n)
        return f"Fake final QC selected {len(df)}.", df, image_paths(dirs(case_name)["final"])

    api_module.score_select_pipeline = fake_seed_qc
    api_module.score_select_final_pipeline = fake_final_qc


def main():
    with tempfile.TemporaryDirectory(prefix="chimera_full_pipeline_") as temp_dir:
        root = Path(temp_dir)
        install_env(root)

        from automatic_lora_trainer import api

        install_fake_qc(api)
        client = TestClient(api.create_app())

        case_response = client.post("/api/cases", json={"label": "contract"})
        assert case_response.status_code == 200, case_response.text
        case_name = case_response.json()["case"]

        upload_response = client.post(
            f"/api/cases/{case_name}/references",
            data={"consent": "true"},
            files={"files": ("ref.png", png_bytes(), "image/png")},
        )
        assert upload_response.status_code == 200, upload_response.text

        run_response = client.post(
            f"/api/cases/{case_name}/pipeline/full-run",
            json={
                "trigger": "zphchar",
                "count": 3,
                "top_n": 2,
                "zimage_count": 4,
                "final_top_n": 3,
                "min_face_area": 0.01,
                "base_caption": "",
                "model_name": "z-image-turbo",
                "rank": 32,
                "steps": 1200,
                "lr": "1e-4",
                "sample_prompts": "zphchar person, portrait photo",
                "sample_every": 200,
                "save_every": 250,
                "generation_backend": "flux2_pulid",
                "allow_smoke_fallback": False,
            },
        )
        assert run_response.status_code == 200, run_response.text
        payload = run_response.json()
        state = payload["state"]

        assert len(state["refs"]) == 1
        assert len(state["candidates"]) == 3
        assert len(state["train"]) == 2
        assert len(state["identity_lora_artifacts"]) == 1
        assert len(state["production_candidates"]) == 4
        assert len(state["final"]) == 3
        assert len(state["model_lora_artifacts"]) == 4

        final_txts = sorted((root / "work" / "cases" / case_name / "curated" / "final").glob("*.txt"))
        assert len(final_txts) == 3
        assert all(path.read_text().startswith("zphchar person") for path in final_txts)

        print("Full pipeline contract passed")


if __name__ == "__main__":
    main()
