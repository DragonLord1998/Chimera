import pandas as pd
import gradio as gr

from .captioning import caption_curated, preview_captions
from .dashboard import dashboard_html, dashboard_sample, prompt_card_html, refresh_dashboard
from .face_qc import auto_select, score_candidates, score_select_pipeline
from .generation import import_candidates, import_candidates_pipeline, save_refs, save_refs_pipeline, smoke_generate, smoke_generate_pipeline, start_background
from .media import candidates_sheet, qc_sheet, refs_sheet, train_sheet
from .paths import create_case as make_case, dirs, ensure_case, image_paths, list_cases
from .settings import AI_TOOLKIT_DIR, WORK_ROOT
from .state import read_log
from .training import build_ai_toolkit_config, prepare_pipeline_training, start_pipeline_training, start_training


DEFAULT_GENERATION_COMMAND = """# Replace this with your real generator command.
# The server sets REF_DIR, CANDIDATE_DIR, CASE_DIR, COUNT, and TRIGGER.
# Example: copy existing ComfyUI outputs into this case:
python - <<'PYGEN'
import glob, os, shutil
src = "/content/drive/MyDrive/GenAI/ComfyUI/output"
out = os.environ["CANDIDATE_DIR"]
os.makedirs(out, exist_ok=True)
for i, path in enumerate(glob.glob(src + "/*.png")[: int(os.environ.get("COUNT", "200"))]):
    shutil.copy2(path, os.path.join(out, f"comfy_{i+1:04d}.png"))
print("copied ComfyUI outputs")
PYGEN
"""


DEFAULT_SAMPLE_PROMPTS = """[trigger] person, studio portrait, plain white background
[trigger] person, side profile photo, natural lighting
[trigger] person, full body photo, wearing a black jacket
[trigger] person, cinematic close-up, night street
[trigger] person, casual selfie, indoor lighting
[trigger] person, half body photo, outdoor background
[trigger] person, fashion editorial portrait
[trigger] person, close-up portrait, soft daylight"""


def create_case(label):
    name = make_case(label)
    return gr.update(choices=list_cases(), value=name), f"Created case: {name}"


def refresh_all(case_name):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    qc = paths["case"] / "qc_scores.csv"
    df = pd.read_csv(qc) if qc.exists() else None
    return image_paths(paths["refs"]), image_paths(paths["candidates"]), image_paths(paths["train"]), df


def refresh_pipeline(case_name, top_n):
    case_name = ensure_case(case_name)
    paths = dirs(case_name)
    qc = paths["case"] / "qc_scores.csv"
    df = pd.read_csv(qc) if qc.exists() else None
    return refs_sheet(case_name), candidates_sheet(case_name), qc_sheet(case_name, top_n=top_n), train_sheet(case_name), df


def build_app():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    initial_case = list_cases()[0]
    dirs(initial_case)

    with gr.Blocks(title="Project Chimera") as demo:
        gr.Markdown("# Project Chimera\nUpload references, generate/import candidates, run identity QC, caption, then train a LoRA with ai-toolkit.")
        with gr.Row():
            case = gr.Dropdown(label="Case", choices=list_cases(), value=initial_case, scale=2)
            label = gr.Textbox(label="New case label", value="character", scale=1)
            create_btn = gr.Button("Create Case", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 1. Upload")
                    pipe_consent = gr.Checkbox(label="I have consent/rights to use these images.", value=False)
                    pipe_upload = gr.File(label="Reference images", file_count="multiple", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                    pipe_save_refs_btn = gr.Button("Save References", variant="primary")
                    pipe_refs_sheet = gr.Image(label="Reference Input", value=refs_sheet(initial_case), type="pil", height=360)
                with gr.Column(scale=1):
                    gr.Markdown("## 2. Generate")
                    pipe_count = gr.Number(label="Candidate count / import limit", value=200, precision=0)
                    pipe_smoke_btn = gr.Button("Smoke Test Generate")
                    pipe_import_folder = gr.Textbox(label="Import folder", value="/content/drive/MyDrive/GenAI/ComfyUI/output")
                    pipe_import_btn = gr.Button("Import Images")
                    pipe_candidates_sheet = gr.Image(label="Synthetic Candidates", value=candidates_sheet(initial_case), type="pil", height=360)
                with gr.Column(scale=1):
                    gr.Markdown("## 3. QC Select")
                    pipe_identity_threshold = gr.Slider(label="Identity threshold", minimum=0.1, maximum=0.8, step=0.01, value=0.32)
                    pipe_min_face_area = gr.Slider(label="Minimum face area", minimum=0.001, maximum=0.2, step=0.001, value=0.01)
                    pipe_top_n = gr.Number(label="Keep top N", value=100, precision=0)
                    pipe_qc_btn = gr.Button("Score + Select", variant="primary")
                    pipe_qc_sheet = gr.Image(label="QC Highlights", value=qc_sheet(initial_case), type="pil", height=360)
            with gr.Row():
                pipe_refresh_btn = gr.Button("Refresh Pipeline")
                pipe_train_sheet = gr.Image(label="Curated Training Set", value=train_sheet(initial_case), type="pil", height=260)
            pipe_qc_table = gr.Dataframe(label="QC scores")

            gr.Markdown("## 4. Training Dashboard")
            with gr.Row():
                with gr.Column(scale=2):
                    pipe_dashboard = gr.HTML(value=dashboard_html(initial_case, 2000, DEFAULT_SAMPLE_PROMPTS, 250, 250))
                with gr.Column(scale=1):
                    pipe_dashboard_sample = gr.Image(label="Latest Sample", value=dashboard_sample(initial_case), type="pil", height=280)
            pipe_prompt_cards = gr.HTML(value=prompt_card_html(DEFAULT_SAMPLE_PROMPTS, initial_case, 2000, 250))
            with gr.Row():
                pipe_trigger = gr.Textbox(label="Trigger token", value="zphchar")
                pipe_model_name = gr.Textbox(label="Base model", value="black-forest-labs/FLUX.2-klein-base-9B", scale=2)
            pipe_base_caption = gr.Textbox(label="Base caption suffix", value="natural skin texture, realistic photo, high detail")
            with gr.Row():
                pipe_rank = gr.Number(label="LoRA rank", value=64, precision=0)
                pipe_steps = gr.Number(label="Steps", value=2000, precision=0)
                pipe_lr = gr.Textbox(label="Learning rate", value="1e-4")
                pipe_sample_every = gr.Number(label="Sample prompts every N steps", value=250, precision=0)
                pipe_save_every = gr.Number(label="Save checkpoint every N steps", value=250, precision=0)
            pipe_sample_prompts = gr.Textbox(label="Sample prompts", value=DEFAULT_SAMPLE_PROMPTS, lines=6)
            with gr.Row():
                pipe_prepare_train_btn = gr.Button("Caption + Write Config", variant="primary")
                pipe_start_train_btn = gr.Button("Start Training")
                pipe_refresh_dashboard_btn = gr.Button("Refresh Dashboard")
            pipe_config_path = gr.Textbox(label="ai-toolkit config path")
            pipe_train_log = gr.Textbox(label="Training log", lines=10)

        with gr.Tab("1. References"):
            consent = gr.Checkbox(label="I have consent/rights to use these images for this LoRA dataset.", value=False)
            refs_upload = gr.File(label="Reference images", file_count="multiple", file_types=[".png", ".jpg", ".jpeg", ".webp"])
            save_refs_btn = gr.Button("Save References", variant="primary")
            refs_gallery = gr.Gallery(label="References", columns=4, height=260)

        with gr.Tab("2. Generate / Import"):
            gr.Markdown("Use Smoke Test only to verify the UI. For production, run your FLUX/ComfyUI generation command or import an output folder.")
            count = gr.Number(label="Candidate count / import limit", value=200, precision=0)
            with gr.Row():
                smoke_btn = gr.Button("Smoke Test Generate")
                import_folder = gr.Textbox(label="Import folder", value="/content/drive/MyDrive/GenAI/ComfyUI/output")
                import_btn = gr.Button("Import Images")
            generation_command = gr.Textbox(label="Generation command", value=DEFAULT_GENERATION_COMMAND, lines=12)
            start_generation_btn = gr.Button("Start Generation Command", variant="primary")
            generation_log = gr.Textbox(label="Generation log", lines=12)
            candidates_gallery = gr.Gallery(label="Candidates", columns=6, height=420)

        with gr.Tab("3. Quality Control"):
            with gr.Row():
                identity_threshold = gr.Slider(label="Identity threshold", minimum=0.1, maximum=0.8, step=0.01, value=0.32)
                min_face_area = gr.Slider(label="Minimum face area", minimum=0.001, maximum=0.2, step=0.001, value=0.01)
                top_n = gr.Number(label="Keep top N", value=100, precision=0)
            score_btn = gr.Button("Score Candidates", variant="primary")
            qc_table = gr.Dataframe(label="QC scores")
            auto_select_btn = gr.Button("Auto-select Training Set")
            train_gallery = gr.Gallery(label="Curated Training Images", columns=6, height=420)

        with gr.Tab("4. Captions"):
            trigger = gr.Textbox(label="Trigger token", value="zphchar")
            base_caption = gr.Textbox(label="Base caption suffix", value="natural skin texture, realistic photo, high detail")
            caption_btn = gr.Button("Caption Curated Images", variant="primary")
            captions_table = gr.Dataframe(label="Caption preview")

        with gr.Tab("5. Train LoRA"):
            model_name = gr.Textbox(label="Base model", value="black-forest-labs/FLUX.2-klein-base-9B")
            with gr.Row():
                rank = gr.Number(label="LoRA rank", value=64, precision=0)
                steps = gr.Number(label="Steps", value=2000, precision=0)
                lr = gr.Textbox(label="Learning rate", value="1e-4")
                sample_every = gr.Number(label="Sample every N steps", value=250, precision=0)
                save_every = gr.Number(label="Checkpoint every N steps", value=250, precision=0)
            sample_prompts = gr.Textbox(label="Sample prompts", value=DEFAULT_SAMPLE_PROMPTS, lines=6)
            config_btn = gr.Button("Write ai-toolkit Config", variant="primary")
            config_path = gr.Textbox(label="Config path")
            config_preview = gr.Code(label="Config preview", language="yaml", lines=18)
            train_btn = gr.Button("Start Training")
            train_log = gr.Textbox(label="Training log", lines=14)

        with gr.Tab("Refresh"):
            refresh_btn = gr.Button("Refresh Galleries / Logs")

        create_event = create_btn.click(create_case, inputs=[label], outputs=[case, status])
        create_event.then(refresh_pipeline, inputs=[case, pipe_top_n], outputs=[pipe_refs_sheet, pipe_candidates_sheet, pipe_qc_sheet, pipe_train_sheet, pipe_qc_table])
        create_event.then(refresh_dashboard, inputs=[case, pipe_steps, pipe_sample_prompts, pipe_sample_every, pipe_save_every], outputs=[pipe_dashboard, pipe_dashboard_sample, pipe_prompt_cards, pipe_train_log])
        case.change(refresh_pipeline, inputs=[case, pipe_top_n], outputs=[pipe_refs_sheet, pipe_candidates_sheet, pipe_qc_sheet, pipe_train_sheet, pipe_qc_table])
        case.change(refresh_dashboard, inputs=[case, pipe_steps, pipe_sample_prompts, pipe_sample_every, pipe_save_every], outputs=[pipe_dashboard, pipe_dashboard_sample, pipe_prompt_cards, pipe_train_log])
        pipe_save_refs_btn.click(save_refs_pipeline, inputs=[case, pipe_upload, pipe_consent], outputs=[status, pipe_refs_sheet])
        pipe_smoke_btn.click(smoke_generate_pipeline, inputs=[case, pipe_count], outputs=[status, pipe_candidates_sheet, pipe_qc_sheet])
        pipe_import_btn.click(import_candidates_pipeline, inputs=[case, pipe_import_folder, pipe_count], outputs=[status, pipe_candidates_sheet, pipe_qc_sheet])
        pipe_qc_btn.click(score_select_pipeline, inputs=[case, pipe_identity_threshold, pipe_min_face_area, pipe_top_n], outputs=[status, pipe_qc_table, pipe_qc_sheet, pipe_train_sheet])
        pipe_refresh_btn.click(refresh_pipeline, inputs=[case, pipe_top_n], outputs=[pipe_refs_sheet, pipe_candidates_sheet, pipe_qc_sheet, pipe_train_sheet, pipe_qc_table])
        pipe_prepare_train_btn.click(prepare_pipeline_training, inputs=[case, pipe_trigger, pipe_base_caption, pipe_model_name, pipe_rank, pipe_steps, pipe_lr, pipe_sample_prompts, pipe_sample_every, pipe_save_every], outputs=[status, pipe_config_path, pipe_dashboard, pipe_dashboard_sample, pipe_prompt_cards, pipe_train_log])
        pipe_start_train_btn.click(start_pipeline_training, inputs=[case, pipe_config_path, pipe_trigger, pipe_steps, pipe_sample_prompts, pipe_sample_every, pipe_save_every], outputs=[status, pipe_dashboard, pipe_dashboard_sample, pipe_prompt_cards, pipe_train_log])
        pipe_refresh_dashboard_btn.click(refresh_dashboard, inputs=[case, pipe_steps, pipe_sample_prompts, pipe_sample_every, pipe_save_every], outputs=[pipe_dashboard, pipe_dashboard_sample, pipe_prompt_cards, pipe_train_log])
        save_refs_btn.click(save_refs, inputs=[case, refs_upload, consent], outputs=[status, refs_gallery])
        smoke_btn.click(smoke_generate, inputs=[case, count], outputs=[status, candidates_gallery])
        import_btn.click(import_candidates, inputs=[case, import_folder, count], outputs=[status, candidates_gallery])
        start_generation_btn.click(start_background, inputs=[case, gr.State("generate"), generation_command, trigger, count], outputs=[status, generation_log])
        score_btn.click(score_candidates, inputs=[case, identity_threshold, min_face_area], outputs=[status, qc_table])
        auto_select_btn.click(auto_select, inputs=[case, top_n], outputs=[status, train_gallery])
        caption_btn.click(caption_curated, inputs=[case, trigger, base_caption], outputs=[status]).then(preview_captions, inputs=[case], outputs=[captions_table])
        config_btn.click(build_ai_toolkit_config, inputs=[case, trigger, model_name, rank, steps, lr, sample_prompts, sample_every, save_every], outputs=[config_path, config_preview])
        train_btn.click(start_training, inputs=[case, config_path, trigger, steps], outputs=[status, train_log])
        refresh_btn.click(refresh_all, inputs=[case], outputs=[refs_gallery, candidates_gallery, train_gallery, qc_table])
        refresh_btn.click(read_log, inputs=[case, gr.State("generate")], outputs=[generation_log])
        refresh_btn.click(read_log, inputs=[case, gr.State("train")], outputs=[train_log])
    return demo
