"""
LoRATrainer — wraps AI Toolkit by Ostris to train a Z-Image LoRA.

AI Toolkit is consumed as a git-cloned directory (not a pip package), so it is
imported dynamically by prepending ``toolkit_path`` to ``sys.path`` at runtime.

Typical usage on RunPod RTX PRO 6000 (96 GB VRAM):

    trainer = LoRATrainer(
        model_path="/workspace/models/z_image",
        toolkit_path="/workspace/ai-toolkit",
    )
    lora_path = trainer.train(
        dataset_dir="/workspace/jobs/chr7x/dataset",
        output_dir="/workspace/jobs/chr7x/output",
        output_name="chr7x_lora",
        trigger_word="chr7x",
    )
    trainer.cleanup()
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Callable, Optional


class LoRATrainerError(Exception):
    """Raised when LoRA training fails or AI Toolkit cannot be imported."""


class LoRATrainer:
    """
    Train a Z-Image LoRA using AI Toolkit by Ostris.

    AI Toolkit is not a pip-installable package; it is git-cloned and its root
    directory is inserted into ``sys.path`` at runtime so that
    ``toolkit.job.run_job`` can be imported.

    Attributes:
        model_path: Path to the Z-Image De-Turbo model directory (must contain
            both ``transformer/`` and ``text_encoder/`` subdirectories).
        toolkit_path: Path to the cloned ``ai-toolkit`` repository root.
        device: PyTorch device string (default ``"cuda"``).

    Notes:
        - Z-Image uses the ``flowmatch`` noise scheduler (same as Flux).
        - On an RTX PRO 6000 (96 GB VRAM) the model sits at ~12 GB + ~4 GB optimizer
          states, so quantisation is unnecessary.
        - ``cache_latents_to_disk`` pre-encodes images once, skipping the VAE
          on every subsequent training step and saving significant VRAM.
        - Final checkpoints follow AI Toolkit's naming convention:
          ``{output_dir}/{output_name}/{output_name}_step{N}.safetensors``
    """

    def __init__(
        self,
        model_path: str,
        toolkit_path: str,
        device: str = "cuda",
        base_model: str = "zimage",
    ) -> None:
        """
        Initialise the trainer.

        Args:
            model_path: Path to the model directory.
            toolkit_path: Path to the cloned ``ai-toolkit`` repository root.
            device: PyTorch device string.  Defaults to ``"cuda"``.
            base_model: Which base model to use — ``"zimage"`` or ``"flux_krea"``.

        Raises:
            ValueError: If any required path argument is empty.
        """
        if not model_path:
            raise ValueError("model_path must not be empty.")
        if not toolkit_path:
            raise ValueError("toolkit_path must not be empty.")

        self.model_path = model_path
        self.toolkit_path = toolkit_path
        self.device = device
        self.base_model = base_model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_dir: str,
        output_dir: str,
        output_name: str,
        trigger_word: str,
        rank: int = 16,
        learning_rate: float = 1e-4,
        steps: int = 1000,
        resolution: int = 1024,
        batch_size: int = 1,
        save_every: int = 250,
        sample_every: int = 250,
        sample_prompts: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Train a Z-Image LoRA and return the path to the final checkpoint.

        Steps:
        1. Validate inputs and ensure output directory exists.
        2. Insert ``toolkit_path`` into ``sys.path`` (idempotent).
        3. Set required environment variables.
        4. Build an AI Toolkit config dict.
        5. Import and call ``toolkit.job.run_job(config)``.
        6. Locate and return the last ``.safetensors`` file written.

        Args:
            dataset_dir: Directory containing training images and ``.txt``
                captions, one per image.
            output_dir: Root directory under which AI Toolkit writes
                ``{output_name}/`` with checkpoints and samples.
            output_name: Name for this LoRA run.  Used as the subdirectory
                name and as the stem of every checkpoint filename.
            trigger_word: Short unique token that identifies the character
                (e.g. ``"chr7x"``).  Injected into every sample prompt.
            rank: LoRA rank.  Higher values capture more detail at the cost
                of file size and training time.  Defaults to 16.
            learning_rate: AdamW learning rate.  Defaults to ``1e-4``.
            steps: Total training steps.  Defaults to 1000.
            resolution: Square resolution for training images and samples.
                Defaults to 1024.
            batch_size: Images per gradient step.  Defaults to 1.
            save_every: Save a checkpoint every N steps.  Defaults to 250.
            sample_every: Generate sample images every N steps.  Defaults
                to 250.
            sample_prompts: Optional list of prompts used for sample
                generation during training.  If ``None``, two default
                portrait prompts referencing ``trigger_word`` are used.
            progress_callback: Optional ``(current_step, total_steps)``
                callable invoked after each checkpoint save.  AI Toolkit
                does not expose a step-level hook natively, so this is
                reserved for future integration.

        Returns:
            Absolute path to the final ``.safetensors`` checkpoint file.

        Raises:
            LoRATrainerError: If AI Toolkit cannot be imported (toolkit not
                cloned) or if no checkpoint is produced after training.
            ValueError: If required string arguments are empty.
        """
        if not dataset_dir:
            raise ValueError("dataset_dir must not be empty.")
        if not output_dir:
            raise ValueError("output_dir must not be empty.")
        if not output_name:
            raise ValueError("output_name must not be empty.")
        if not trigger_word:
            raise ValueError("trigger_word must not be empty.")

        os.makedirs(output_dir, exist_ok=True)

        self._ensure_toolkit_on_path()
        self._set_env()

        config = self._build_config(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_name=output_name,
            trigger_word=trigger_word,
            rank=rank,
            learning_rate=learning_rate,
            steps=steps,
            resolution=resolution,
            batch_size=batch_size,
            save_every=save_every,
            sample_every=sample_every,
            sample_prompts=sample_prompts,
        )

        print(
            f"[Chimera] LoRATrainer: starting training — "
            f"name={output_name!r}, steps={steps}, rank={rank}, "
            f"lr={learning_rate}, resolution={resolution}"
        )

        # When sample_every >= steps, disable sampling entirely by
        # monkey-patching BaseSDTrainProcess.sample to a no-op.  AI Toolkit
        # unconditionally calls self.sample() for baseline images even when
        # no sample config is present, which crashes on Z-Image's Qwen3
        # tokenizer path.
        _patched_sample = False
        _original_sample = None
        if sample_every >= steps:
            try:
                from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
                _original_sample = BaseSDTrainProcess.sample
                BaseSDTrainProcess.sample = lambda self, *a, **kw: None
                _patched_sample = True
                print("[Chimera] LoRATrainer: sampling disabled for this run")
            except ImportError:
                pass

        try:
            self._run_toolkit(config)
        finally:
            if _patched_sample and _original_sample is not None:
                from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
                BaseSDTrainProcess.sample = _original_sample

        final_checkpoint = self._find_final_checkpoint(output_dir, output_name)
        print(
            f"[Chimera] LoRATrainer: training complete → {final_checkpoint}"
        )
        return final_checkpoint

    def cleanup(self) -> None:
        """
        Release GPU memory after training.

        Empties the CUDA cache and triggers a Python garbage-collection
        cycle.  Safe to call even if training did not use CUDA.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[Chimera] LoRATrainer: CUDA cache cleared.")
        except Exception as exc:
            print(
                f"[Chimera] LoRATrainer: WARNING — could not clear "
                f"CUDA cache: {exc}"
            )
        gc.collect()

    # ------------------------------------------------------------------
    # Private helpers — config construction
    # ------------------------------------------------------------------

    def _build_config(
        self,
        dataset_dir: str,
        output_dir: str,
        output_name: str,
        trigger_word: str,
        rank: int,
        learning_rate: float,
        steps: int,
        resolution: int,
        batch_size: int,
        save_every: int,
        sample_every: int,
        sample_prompts: Optional[list[str]],
    ) -> dict:
        """
        Build an AI Toolkit job config dict for Z-Image LoRA training.

        The schema mirrors the YAML format accepted by AI Toolkit's
        ``run_job()`` entry point, but is passed in-memory as a Python dict
        so no temporary files need to be written.

        Args:
            dataset_dir: Path to the captioned training image directory.
            output_dir: Root output directory (AI Toolkit will create
                ``{output_dir}/{output_name}/`` automatically).
            output_name: Run name used for the output subdirectory and
                checkpoint filenames.
            trigger_word: Unique token identifying the character.
            rank: LoRA rank (``linear`` and ``linear_alpha``).
            learning_rate: Optimizer learning rate.
            steps: Total training steps.
            resolution: Square pixel resolution.
            batch_size: Images per gradient step.
            save_every: Checkpoint interval in steps.
            sample_every: Sample-generation interval in steps.
            sample_prompts: Prompts for in-training samples.  If ``None``,
                two default prompts are generated from ``trigger_word``.

        Returns:
            Dict conforming to the AI Toolkit job schema.
        """
        if sample_prompts is None:
            sample_prompts = [
                f"a portrait of {trigger_word}, looking at the camera, studio lighting",
                f"{trigger_word} walking in a park, natural lighting, full body shot",
            ]

        return {
            "job": "extension",
            "config": {
                "name": output_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": self.device,
                        "trigger_word": trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": rank,
                            "linear_alpha": rank,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": save_every,
                            "max_step_saves_to_keep": 4,
                            "push_to_hub": False,
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_dir,
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                # Pre-encodes images to skip VAE on every step.
                                "cache_latents_to_disk": True,
                                "resolution": [resolution],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            "steps": steps,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": True,
                            # Z-Image uses the same flowmatch scheduler as Flux.
                            "noise_scheduler": "flowmatch",
                            "optimizer": "adamw8bit",
                            "lr": learning_rate,
                            "dtype": "bf16",
                        },
                        "model": self._model_block(),
                        **({"sample": {
                            "sampler": "flowmatch",
                            "sample_every": sample_every,
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts,
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 4.5 if self.base_model == "flux_krea" else 4.0,
                            "sample_steps": 20,
                        }} if sample_every < steps else {}),
                    }
                ],
            },
            "meta": {
                "name": output_name,
                "version": "1.0",
            },
        }

    def _model_block(self) -> dict:
        """Return the AI Toolkit model config block for the selected base model."""
        if self.base_model == "flux_dev":
            # FLUX.1-dev — used for first-pass LoRA training in enhanced mode.
            # Downloads from HF Hub via repo ID (gated, requires HF_TOKEN).
            # Quantize transformer to fp8 for faster training (~2x speedup).
            return {
                "name_or_path": self.model_path,
                "is_flux": True,
                "quantize": True,
            }
        if self.base_model == "flux_krea":
            return {
                "name_or_path": self.model_path,
                "is_flux": True,
                # 96 GB VRAM — no need to quantise the transformer.
                "quantize": False,
            }
        # Default: Z-Image De-Turbo
        return {
            "name_or_path": self.model_path,
            "arch": "zimage",
            # RTX PRO 6000 has 96 GB VRAM — transformer quantisation is not needed.
            "quantize": False,
            # Quantise the text encoder to save ~4 GB VRAM.
            "quantize_te": True,
            "qtype_te": "qfloat8",
        }

    # ------------------------------------------------------------------
    # Private helpers — toolkit lifecycle
    # ------------------------------------------------------------------

    def _ensure_toolkit_on_path(self) -> None:
        """
        Prepend ``toolkit_path`` to ``sys.path`` if not already present.

        Raises:
            LoRATrainerError: If the directory does not exist on disk.
        """
        toolkit_dir = os.path.abspath(self.toolkit_path)
        if not os.path.isdir(toolkit_dir):
            raise LoRATrainerError(
                f"AI Toolkit directory not found: {toolkit_dir!r}. "
                "Clone it with: git clone https://github.com/ostris/ai-toolkit"
            )
        if toolkit_dir not in sys.path:
            sys.path.insert(0, toolkit_dir)
            print(
                f"[Chimera] LoRATrainer: added AI Toolkit to sys.path: {toolkit_dir}"
            )

    @staticmethod
    def _set_env() -> None:
        """Set environment variables required or recommended by AI Toolkit."""
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    def _patch_custom_adapter(self) -> None:
        """
        Wrap single-line ``from transformers import …`` statements in
        ``custom_adapter.py`` with try/except so that classes removed in
        transformers 5.x (ViTHybridImageProcessor, etc.) don't crash the
        import.  Multiline imports (parenthesized or backslash-continued)
        are skipped — they contain classes that still exist.
        """
        import re
        import shutil
        import subprocess

        target = os.path.join(self.toolkit_path, "toolkit", "custom_adapter.py")
        if not os.path.isfile(target):
            return

        # Reset to clean upstream state first so we never stack patches
        # on top of a previously-broken file.
        git_dir = os.path.join(self.toolkit_path, ".git")
        if os.path.isdir(git_dir):
            subprocess.run(
                ["git", "checkout", "--", "toolkit/custom_adapter.py"],
                cwd=self.toolkit_path, capture_output=True, timeout=5,
            )

        # Clear cached bytecode so Python re-reads the patched source.
        pycache = os.path.join(self.toolkit_path, "toolkit", "__pycache__")
        if os.path.isdir(pycache):
            shutil.rmtree(pycache, ignore_errors=True)

        with open(target) as f:
            lines = f.readlines()

        patched = 0
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            already_wrapped = i > 0 and lines[i - 1].strip() == "try:"
            if stripped.startswith("from transformers import") and not already_wrapped:
                # Skip multiline imports — backslash continuation or parens
                if stripped.endswith("\\") or ("(" in stripped and ")" not in stripped):
                    i += 1
                    continue
                m = re.match(r"from transformers import (.+)", stripped)
                if m:
                    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                    names = [n.strip() for n in m.group(1).split(",")]
                    block = [
                        indent + "try:\n",
                        indent + "    " + stripped + "\n",
                        indent + "except ImportError:\n",
                    ]
                    for name in names:
                        block.append(indent + "    " + name + " = None\n")
                    lines[i : i + 1] = block
                    patched += 1
                    i += len(block)
                    continue
            i += 1

        if patched:
            with open(target, "w") as f:
                f.writelines(lines)
            print(
                f"[Chimera] Patched AI Toolkit: wrapped {patched} "
                f"transformers import(s) in try/except"
            )

    def _run_toolkit(self, config: dict) -> None:
        """
        Import AI Toolkit's ``run_job`` and execute the training config.

        Args:
            config: AI Toolkit job config dict produced by :meth:`_build_config`.

        Raises:
            LoRATrainerError: If ``toolkit.job`` cannot be imported, most
                likely because ``toolkit_path`` does not point to a valid
                AI Toolkit checkout.
        """
        self._patch_custom_adapter()

        # Monkey-patch transformers to return stubs for removed classes.
        # Belt-and-suspenders: even if the file patcher missed something,
        # this intercepts the __getattr__ lookup that raises ImportError.
        try:
            import transformers as _tf
            _STUB_NAMES = frozenset({
                "ViTHybridImageProcessor", "ViTHybridForImageClassification",
                "ViTFeatureExtractor", "ViTForImageClassification",
            })
            _orig_getattr = _tf.__class__.__getattr__

            def _patched_getattr(self, name):
                if name in _STUB_NAMES:
                    return type(name, (), {})
                return _orig_getattr(self, name)

            _tf.__class__.__getattr__ = _patched_getattr
        except Exception:
            pass

        try:
            from toolkit.job import run_job  # type: ignore[import]
        except ImportError as exc:
            raise LoRATrainerError(
                "Could not import 'toolkit.job' from AI Toolkit. "
                f"Ensure toolkit_path={self.toolkit_path!r} points to a valid "
                "ai-toolkit checkout (https://github.com/ostris/ai-toolkit) and "
                "that its dependencies are installed (pip install -r requirements.txt). "
                f"Original error: {exc}"
            ) from exc

        # Patch tokenizer __call__ to handle None/non-string text inputs.
        # AI Toolkit's encode_prompts_flux passes None as the unconditional
        # prompt text, which crashes transformers 5.x tokenizers.  We patch
        # the tokenizer base class to convert None → "" at the call boundary.
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "_chimera_patched"):
                _orig_call = PreTrainedTokenizerBase.__call__

                def _safe_tokenizer_call(self, text=None, *args, **kwargs):
                    if text is None:
                        text = ""
                    if isinstance(text, list):
                        text = [t if isinstance(t, str) else ("" if t is None else str(t)) for t in text]
                    elif not isinstance(text, str):
                        text = str(text)
                    return _orig_call(self, text, *args, **kwargs)

                PreTrainedTokenizerBase.__call__ = _safe_tokenizer_call
                PreTrainedTokenizerBase._chimera_patched = True
                print("[Chimera] LoRATrainer: patched tokenizer __call__ for None text safety")
        except Exception as exc:
            print(f"[Chimera] LoRATrainer: WARNING — could not patch tokenizer: {exc}")

        run_job(config)

    @staticmethod
    def _find_final_checkpoint(output_dir: str, output_name: str) -> str:
        """
        Locate the last ``.safetensors`` file written by AI Toolkit.

        AI Toolkit writes checkpoints to:
            ``{output_dir}/{output_name}/{output_name}_step{N}.safetensors``

        Files are sorted lexicographically; because step numbers are
        zero-padded by AI Toolkit, the last item in the sorted list is
        always the highest step.

        Args:
            output_dir: Training output root passed to AI Toolkit.
            output_name: Run name used for the output subdirectory.

        Returns:
            Absolute path to the final checkpoint file.

        Raises:
            LoRATrainerError: If no ``.safetensors`` files exist in the
                expected output directory.
        """
        run_dir = Path(output_dir) / output_name
        candidates = sorted(run_dir.glob("*.safetensors"))
        if not candidates:
            raise LoRATrainerError(
                f"No .safetensors checkpoints found in {run_dir}. "
                "Training may have failed silently — check AI Toolkit logs."
            )
        return str(candidates[-1])
