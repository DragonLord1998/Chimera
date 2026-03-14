"""
Patch transformers 5.x BACKENDS_MAPPING to handle unknown backends gracefully.

RunPod images have leftover TF/keras stubs that cause transformers to crash with:
  "Backend should be defined in the BACKENDS_MAPPING. Offending backend: tf"

This script patches the installed transformers package to skip unknown backends
instead of raising ValueError.

Usage:
    python3 fix_transformers.py
"""

import importlib.util
import os
import sys


def patch():
    spec = importlib.util.find_spec("transformers")
    if not spec or not spec.submodule_search_locations:
        print("transformers not found — nothing to patch.")
        return

    filepath = os.path.join(
        list(spec.submodule_search_locations)[0], "utils", "import_utils.py"
    )
    if not os.path.isfile(filepath):
        print(f"import_utils.py not found at {filepath}")
        return

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    patches = 0

    # Patch 1: _LazyModule.__init__ — unknown backend in module loading
    old1 = (
        '                        else:\n'
        '                            raise ValueError(\n'
        '                                f"Backend should be defined in the BACKENDS_MAPPING.'
        ' Offending backend: {backend}"\n'
        '                            )'
    )
    new1 = (
        '                        else:\n'
        '                            # Unknown backend — treat as unavailable\n'
        '                            callable = lambda: False'
    )
    if old1 in content:
        content = content.replace(old1, new1)
        patches += 1

    # Patch 2: requires decorator — unknown backend
    old2 = (
        '                raise ValueError(f"Backend should be defined in the '
        'BACKENDS_MAPPING. Offending backend: {backend}")'
    )
    new2 = '                continue'
    if old2 in content:
        content = content.replace(old2, new2)
        patches += 1

    # Patch 3: Backend class __init__ — unknown package_name
    old3 = (
        '        if self.package_name not in BACKENDS_MAPPING:\n'
        '            raise ValueError(\n'
        '                f"Backends should be defined in the BACKENDS_MAPPING.'
        ' Offending backend: {self.package_name}"\n'
        '            )'
    )
    new3 = (
        '        if self.package_name not in BACKENDS_MAPPING:\n'
        '            BACKENDS_MAPPING[self.package_name] = '
        '(lambda: False, f"{self.package_name} is not installed.")'
    )
    if old3 in content:
        content = content.replace(old3, new3)
        patches += 1

    if patches > 0:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Patched {patches} location(s) in {filepath}")
    else:
        if content == original:
            print("Already patched or patterns not found.")
        else:
            print("No patches needed.")


def patch_florence2():
    """Patch Florence 2 custom code for transformers 5.x compatibility.

    Fixes:
    1. forced_bos_token_id AttributeError in config
    2. additional_special_tokens AttributeError in processor
    3. _supports_sdpa missing on Florence2ForConditionalGeneration
    4. torch.linspace meta tensor crash during DaViT init
    5. EncoderDecoderCache not subscriptable (past_key_values[0][0] → .get_seq_length())
    """
    import glob

    # Find Florence 2 files in model dir and all possible HF cache locations
    search_paths = [
        "/workspace/models/florence2",
        os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/florence2"),
        "/workspace/.cache/huggingface/modules/transformers_modules/florence2",
        "/root/.cache/huggingface/modules/transformers_modules/florence2",
    ]
    # Also check HF_HOME / TRANSFORMERS_CACHE env vars
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env_val = os.environ.get(env_var)
        if env_val:
            candidate = os.path.join(env_val, "modules", "transformers_modules", "florence2")
            if candidate not in search_paths:
                search_paths.append(candidate)

    patched = 0

    for base_dir in search_paths:
        if not os.path.isdir(base_dir):
            continue

        # --- configuration_florence2.py ---
        cfg_path = os.path.join(base_dir, "configuration_florence2.py")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                content = f.read()
            orig = content
            content = content.replace(
                "if self.forced_bos_token_id is None",
                'if getattr(self, "forced_bos_token_id", None) is None',
            )
            if content != orig:
                with open(cfg_path, "w") as f:
                    f.write(content)
                patched += 1

        # --- processing_florence2.py ---
        proc_path = os.path.join(base_dir, "processing_florence2.py")
        if os.path.isfile(proc_path):
            with open(proc_path, "r") as f:
                content = f.read()
            orig = content
            content = content.replace(
                "tokenizer.additional_special_tokens + ",
                "getattr(tokenizer, 'additional_special_tokens', []) + ",
            )
            if content != orig:
                with open(proc_path, "w") as f:
                    f.write(content)
                patched += 1

        # --- modeling_florence2.py ---
        model_path = os.path.join(base_dir, "modeling_florence2.py")
        if os.path.isfile(model_path):
            with open(model_path, "r") as f:
                content = f.read()
            orig = content

            # Fix _supports_sdpa / _supports_flash_attn_2 property -> getattr
            content = content.replace(
                "return self.language_model._supports_sdpa",
                'return getattr(self.language_model, "_supports_sdpa", True)',
            )
            content = content.replace(
                "return self.language_model._supports_flash_attn_2",
                'return getattr(self.language_model, "_supports_flash_attn_2", False)',
            )

            # Add class attributes to Florence2ForConditionalGeneration
            old_class = "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n    _tied_weights_keys"
            new_class = (
                "class Florence2ForConditionalGeneration(Florence2PreTrainedModel):\n"
                "    _supports_sdpa = True\n"
                "    _supports_flash_attn_2 = True\n"
                "    _tied_weights_keys"
            )
            if old_class in content and "_supports_sdpa = True\n    _supports_flash_attn_2 = True\n    _tied" not in content:
                content = content.replace(old_class, new_class)

            # Fix meta tensor crash in DaViT torch.linspace
            content = content.replace(
                "torch.linspace(0, drop_path_rate, sum(depths)*2)",
                'torch.linspace(0, drop_path_rate, sum(depths)*2, device="cpu")',
            )

            # Fix EncoderDecoderCache not subscriptable (transformers 5.x)
            content = content.replace(
                "past_length = past_key_values[0][0].shape[2]",
                'past_length = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else past_key_values[0][0].shape[2]',
            )
            content = content.replace(
                "past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0",
                'past_key_values_length = (past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else past_key_values[0][0].shape[2]) if past_key_values is not None else 0',
            )
            content = content.replace(
                "past_key_value = past_key_values[idx] if past_key_values is not None else None",
                'past_key_value = (past_key_values.self_attention_cache[idx] if hasattr(past_key_values, "self_attention_cache") else past_key_values[idx]) if past_key_values is not None else None',
            )

            if content != orig:
                with open(model_path, "w") as f:
                    f.write(content)
                patched += 1

    if patched > 0:
        print(f"[Florence 2] Patched {patched} file(s) for transformers 5.x compat.")
    else:
        print("[Florence 2] Already patched or files not found.")


if __name__ == "__main__":
    patch()
    patch_florence2()
