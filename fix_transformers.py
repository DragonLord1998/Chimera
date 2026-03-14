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


if __name__ == "__main__":
    patch()
