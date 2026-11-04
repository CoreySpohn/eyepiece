"""Manim stays optional: who imports it, and what a missing install says."""

import subprocess
import sys

import pytest


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_base_modules_never_load_manim():
    code = (
        "import sys\n"
        "import eyepiece, eyepiece.prepared, eyepiece.style, eyepiece.mpl\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('manim', 'manim_slides')))\n"
    )
    assert _run(code) == "[]"


def test_missing_manim_names_the_optional_extra():
    code = (
        "import sys\n"
        "sys.modules['manim'] = None\n"
        "try:\n"
        "    import eyepiece.manim\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    message = _run(code)
    assert "eyepiece[manim]" in message
    assert "Cairo" in message


def test_manim_namespace_never_imports_manim_slides():
    pytest.importorskip("manim")
    code = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'manim_slides':\n"
        "            raise ImportError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import eyepiece.manim as em\n"
        "from eyepiece.manim import Units, ensure_font\n"
        "from eyepiece.manim._render import _ensure_font\n"
        "assert callable(ensure_font)\n"
        "assert _ensure_font is ensure_font\n"
        "assert Units().font_size(24.0) == 24.0\n"
        "print(sorted(em.__all__))\n"
    )
    assert _run(code) == (
        "['ManimClip', 'ManimResult', 'Units', 'animate', 'ensure_font', 'render']"
    )
