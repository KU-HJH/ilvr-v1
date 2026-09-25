"""Entrypoint for the staged latent image attention analysis CLI.

Inserts the worktree's patched Transformers, `src`, and the repository root
(for `import eval`) at the front of `sys.path` so this runs correctly without
requiring the caller to set `PYTHONPATH` first.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_ROOT, _ROOT / "src", _ROOT / "transformers" / "src"):
    sys.path.insert(0, str(_extra))

from evaluation.latent_image_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
