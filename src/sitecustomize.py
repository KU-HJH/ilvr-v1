"""Temporary observer bootstrap, restricted to the two evidence-gain runs."""
import os
from pathlib import Path
import runpy
import sys

_gain_runs = {'ours1-gain-image-allowed-s40-20260914t172000z',
              'ours1-gain-image-blocked-s40-20260914t172000z'}
if (sys.argv and sys.argv[0].endswith('/src/main.py')
        and '--save_model_path' in sys.argv
        and Path(sys.argv[sys.argv.index('--save_model_path') + 1]).name in _gain_runs):
    try:
        runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          'logs/evidence-gain-20260914/observe.py'))['install']()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(71)
