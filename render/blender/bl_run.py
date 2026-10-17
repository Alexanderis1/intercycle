"""Run a render script inside desktop Blender (standalone 5.1, its bundled Python 3.13).

    blender -b --factory-startup --python render/blender/bl_run.py -- <script.py> [script args...]

Blender's Python has numpy but not pyyaml/pandas/pyarrow/pillow/opencv. Those live in a --target folder
(default D:\\icdata\\pylib, override with INTERCYCLE_PYLIB), appended AFTER Blender's site-packages so
Blender's own numpy wins. Everything after `--` is passed to the script as its argv.
"""
import os
import runpy
import sys
from pathlib import Path

PYLIB = os.environ.get("INTERCYCLE_PYLIB", r"D:\icdata\pylib")
if PYLIB not in sys.path:
    sys.path.append(PYLIB)

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    sys.exit("usage: blender -b --python bl_run.py -- <script.py> [args...]")
script = Path(argv[0]).resolve()
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *argv[1:]]
try:
    runpy.run_path(str(script), run_name="__main__")
except SystemExit as e:              # let Blender exit with the script's code instead of hanging in -b
    if e.code not in (None, 0):
        print(f"[bl_run] {script.name} exited with {e.code}", flush=True)
        os._exit(int(e.code) if isinstance(e.code, int) else 1)
