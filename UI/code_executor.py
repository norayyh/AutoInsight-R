"""
code_executor.py
----------------
Executes generated Python code in an isolated subprocess.

Why subprocess instead of exec():
- exec() shares the host process namespace, meaning generated code can
  import os, modify globals, or cause side effects that corrupt app state.
- subprocess gives us true process isolation, a configurable timeout,
  and clean stdout/stderr capture without any of those risks.

Captured outputs:
- stdout : printed text results
- stderr : tracebacks and error messages
- images : any .png files written to the temp directory are collected
           and returned as raw bytes for display in Streamlit.
"""

import subprocess
import sys
import os
import tempfile
import glob
import textwrap


_HARNESS_TEMPLATE = textwrap.dedent("""\
import os, sys
os.chdir({tmpdir!r})

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Figure counter used by the patched plt.show() replacement.
_fig_count = 0

# Make the dataset available both as a file path and as a DataFrame.
_CSV_PATH = {csv_path!r}
df = pd.read_csv(_CSV_PATH)

{user_code}
""")


def execute_code(code: str, csv_path: str, timeout: int = 30) -> dict:
    """
    Run `code` in a subprocess with access to the CSV at `csv_path`.

    Returns a dict with keys:
        success  (bool)
        stdout   (str)   — printed output
        stderr   (str)   — error / traceback
        images   (list)  — list of raw PNG bytes for any saved figures
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "_autoinsight_exec.py")
        csv_copy    = os.path.join(tmpdir, "data.csv")

        # Copy the CSV into the temp dir so the script can always find it.
        import shutil
        shutil.copy(csv_path, csv_copy)

        # Patch plt.show() to auto-save figures instead of opening a window.
        patched_code = _patch_plt_show(code, tmpdir)

        harness = _HARNESS_TEMPLATE.format(
            tmpdir   = tmpdir,
            csv_path = csv_copy,
            user_code = patched_code,
        )

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(harness)

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output = True,
                text           = True,
                timeout        = timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout":  "",
                "stderr":  f"TimeoutError: execution exceeded {timeout} seconds.",
                "images":  [],
            }

        images = _collect_images(tmpdir)

        if proc.returncode == 0:
            return {
                "success": True,
                "stdout":  proc.stdout.strip(),
                "stderr":  proc.stderr.strip(),
                "images":  images,
            }
        else:
            return {
                "success": False,
                "stdout":  proc.stdout.strip(),
                "stderr":  proc.stderr.strip(),
                "images":  [],
            }


def _patch_plt_show(code: str, tmpdir: str) -> str:
    """
    Replace plt.show() calls with code that saves the current figure to a
    uniquely named PNG file in tmpdir, then closes it.

    Uses a global counter (_fig_count) injected into the harness to name
    files sequentially. This avoids nested f-string quote conflicts that
    arise when embedding os.listdir(tmpdir) inside a generated f-string.

    The save path is built at patch time (Python side) using os.path.join,
    so the generated snippet contains only a plain string literal with no
    nested quotes or f-strings.
    """
    # Build the output path prefix on the Python side to avoid any
    # quote-nesting issues inside the generated code string.
    fig_prefix = os.path.join(tmpdir, "figure_")
    save_snippet = (
        ""
        "_fig_count += 1; "
        "plt.savefig(" + repr(fig_prefix + "{}") + ".format(_fig_count),"
        " dpi=100, bbox_inches=" + repr("tight") + "); "
        "plt.close()"
    )
    return code.replace("plt.show()", save_snippet)


def _collect_images(tmpdir: str) -> list:
    """Return a list of raw PNG bytes for every figure saved in tmpdir."""
    images = []
    for path in sorted(glob.glob(os.path.join(tmpdir, "figure_*.png"))):
        with open(path, "rb") as f:
            images.append(f.read())
    return images
