# Slalom path visualization helpers and command-line entry points.
from pathlib import Path

from mjlab import MJLAB_SRC_PATH


# All generated path figures are collected under the Slalom diagnostics tree.
VIZ_PATH_OUTPUT_ROOT = MJLAB_SRC_PATH.parents[1] / "logs" / "diagnostics" / "SQuRo_Slalom" / "path"
