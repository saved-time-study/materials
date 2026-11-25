"""Shared constants for the Stage 1 figures: paths, type sizes, Crameri colours and the panel-letter helper.

Each figure script sets its own matplotlib rcParams and writes its own files, so that the source validator (vendor/validate_figure.py) can read every setting in the script it checks. This file holds only what must be identical
across the figures.

Figures are drawn at their final printed size, 180 mm wide (the double-column width of Nature Human Behaviour), and
saved without a tight bounding box, so that the audited PDF is the size that is printed. Type: Arial in three sizes
(8 pt bold for panel letters, 7 pt for axis labels, 6 pt for tick labels and notes), in one dark ink. Colours are
taken from Fabio Crameri's batlow map (the cmcrameri package), at lightness steps that stay apart in greyscale.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
from cmcrameri import cm  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
QA = HERE / "qa"
VENDOR = HERE / "vendor"  # the panel-alignment check, copied with its licence (vendor/SOURCE.md)
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
# The other check scripts from the same source, also in vendor/, which only run_qa.py runs; they are not needed to
# draw the figures.
SKILL_SCRIPTS = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", VENDOR))

# Three type sizes, one ink.
PANEL_PT = 8
LABEL_PT = 7
SMALL_PT = 6
INK = "#1f1f1f"
MUTED = "#6e6e6e"      # rules next to the data (zero line, release line), never text
RULE = "#bdbdbd"
SHADE = "#f1f1f1"


def batlow(x: float) -> str:
    return mcolors.to_hex(cm.batlow(x))


# Crameri batlow at lightness steps (CIE L* about 24, 40, 65, 78) that stay distinct in greyscale.
NAVY = batlow(0.10)    # #0f3c5f
TEAL = batlow(0.30)    # #30685c
OCHRE = batlow(0.65)   # #cb923e
PINK = batlow(0.85)
GREY = "#d2d2d2"


def panel_label(ax, letter: str, dx_pt: float = -24, dy_pt: float = 6) -> None:
    """Bold lowercase letter at a fixed point offset from the axes' top-left corner."""
    offset = mpl.transforms.ScaledTranslation(dx_pt / 72, dy_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter, transform=ax.transAxes + offset, fontsize=PANEL_PT, fontweight="bold",
            va="bottom", ha="left", color=INK)
