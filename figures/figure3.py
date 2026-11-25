"""Figure 3: what each pattern of true differences predicts for H1 to H5, and how often the registered tests give the
result that the pattern implies.

Run from the repository's root:  python figures/figure3.py

Every number is read from analysis/output/power_by_simulation.json, which analysis/power_by_simulation.py writes:
  panel a  calibration -> <pattern> -> planted -> H1..H5 -> d: the difference planted between the two groups of each
           hypothesis, in standard deviations, in one data set of calibration -> participants_per_group a group.
           Drawn as direction only (no values on the axis).
  panel b  patterns -> <pattern> -> by_bound_form -> <registered bound form> -> H1..H5 -> expected_reading and
           share_by_reading (its share); the number of data sets is <pattern> -> draws. The word in each tile is the
           statistical result behind the expected reading (higher, lower or equivalent), from HYPOTHESES -> readings in
           analysis/registered_constants.py. That file and its BOUND_FORM are read as text (nothing in analysis/ is
           run).
Rows are the four patterns for which the study is powered; columns are H1 to H5 in both panels. The fifth pattern of
the simulation, halfway (AI-saved time halfway between the two breaks), is not drawn: the study is not powered for it,
and its readings split over three results that one tile cannot show (analysis/README.md reports them).
"""
from __future__ import annotations

import ast
import json
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from style import HERE, INK, LABEL_PT, NAVY, PANEL_PT, QA, REPO, RULE, SMALL_PT, TEAL, batlow, panel_label
from audit_panel_alignment import require_matplotlib_panel_alignment  # the alignment check, in figures/vendor/

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",     # editable text in SVG
    "pdf.fonttype": 42,         # editable TrueType text in PDF
    "font.size": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "xtick.minor.size": 1.3,
    "xtick.major.pad": 2,
    "legend.frameon": False,
    "figure.dpi": 150,
})

SIM = REPO / "analysis" / "output" / "power_by_simulation.json"
CONSTANTS = REPO / "analysis" / "registered_constants.py"

PATTERNS = [  # (key in the file, row label)
    ("windfall_account", "Windfall account"),
    ("work_account", "Work account"),
    ("hypotheses_true_at_bound", "All hypotheses\ntrue"),
    ("no_effects", "No difference\nbetween groups"),
]
HYPOTHESES = [  # (key, the two groups compared, left to right, and the outcome)
    ("H1", "Planned → found", "fun videos"),
    ("H2", "Found → AI-saved", "paid work"),
    ("H3", "Planned → AI-saved", "fun videos"),
    ("H4", "Unlabeled → labeled", "fun videos"),
    ("H5", "Found → AI-saved", "fun videos"),
]
# Every result that the registered analysis can give (registered_constants.READINGS); a result outside this list stops
# the script.
RESULTS = {"supported", "opposite", "equivalent", "higher", "lower", "higher than planned", "lower than planned",
           "unresolved", "not interpretable"}
# Panel b's tiles: each names the statistical result that its pattern implies for the hypothesis's two groups (higher,
# lower or equivalent), so that one result has one word and one colour in every column (cold read, item 48); the
# legend says which result supports which hypothesis. The fill is Crameri's batlow at three lightness steps (CIE L*
# about 24, 40 and 80) that stay apart in greyscale, and the word is written in the tile, so that the colour never
# carries the result alone. (fill, text colour, word in the tile)
TILES = {
    "higher": (NAVY, "white", "Higher"),
    "lower": (TEAL, "white", "Lower"),
    "equivalent": (batlow(0.88), INK, "Equivalent"),
}
HSPACE = 0.32


def registered_bound_form() -> str:
    match = re.search(r'^BOUND_FORM\s*=\s*"([a-z_]+)"', CONSTANTS.read_text(encoding="utf-8"), re.M)
    assert match, "BOUND_FORM not found in analysis/registered_constants.py"
    return match.group(1)


def result_of_reading() -> dict:
    """{hypothesis: {reading: result}}, from HYPOTHESES in analysis/registered_constants.py, read without running it."""
    for node in ast.parse(CONSTANTS.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HYPOTHESES":
            hypotheses = ast.literal_eval(node.value)
            return {h: {reading: result for result, reading in spec["readings"].items()}
                    for h, spec in hypotheses.items()}
    raise AssertionError("HYPOTHESES not found in analysis/registered_constants.py")


def load() -> dict:
    sim = json.loads(SIM.read_text(encoding="utf-8"))
    form = registered_bound_form()
    by_pattern = sim["patterns"]
    draws = {by_pattern[key]["draws"] for key, _ in PATTERNS}
    assert len(draws) == 1, f"the patterns were run with different numbers of data sets: {draws}"
    out = {"form": form, "draws": draws.pop(), "source": "patterns",
           "effect_d": float(sim["assumptions"]["effect_d"]), "planted": {}, "shares": {}, "expected": {},
           "result": {}}
    results = result_of_reading()
    for key, _ in PATTERNS:
        planted = sim["calibration"][key]["planted"]
        out["planted"][key] = {h: float(planted[h]["d"]) for h, *_ in HYPOTHESES}
        tests = by_pattern[key]["by_bound_form"][form]
        out["shares"][key] = {h: {r: float(s) for r, s in tests[h]["share_by_reading"].items()} for h, *_ in HYPOTHESES}
        out["expected"][key] = {h: tests[h]["expected_reading"] for h, *_ in HYPOTHESES}
        out["result"][key] = {h: results[h][out["expected"][key][h]] for h, *_ in HYPOTHESES}
        for h, *_ in HYPOTHESES:
            total = sum(out["shares"][key][h].values())
            assert abs(total - 1) < 1e-6, (key, h, total)
            assert set(out["shares"][key][h]) <= RESULTS, out["shares"][key][h]
            assert out["result"][key][h] in TILES, out["result"][key][h]
    return out


def draw_a(ax, d: float, effect_d: float) -> None:
    """Two groups' planted means, the first at zero: the slope shows the direction (the axis has no values)."""
    ax.plot([0, 1], [0, d], color=INK, lw=0.8, zorder=2)
    ax.plot([0, 1], [0, d], "o", ms=3.2, mfc=INK, mec=INK, zorder=3)
    ax.axhline(0, color=RULE, lw=0.5, zorder=1)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-1.3 * effect_d, 1.3 * effect_d)  # H5 falls where the others rise
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_b(ax, shares_by_row: list[dict], expected_by_row: list[str], results_by_row: list[str]) -> None:
    """One tile for each pattern: the result that the pattern implies, and the share of data sets that gave it."""
    half = 0.5 / (1 + HSPACE)  # a tile is exactly as tall as panel a's row, so that the rows line up
    for row, (shares, expected, result) in enumerate(zip(shares_by_row, expected_by_row, results_by_row)):
        fill, ink, word = TILES[result]
        two_lines = "\n" in word
        ax.add_patch(Rectangle((0, row - half), 1, 2 * half, facecolor=fill, edgecolor="none", zorder=1))
        ax.text(0.5, row - (0.15 if two_lines else 0.13), word, ha="center", va="center", fontsize=SMALL_PT,
                color=ink, linespacing=1.1, zorder=2)
        ax.text(0.5, row + (0.22 if two_lines else 0.17), f"{100 * shares[expected]:.1f}%", ha="center",
                va="center", fontsize=PANEL_PT, color=ink, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(shares_by_row) - 1 + half, -half)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    data = load()
    effect_d = data["effect_d"]  # the step between two groups in the simulation, in SD units

    k = len(HYPOTHESES)
    fig = plt.figure(figsize=(180 / 25.4, 74 / 25.4))  # final size: 180 mm by 74 mm
    gs = GridSpec(len(PATTERNS), 2 * k + 1, figure=fig, left=0.115, right=0.99, top=0.775, bottom=0.10,
                  width_ratios=[12.8] * k + [3] + [19.2] * k, wspace=0.12, hspace=HSPACE)
    a_axes = [[fig.add_subplot(gs[i, j]) for j in range(k)] for i in range(len(PATTERNS))]
    b_axes = [fig.add_subplot(gs[:, k + 1 + j]) for j in range(k)]

    for i, (pattern, _) in enumerate(PATTERNS):
        for j, (h, *_) in enumerate(HYPOTHESES):
            draw_a(a_axes[i][j], data["planted"][pattern][h], effect_d)
    for j, (h, *_) in enumerate(HYPOTHESES):
        draw_b(b_axes[j], [data["shares"][p][h] for p, _ in PATTERNS], [data["expected"][p][h] for p, _ in PATTERNS],
               [data["result"][p][h] for p, _ in PATTERNS])

    fig.canvas.draw()
    pt = 1 / (fig.get_size_inches()[1] * 72)  # one point, as a fraction of the figure's height

    # Column headers: the hypothesis, then (panel a) the two groups compared and the outcome.
    for j, (h, groups, outcome) in enumerate(HYPOTHESES):
        pa, pb = a_axes[0][j].get_position(), b_axes[j].get_position()
        xa = (pa.x0 + pa.x1) / 2
        fig.text(xa, pa.y1 + 27 * pt, h, ha="center", va="bottom", fontsize=LABEL_PT)
        fig.text(xa, pa.y1 + 25 * pt, groups.replace(" → ", "\n→ "), ha="center", va="top",
                 fontsize=SMALL_PT, linespacing=1.15)
        fig.text(xa, pa.y1 + 3 * pt, outcome, ha="center", va="bottom", fontsize=SMALL_PT)
        fig.text((pb.x0 + pb.x1) / 2, pa.y1 + 27 * pt, h, ha="center", va="bottom", fontsize=LABEL_PT)
    # Row labels: the four patterns.
    for i, (_, label) in enumerate(PATTERNS):
        p = a_axes[i][0].get_position()
        fig.text(0.012, (p.y0 + p.y1) / 2, label, ha="left", va="center", fontsize=SMALL_PT, linespacing=1.15)
    # Under each panel, one line on what its marks are.
    b0, b_last = b_axes[0].get_position(), b_axes[-1].get_position()
    fig.text((b0.x0 + b_last.x1) / 2, b0.y0 - 5 * pt,
             f"Share of the {data['draws']:,} simulated data sets with the result that the pattern implies",
             ha="center", va="top", fontsize=SMALL_PT)
    pa_bottom = a_axes[-1][0].get_position()
    fig.text((pa_bottom.x0 + a_axes[-1][-1].get_position().x1) / 2, pa_bottom.y0 - 5 * pt,
             f"Predicted group means, direction only (each rise or fall is d = {effect_d:.2f})", ha="center", va="top",
             fontsize=SMALL_PT)

    panel_label(a_axes[0][0], "a", dx_pt=-52.5, dy_pt=29)  # in line with the row labels, 6 pt from the edge
    panel_label(b_axes[0], "b", dx_pt=-12, dy_pt=29)

    QA.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig, json_out=QA / "figure3.alignment.json",
                                       overlay_svg=QA / "figure3.alignment.svg", tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(HERE / "figure3.svg")
    fig.savefig(HERE / "figure3.pdf")
    fig.savefig(HERE / "figure3.png", dpi=600)
    plt.close(fig)
    print(f"figure3: {data['source']} run, {data['draws']} data sets a pattern, bound form {data['form']}")


if __name__ == "__main__":
    main()
