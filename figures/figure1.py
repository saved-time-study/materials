"""Figure 1: the national time diaries before and after ChatGPT, and the reproduction of Jiang et al.

Run from the repository's root:  python figures/figure1.py

Every number is read from data/atus/results/ by this script:
  panel a  estimates.csv, spec event_study_ref2019, measure gpt4_beta:third, sample all,
           outcomes leisure and paid_work, terms ev_2015 to ev_2025 (coef, ci_lo, ci_hi); 2019 is the reference (0).
  panel b  jpxz_estimates.csv, measure gpt4_beta, outcomes work_wk and leisure_wk, columns iqr_hours_week, iqr_ci_lo,
           iqr_ci_hi, for the specs replication_2022_2023, extended_2022_2025_one_post,
           prepandemic_2015_2019_vs_2023 and prepandemic_2015_2019_vs_2024_2025; the published estimates are
           jpxz_key_numbers.json, paper_table2_col1, iqr_work_hours_week and iqr_leisure_hours_week, and their 95%
           intervals are computed from those estimates and their t statistics (work_t, leisure_t) by
           published_interval(): estimate plus or minus 1.96 times estimate / t.
Rows: leisure on top, paid work below, in both panels. Columns: a (our event study) and b (their model).
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import transforms
from matplotlib.gridspec import GridSpec

from style import HERE, INK, LABEL_PT, MUTED, NAVY, OCHRE, QA, REPO, RULE, SHADE, SMALL_PT, panel_label
from audit_panel_alignment import require_matplotlib_panel_alignment  # the render-time alignment check, in figures/vendor/

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
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "axes.labelpad": 3,
    "legend.frameon": False,
    "figure.dpi": 150,
})

RESULTS = REPO / "data" / "atus" / "results"
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
REFERENCE_YEAR = 2019
CHATGPT_RELEASE = 2022 - 0.5 + 334 / 365  # 30 November 2022, with each year's diaries centred on its tick

# Panel b, in reading order. Against 2022: the estimate that Jiang and colleagues published for 2023, our
# rebuild of their model for 2023, and for 2023 to 2025. Against 2015 to 2019: our rebuild for 2023, and
# for 2024 to 2025. Each tick says whose estimate it is and which years it compares with the baseline.
B_MARKS = [
    # (x, spec or "published", baseline, tick label)
    (0.00, "published", "2022", "Published\n2023"),
    (1.25, "replication_2022_2023", "2022", "Rebuilt\n2023"),
    (2.50, "extended_2022_2025_one_post", "2022", "Rebuilt\n2023–2025"),
    (4.00, "prepandemic_2015_2019_vs_2023", "2015-2019", "Rebuilt\n2023"),
    (5.25, "prepandemic_2015_2019_vs_2024_2025", "2015-2019", "Rebuilt\n2024–2025"),
]
B_DIVIDER = 3.25          # the line between the two baselines
B_XLIM = (-0.7, 5.95)
B_GROUPS = ((-0.45, 2.95, "Against 2022", OCHRE), (3.55, 5.70, "Against 2015–2019", NAVY))
BASELINE_COLOUR = {"2022": OCHRE, "2015-2019": NAVY}


def year_colour(year: int) -> str:
    if year <= 2019:
        return NAVY
    if year == 2022:
        return OCHRE
    return INK


def load_panel_a() -> pd.DataFrame:
    est = pd.read_csv(RESULTS / "estimates.csv")
    rows = est[(est.spec == "event_study_ref2019") & (est.measure == "gpt4_beta:third") & (est["sample"] == "all")
               & est.outcome.isin(["leisure", "paid_work"])].copy()
    rows["year"] = rows.term.str.replace("ev_", "", regex=False).astype(int)
    expected = {(o, y) for o in ("leisure", "paid_work") for y in YEARS if y != REFERENCE_YEAR}
    found = set(zip(rows.outcome, rows.year))
    assert found == expected, f"event-study rows differ from the expected years: {sorted(found ^ expected)}"
    assert rows.n.nunique() == 1 and rows.clusters.nunique() == 1
    return rows


PUBLISHED_Z = 1.96  # the normal 97.5th percentile, for a 95% interval from a published estimate and its t statistic


def published_interval(estimate: float, t: float) -> tuple[float, float]:
    """The 95% interval of a published estimate from its t statistic: the standard error is estimate / t."""
    se = estimate / t
    assert se > 0, (estimate, t)
    return estimate - PUBLISHED_Z * se, estimate + PUBLISHED_Z * se


def published_estimates() -> dict:
    """{outcome: (estimate, t, lower, upper)} for the estimates that Jiang and colleagues published (Table 2, col. 1)."""
    published = json.loads((RESULTS / "jpxz_key_numbers.json").read_text())["paper_table2_col1"]
    out = {}
    for outcome, est_key, t_key in (("leisure", "iqr_leisure_hours_week", "leisure_t"),
                                    ("work", "iqr_work_hours_week", "work_t")):
        est, t = float(published[est_key]), float(published[t_key])
        out[outcome] = (est, t, *published_interval(est, t))
    return out


def load_panel_b() -> dict:
    jp = pd.read_csv(RESULTS / "jpxz_estimates.csv")
    published = published_estimates()
    out = {}
    for outcome, col in (("leisure", "leisure_wk"), ("work", "work_wk")):
        marks = []
        for x, spec, baseline, tick in B_MARKS:
            if spec == "published":
                est, _, lo, hi = published[outcome]
                marks.append(dict(x=x, est=est, lo=lo, hi=hi, baseline=baseline, tick=tick, published=True, n=None))
                continue
            row = jp[(jp.spec == spec) & (jp.measure == "gpt4_beta") & (jp.outcome == col)]
            assert len(row) == 1, (spec, col, len(row))
            r = row.iloc[0]
            marks.append(dict(x=x, est=float(r.iqr_hours_week), lo=float(r.iqr_ci_lo), hi=float(r.iqr_ci_hi),
                              baseline=baseline, tick=tick, published=False, n=int(r.n)))
        out[outcome] = marks
    return out


def draw_a(ax, rows: pd.DataFrame, outcome: str, label: str, top: bool) -> None:
    ax.axvspan(CHATGPT_RELEASE, 2025.6, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    ax.axvline(CHATGPT_RELEASE, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    sub = rows[rows.outcome == outcome].sort_values("year")
    ylim = (-75, 50)
    assert sub.ci_lo.min() > ylim[0] and sub.ci_hi.max() < ylim[1], "an interval leaves panel a's axis"
    for r in sub.itertuples():
        c = year_colour(r.year)
        ax.plot([r.year, r.year], [r.ci_lo, r.ci_hi], color=c, lw=0.7, zorder=2)
        ax.plot(r.year, r.coef, "o", ms=3.4, mfc=c, mec=c, mew=0.6, zorder=3)
    # The reference year: a filled diamond, so that a hollow mark means only "published" across the figure.
    ax.plot(REFERENCE_YEAR, 0, "D", ms=3.0, mfc=NAVY, mec=NAVY, mew=0.6, zorder=3)
    ax.set_xlim(2014.4, 2025.6)
    ax.set_ylim(*ylim)
    ax.set_yticks([-60, -40, -20, 0, 20, 40])
    ax.set_xticks([2015, 2017, 2019, 2021, 2023, 2025])
    ax.set_xticks([2016, 2018, 2022, 2024], minor=True)
    ax.text(0.015, 0.97, label, transform=ax.transAxes, fontsize=LABEL_PT, va="top", ha="left")
    if top:
        ax.text(CHATGPT_RELEASE + 0.12, 47, "ChatGPT\nreleased", fontsize=SMALL_PT, va="top", ha="left",
                linespacing=1.15)
        ax.text(REFERENCE_YEAR, 7, "2019 = 0", fontsize=SMALL_PT, va="bottom", ha="center")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Year of the diary", fontsize=LABEL_PT)


def draw_b(ax, marks: list, label: str, top: bool) -> None:
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1)
    ax.axvline(B_DIVIDER, color=RULE, lw=0.5, zorder=0)
    for m in marks:
        c = BASELINE_COLOUR[m["baseline"]]
        # Every estimate has its 95% interval, drawn the same way; the published estimate's mark is hollow.
        ax.plot([m["x"], m["x"]], [m["lo"], m["hi"]], color=c, lw=0.7, zorder=2)
        if m["published"]:
            ax.plot(m["x"], m["est"], "o", ms=3.6, mfc="white", mec=c, mew=0.9, zorder=3)
        else:
            ax.plot(m["x"], m["est"], "o", ms=3.4, mfc=c, mec=c, mew=0.6, zorder=3)
    ylim = (-7.5, 7.5)
    values = [v for m in marks for v in (m["est"], m["lo"], m["hi"])]
    assert min(values) > ylim[0] and max(values) < ylim[1] - 0.8, "an interval leaves panel b's axis"
    ax.set_xlim(*B_XLIM)
    ax.set_ylim(*ylim)
    ax.set_yticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_xticks([m["x"] for m in marks])
    ax.set_xticklabels([m["tick"] for m in marks], linespacing=1.15)
    ax.text(0.985, 0.97, label, transform=ax.transAxes, fontsize=LABEL_PT, va="top", ha="right")
    if top:
        ax.tick_params(labelbottom=False)
    else:
        # Brackets that name the baseline of each group, a fixed distance below the two-line tick labels.
        base = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        line_t = transforms.offset_copy(base, fig=ax.figure, y=-23, units="points")
        text_t = transforms.offset_copy(base, fig=ax.figure, y=-26, units="points")
        for (x0, x1, text, colour) in B_GROUPS:
            ax.plot([x0, x1], [0, 0], color=colour, lw=1.2, transform=line_t, clip_on=False,
                    solid_capstyle="butt")
            ax.text((x0 + x1) / 2, 0, text, transform=text_t, fontsize=SMALL_PT, va="top", ha="center")


def main() -> None:
    rows = load_panel_a()
    marks = load_panel_b()

    fig = plt.figure(figsize=(180 / 25.4, 98 / 25.4))  # final size: 180 mm by 98 mm
    gs = GridSpec(2, 2, figure=fig, left=0.095, right=0.985, top=0.935, bottom=0.165,
                  width_ratios=[1.15, 1.0], wspace=0.25, hspace=0.10)
    a_top, a_bot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    b_top, b_bot = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    draw_a(a_top, rows, "leisure", "Leisure", top=True)
    draw_a(a_bot, rows, "paid_work", "Paid work", top=False)
    draw_b(b_top, marks["leisure"], "Leisure", top=True)
    draw_b(b_bot, marks["work"], "Work", top=False)

    # One y label per column, centred on its two rows, in two lines.
    fig.canvas.draw()
    for top_ax, bot_ax, text in ((a_top, a_bot, "Most minus least exposed third,\nagainst 2019 (minutes a day)"),
                                 (b_top, b_bot, "Change, 75th minus 25th percentile\nof exposure (hours a week)")):
        p_top, p_bot = top_ax.get_position(), bot_ax.get_position()
        ticks = max(t.get_window_extent().width for t in top_ax.get_yticklabels() + bot_ax.get_yticklabels())
        gap_px = (2.5 + 2 + 3) * fig.dpi / 72  # tick length, tick pad and 3 pt of air, in pixels
        x = p_top.x0 - (ticks + gap_px) / fig.bbox.width
        fig.text(x, (p_top.y1 + p_bot.y0) / 2, text, rotation=90, rotation_mode="anchor", ha="center",
                 va="bottom", fontsize=LABEL_PT, linespacing=1.15)

    panel_label(a_top, "a", dx_pt=-36)
    panel_label(b_top, "b", dx_pt=-36)

    # The alignment check on the final layout, then the files at the drawn size (no tight bounding box).
    QA.mkdir(exist_ok=True)
    require_matplotlib_panel_alignment(fig, json_out=QA / "figure1.alignment.json",
                                       overlay_svg=QA / "figure1.alignment.svg", tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(HERE / "figure1.svg")
    fig.savefig(HERE / "figure1.pdf")
    fig.savefig(HERE / "figure1.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
