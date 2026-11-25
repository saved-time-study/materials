"""Main figure, supplementary figures and the main table, drawn only from data/atus/results/*.csv.

Run after 04_analysis.py:  python -I data/atus/scripts/05_figures_tables.py
Writes results/figure1_event_study.png (and .pdf), figureS1_measures.png, figureS2_leisure_parts.png,
figureS3_yearly_means.png, table1.md, table1.csv and table1.png.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
LABEL = {"leisure": "Leisure", "paid_work": "Paid work", "household": "Household activities",
         "personal_care": "Personal care and sleep", "rest": "All other activities",
         "leis_tv": "Television", "leis_socializing": "Socializing", "leis_sports": "Sports and exercise",
         "leis_computer_games": "Computer leisure and games", "leis_relaxing": "Relaxing and thinking",
         "leis_other": "Other leisure"}
MEAS = {"gpt4_beta": "GPT-4 rating (Eloundou et al.)", "human_beta": "Human rating (Eloundou et al.)",
        "lm_aioe": "Language-modeling exposure (Felten et al.)", "aioe": "AI exposure (Felten et al.)"}
INK, ACCENT, GREY = "#2b2b2b", "#c8553d", "#9a9a9a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7, "savefig.dpi": 300})


def series(est, spec, measure, outcome, ref):
    s = est[(est["spec"] == spec) & (est["measure"] == measure) & (est["outcome"] == outcome)].copy()
    s["year"] = s["term"].str[3:].astype(int)
    ref_row = pd.DataFrame([{"year": ref, "coef": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}])
    return pd.concat([s[["year", "coef", "ci_lo", "ci_hi"]], ref_row]).sort_values("year")


def xpos(year):
    return YEARS.index(year)


def decorate(ax, ref):
    ax.axhline(0, color=GREY, lw=0.7)
    ax.axvspan(xpos(2022) + 0.5, len(YEARS) - 0.5, color=ACCENT, alpha=0.07, lw=0)
    ax.axvline(xpos(2019) + 0.5, color=GREY, lw=0.6, ls=(0, (1, 2)))
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels([str(y)[2:] if y != ref else f"{str(y)[2:]}\nref." for y in YEARS])
    ax.set_xlim(-0.6, len(YEARS) - 0.4)


def plot_one(ax, s, color, offset=0.0, label=None, ms=4):
    x = [xpos(y) + offset for y in s["year"]]
    ax.errorbar(x, s["coef"], yerr=[s["coef"] - s["ci_lo"], s["ci_hi"] - s["coef"]], fmt="o", ms=ms,
                color=color, ecolor=color, elinewidth=0.9, capsize=0, label=label)


def figure1(est):
    ref, spec = 2019, "event_study_ref2019"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for ax, o in zip(axes.flat, ["leisure", "paid_work", "household", "personal_care"]):
        plot_one(ax, series(est, spec, "gpt4_beta:third", o, ref), INK)
        decorate(ax, ref)
        ax.set_title(LABEL[o], fontsize=9, loc="left", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Minutes a day, most exposed third\nminus least exposed third, vs 2019")
    for ax in axes[1]:
        ax.set_xlabel("Diary year (2020 not collected for most of the year)")
    axes[0, 0].text(xpos(2023) - 0.3, axes[0, 0].get_ylim()[1] * 0.92, "after ChatGPT", color=ACCENT,
                    fontsize=7.5, va="top")
    fig.tight_layout()
    fig.savefig(RES / "figure1_event_study.png")
    fig.savefig(RES / "figure1_event_study.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def figureS1(est):
    ref, spec = 2019, "event_study_ref2019"
    colors = [INK, "#5b7fa6", ACCENT, "#7a9a5a"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharex=True)
    for ax, o in zip(axes.flat, ["leisure", "paid_work", "household", "personal_care"]):
        for i, (m, c) in enumerate(zip(MEAS, colors)):
            plot_one(ax, series(est, spec, f"{m}:z", o, ref), c, offset=(i - 1.5) * 0.17,
                     label=MEAS[m], ms=3)
        decorate(ax, ref)
        ax.set_title(LABEL[o], fontsize=9, loc="left", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Minutes a day per standard deviation\nof exposure, vs 2019")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(RES / "figureS1_measures.png")
    plt.close(fig)


def figureS2(est):
    ref, spec = 2019, "event_study_ref2019"
    parts = ["leis_tv", "leis_socializing", "leis_sports", "leis_computer_games", "leis_relaxing", "leis_other"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for ax, o in zip(axes.flat, parts):
        plot_one(ax, series(est, spec, "gpt4_beta:third", o, ref), INK, ms=3)
        decorate(ax, ref)
        ax.set_title(LABEL[o], fontsize=8.5, loc="left", fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Minutes a day, top minus\nbottom third, vs 2019")
    fig.tight_layout()
    fig.savefig(RES / "figureS2_leisure_parts.png")
    plt.close(fig)


def figureS3():
    y = pd.read_csv(RES / "yearly_means_by_third.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    names = {1: "Least exposed third", 2: "Middle third", 3: "Most exposed third"}
    for ax, o in zip(axes, ["leisure", "paid_work", "household", "personal_care"]):
        for t, c in [(1, GREY), (2, "#5b7fa6"), (3, ACCENT)]:
            s = y[y["third"] == t]
            ax.plot([xpos(v) for v in s["year"]], s[o], "o-", color=c, ms=2.5, lw=1, label=names[t])
        ax.set_xticks(range(0, len(YEARS), 2))
        ax.set_xticklabels([str(YEARS[i])[2:] for i in range(0, len(YEARS), 2)])
        ax.set_title(LABEL[o].replace(" and sleep", ""), fontsize=8, loc="left", fontweight="bold")
    axes[0].set_ylabel("Weighted mean, minutes a day")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(RES / "figureS3_yearly_means.png")
    plt.close(fig)


def table1(est):
    cols = [
        ("main", "gpt4_beta:third", "Before: 2019,\n2021 and 2022\nTop vs bottom third"),
        ("main", "gpt4_beta:z", "Before: 2019,\n2021 and 2022\nPer SD, GPT-4 rating"),
        ("prepandemic_baseline", "gpt4_beta:third", "Before:\n2015 to 2019\nTop vs bottom third"),
        ("prepandemic_baseline", "gpt4_beta:z", "Before:\n2015 to 2019\nPer SD, GPT-4 rating"),
        ("prepandemic_baseline", "lm_aioe:z", "Before:\n2015 to 2019\nPer SD, language-\nmodeling exposure\n(Felten et al.)"),
    ]
    outs = ["leisure", "paid_work", "household", "personal_care", "rest"]
    rows, flat = [], []
    for o in outs:
        cells = []
        for spec, m, head in cols:
            r = est[(est["spec"] == spec) & (est["measure"] == m) & (est["outcome"] == o) &
                    (est["sample"] == "all") & (est["term"] == "x_post")].iloc[0]
            cells.append(f"{r['coef']:+.1f}\n[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]")
            flat.append({"outcome": o, "column": head.replace("\n", "; "), "coef": r["coef"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"], "p_holm_4": r["p_holm_4"],
                         "mde": r["mde"], "n": r["n"], "clusters": r["clusters"], "before_mean": r["pre_mean"]})
        base = est[(est["spec"] == "prepandemic_baseline") & (est["measure"] == "gpt4_beta:z") &
                   (est["outcome"] == o) & (est["term"] == "x_post") & (est["sample"] == "all")]["pre_mean"].iloc[0]
        rows.append([f"{LABEL[o]}\n(2015-19 mean {base:.0f} min)"] + cells)
    f = pd.DataFrame(flat)
    nrow, mrow, crow = ["Diaries"], ["Smallest detectable\nshift (minutes)"], ["Occupations (clusters)"]
    for _, _, head in cols:
        g = f[f["column"] == head.replace("\n", "; ")]
        nrow.append(f"{int(g['n'].iloc[0]):,}")
        crow.append(f"{int(g['clusters'].iloc[0])}")
        free = g[g["outcome"] != "rest"]["mde"]
        mrow.append(f"{free.min():.0f} to {free.max():.0f}")
    rows += [mrow, nrow, crow]
    f.to_csv(RES / "table1.csv", index=False, float_format="%.10g", lineterminator="\n")

    heads = ["Minutes a day"] + [h for _, _, h in cols]
    md = ["| " + " | ".join(h.replace("\n", "<br>") for h in heads) + " |",
          "|" + "---|" * len(heads)]
    md += ["| " + " | ".join(c.replace("\n", "<br>") for c in r) + " |" for r in rows]
    (RES / "table1.md").write_text("\n".join(md) + "\n", newline="\n")

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=heads, loc="center", cellLoc="center",
                 colWidths=[0.2] + [0.16] * len(cols))
    t.auto_set_font_size(False)
    t.set_fontsize(7.6)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#d0d0d0")
        c.set_linewidth(0.5)
        c.set_height(0.12 if i in range(1, 6) else 0.09)
        if i == 0:
            c.set_text_props(fontweight="bold", fontsize=7.2)
            c.set_height(0.23)
        if j == 0:
            c.set_text_props(ha="left")
    fig.savefig(RES / "table1.png", bbox_inches="tight")
    plt.close(fig)


def main():
    est = pd.read_csv(RES / "estimates.csv")
    figure1(est)
    figureS1(est)
    figureS2(est)
    figureS3()
    table1(est)
    print("figures and table written to", RES)


if __name__ == "__main__":
    main()
