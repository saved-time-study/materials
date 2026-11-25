# Figures 1 to 3

This folder holds Figures 1 to 3 of the manuscript, the scripts that draw them, their legends and the checks that each
figure passed. A script reads every number in a figure and in a legend from a results file.

## Files

| File | What it is |
|---|---|
| figure1.py | Figure 1, the time diaries and the model of Jiang and colleagues; reads data/atus/results/ |
| figure2_build.py | Figure 2, the experiment; checks that every line it quotes as screen text is in materials/SESSION-PAGES.md word for word, writes figure2.svg at its printed size, prints it to PDF with Microsoft Edge or Google Chrome, and renders the PNG |
| figure3.py | Figure 3, what each pattern of true differences predicts and the share of simulated data sets with each result; reads analysis/output/power_by_simulation.json and analysis/registered_constants.py |
| style.py | The colours, type sizes and panel letters that the three figures share |
| make_captions.py, captions.md | The three legends, which the script writes from the results files, each followed by the file and key of its numbers |
| run_qa.py, qa/ | The checks of every figure and their reports; qa/summary.json gives the result of each |
| vendor/ | Five check scripts, copied unchanged from a public repository with their Apache 2.0 licence (vendor/SOURCE.md) |
| figureN.pdf, figureN.png, figureN.svg | Each figure as a vector PDF, a 600 dpi PNG and an SVG |

## Commands

Run these commands from the repository's root, in this order. Each command takes a few seconds.

```
python figures/figure1.py
python figures/figure2_build.py
python figures/figure3.py
python figures/make_captions.py
python figures/run_qa.py
```

The scripts need the Python packages in requirements.txt (matplotlib, numpy, pandas, cmcrameri, PyMuPDF and Pillow) and
the Arial font. Windows and macOS include Arial, and on Linux it is in the Microsoft core fonts package
(ttf-mscorefonts-installer). figure2_build.py also measures each label with Arial's font files and prints the SVG to PDF
with Microsoft Edge or Google Chrome, which must be installed. After a results file changes, run all five commands
again, and the figures and legends show the new numbers.

## The checks

run_qa.py runs four checks on each figure and stops with an error if one fails: audit_figure_collisions.py and
audit_pdf_text.py on the PDF, validate_figure.py on the script, and, for Figure 2, audit_svg_schematic.py on the SVG. It
also checks that every font in each PDF is embedded and that none is Type 3. The panel alignment check
(audit_panel_alignment.py) runs inside figure1.py and figure3.py on the final layout. It stops the script before any
file is written if two panels are out of line by more than 1.5 pt.

qa/summary.json lists every warning that the checks give and the reason why it is accepted, and it names the one check
result that does not apply to Figure 2. That check looks for matplotlib's font settings, and figure2_build.py does not
use matplotlib. Instead, figure2_build.py checks that the fonts of its PDF are embedded TrueType and that the PDF's text
can be selected.

Figure 2 is a drawing of the design and holds no data. Each part of its SVG has an id (n: for a part, c: for a group
of parts), so a label can be changed in the SVG and printed again with `python figures/figure2_build.py --pdf-only`.
