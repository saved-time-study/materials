"""Figure 2: the design of the experiment at a glance. A hand-built vector schematic, written as SVG at its final size (510 pt wide; one drawing unit is one point), then printed to PDF by headless
Microsoft Edge and rendered to a 600 dpi PNG with PyMuPDF.

Run from the repository's root:  python figures/figure2_build.py

Content source: materials/SESSION-PAGES.md, the only source for what participants see and when.
Every line that participants see is set in quotation marks, and the build refuses a quoted line that is not in
materials/SESSION-PAGES.md word for word (check_quotes). Every other label is a short description; the details are in the legend and
the Methods.

Panel a, the four groups: one timeline, the same for every group and drawn to scale (review, 3 minutes; choosing,
1 minute; the break, 10 minutes; the questions, about 2 minutes), and a table of the two things that differ: what
Screen 1 told each group, in a few words, and the sentence that says why the review ends. Panel b, the choices in the
break, in three steps: the pick of five of the ten videos, the five 2-minute slots (a video or paid typing, or leave
with full pay) and the two outcomes, with the example of the Methods: three fun and two
self-improvement videos picked (a share of fun of 60 percent) and two slots given to paid typing (4 minutes).

What the drawing takes from elsewhere, or assumes, beyond materials/SESSION-PAGES.md:
- The questions are drawn for about 2 minutes: about 30 seconds before assignment, 30 seconds on Screen 1, 14
  minutes from minute 0 to the questions and about 2 minutes for the questions and the debriefing add up to the 17
  listed minutes.
- The page shows the ten tiles in one random order per person; the order drawn is one such order. The fills that tell
  fun from self-improvement are for the reader only: the page shows titles, with no category labels.
- In the example, the participant gives slots 2 and 5 to paid typing; any two slots would do.

Element ids name the components (n: for a component, c: for a container), so that a label can be changed in the SVG and
printed again with this script's --pdf-only option.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from html import escape
from pathlib import Path

from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import HERE, INK, NAVY, OCHRE, REPO, RULE, TEAL  # noqa: E402

SVG = HERE / "figure2.svg"
PDF = HERE / "figure2.pdf"
PNG = HERE / "figure2.png"
SCREENS = REPO / "materials" / "SESSION-PAGES.md"

W = 510.0              # 179.9 mm: whole points, because Edge rounds a printed page to whole points; the PDF is then
#                        exactly the drawing's size. Edge also stores type sizes in hundredths of a pixel, so 6 pt
#                        type prints at 5.99 pt.
H = 0.0                # set by build() from the lowest element, rounded up to a whole point
FONT = "Arial, Helvetica, sans-serif"


def arial_file(windows_file: str, weight: str) -> str:
    """Arial's font file: Windows' own copy, or else the file that matplotlib finds for Arial (macOS, Linux)."""
    if Path(windows_file).exists():
        return windows_file
    from matplotlib import font_manager
    return font_manager.findfont(font_manager.FontProperties(family="Arial", weight=weight),
                                 fallback_to_default=False)


ARIAL = ImageFont.truetype(arial_file("C:/Windows/Fonts/arial.ttf", "normal"), 1000)
ARIAL_BOLD = ImageFont.truetype(arial_file("C:/Windows/Fonts/arialbd.ttf", "bold"), 1000)
LQ, RQ = "\u201c", "\u201d"

# Colours: grey for the review and the questions, a pale tint of batlow's teal for the break and the videos, navy for
# paid typing, ochre (the one accent) for the line that only the labeled group reads, and two fills that differ in
# lightness (CIE L* about 91 and 75, so that they stay apart in greyscale) for the two kinds of video.
REVIEW_FILL = "#e7e7e7"
FREE_FILL = "#dbe8e3"
FUN_FILL = "#f6e2c4"
SELF_FILL = "#a8bccc"
EDGE = "#8c8c8c"


def tw(text: str, size: float, bold: bool = False) -> float:
    """Width of a text run in points, measured with Arial's own metrics."""
    return (ARIAL_BOLD if bold else ARIAL).getlength(text) / 1000 * size


# ---- Screen text: every quoted line must be in materials/SESSION-PAGES.md word for word -------------------------------------------
QUOTED: list[str] = []


def q(line: str) -> str:
    """A line that participants see, set in quotation marks and recorded for check_quotes()."""
    QUOTED.append(line)
    return f"{LQ}{line}{RQ}"


def check_quotes() -> None:
    screens = " ".join(SCREENS.read_text(encoding="utf-8").split())
    assert "page by page" in screens[:200], "materials/SESSION-PAGES.md is not the session's pages"
    missing = [s for s in QUOTED if " ".join(s.split()) not in screens]
    assert not missing, f"quoted as screen text but not in materials/SESSION-PAGES.md: {missing}"


class Canvas:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.late: list[str] = []      # drawn after everything else (the step numbers; see stage_mark)
        self.open_groups = 0
        self.bottom = 0.0

    def low(self, y: float) -> None:
        self.bottom = max(self.bottom, y)

    def group(self, gid: str) -> None:
        self.parts.append(f'<g id="{escape(gid)}">')
        self.open_groups += 1

    def end(self) -> None:
        self.parts.append("</g>")
        self.open_groups -= 1

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=0.6, rx=0.0) -> None:
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.low(y + h)

    def line(self, x1, y1, x2, y2, stroke=INK, sw=0.6) -> None:
        self.parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
                          f'stroke-width="{sw}"/>')
        self.low(max(y1, y2))

    def poly(self, points, fill=INK, stroke="none", sw=0.6) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self.low(max(y for _, y in points))

    def circle(self, cx, cy, r, fill="none", stroke=INK, sw=0.6) -> None:
        self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
                          f'stroke-width="{sw}"/>')
        self.low(cy + r)

    def text(self, x, y, s, size=6, anchor="start", bold=False, fill=INK) -> None:
        weight = ' font-weight="bold"' if bold else ""
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}"{weight} '
                          f'fill="{fill}">{escape(s, quote=False)}</text>')
        start = {"start": x, "middle": x - tw(s, size, bold) / 2, "end": x - tw(s, size, bold)}[anchor]
        assert start >= -0.01 and start + tw(s, size, bold) <= W + 0.01, f"text runs off the page: {s!r}"
        self.low(y + 0.22 * size)  # Arial's descent

    def arrow(self, x1, y, x2, sw=0.7) -> None:
        """A horizontal arrow from x1 to x2 with a filled head."""
        self.line(x1, y, x2 - 3.2, y, sw=sw)
        self.poly([(x2, y), (x2 - 3.6, y - 1.8), (x2 - 3.6, y + 1.8)])

    def svg(self, height: float) -> str:
        assert self.open_groups == 0
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}pt" height="{height:.0f}pt" '
                f'viewBox="0 0 {W:.3f} {height:.2f}" font-family="{FONT}">')
        return "\n".join([head, f'<rect x="0" y="0" width="{W:.3f}" height="{height:.2f}" fill="white"/>',
                          *self.parts, '<g id="n:step-numbers">', *self.late, "</g>", "</svg>"]) + "\n"


def stage_mark(c: Canvas, cx: float, cy: float, n: str, r: float = 3.6) -> None:
    """A circled step number, drawn last (Canvas.late). Drawn just before its step's title, the number and the title
    come out of the PDF as one line of text, which the circle then crosses."""
    main, c.parts = c.parts, c.late
    c.circle(cx, cy, r, fill="white", stroke=INK, sw=0.6)
    c.text(cx, cy + 2.1, n, size=6, anchor="middle")
    c.late, c.parts = c.parts, main


def play_icon(c: Canvas, cx: float, cy: float, fill: str) -> None:
    c.poly([(cx - 2.0, cy - 2.6), (cx - 2.0, cy + 2.6), (cx + 2.8, cy)], fill=fill)


def boxed_text(c: Canvas, x0: float, x1: float, y: float, s: str, size: float = 6, fill: str = INK) -> None:
    """Centre a label in the span x0 to x1 and check that it fits with at least 2 pt on each side."""
    assert tw(s, size) + 4 <= x1 - x0, f"{s!r} does not fit in {x1 - x0:.1f} pt"
    c.text((x0 + x1) / 2, y, s, size=size, anchor="middle", fill=fill)


# Panel a. The timeline: (name, length in minutes, second line, fill, outline). The questions take about 2 minutes
#.
SEGMENTS = [("Review", 3.0, "3 sections, 3 min", REVIEW_FILL, EDGE),
            ("Choose", 1.0, "1 min", "white", TEAL),
            ("Break", 10.0, "10 min", FREE_FILL, TEAL),
            ("Questions", 2.0, None, REVIEW_FILL, EDGE)]
# The four groups: name, what Screen 1 told them (a description, not quoted), the sentence at minute 3 that says why
# the review ends (the one sentence of the message that differs), and the line that only the labeled group reads.
AI_FIRST = "The AI assistant has reviewed the remaining sections of the plan itself."
GROUPS = [
    ("planned", "Planned break", "3 sections, then a 10-minute break", "You have reviewed the three sections.", None),
    ("found", "Found break", "all 13 sections",
     "The remaining sections of the plan will not be reviewed in this session.", None),
    ("ai-saved", "AI-saved time", "all 13 sections", AI_FIRST, None),
    ("ai-saved-labeled", "AI-saved time, labeled", "all 13 sections", AI_FIRST, "That saved you 10 minutes."),
]

# Panel b: the ten tiles in one mixed order (F fun, S self-improvement), the five picks and the two typing slots of the
# example.
TILES = "FSSFSFFSFS"
PICKED = {0, 2, 5, 6, 9}       # three fun (tiles 0, 5, 6) and two self-improvement (tiles 2, 9)
WORK_SLOTS = {2, 5}            # the example gives slots 2 and 5 to paid typing


def build() -> Canvas:
    QUOTED.clear()
    c = Canvas()

    # ---- Panel a: the four groups -----------------------------------------------------------------------------------
    c.text(0, 10, "a", size=8, bold=True)
    c.text(11, 10, "The four groups", size=7)

    names = [g[1] for g in GROUPS]
    x0 = math.ceil(max(tw(s, 6, bold=True) for s in names) + 12)   # where the timeline and the second column start
    total = sum(seg[1] for seg in SEGMENTS)
    per_min = (W - 0.5 - x0) / total

    # One timeline, the same for every group, to scale.
    c.group("n:timeline")
    ty, th = 22.0, 26.0
    c.text(0, ty + th / 2 + 2.1, "Every group", size=6)
    x = x0
    for name, minutes, second, fill, outline in SEGMENTS:
        w = minutes * per_min
        c.rect(x, ty, w, th, fill=fill, stroke=outline, sw=0.6)
        if second:
            boxed_text(c, x, x + w, ty + 11.0, name)
            boxed_text(c, x, x + w, ty + 19.2, second)
        else:
            boxed_text(c, x, x + w, ty + th / 2 + 2.1, name)
        x += w
    assert abs(x - (W - 0.5)) < 0.01
    c.end()

    # The table: what Screen 1 told each group and why the review ends.
    c.group("n:groups-table")
    col2 = x0
    col3 = col2 + max(tw(g[2], 6) for g in GROUPS) + 22
    hy = ty + th + 20
    c.text(col2, hy, "Screen 1 told them", size=6)
    c.text(col3, hy, "Why the review ends", size=6)
    c.line(0, hy + 4, W - 0.5, hy + 4, stroke=RULE, sw=0.5)
    pitch = 16.0
    for i, (gid, name, told, why, labeled) in enumerate(GROUPS):
        y = hy + 18 + i * pitch
        c.group(f"c:row-{gid}")
        c.text(0, y, name, size=6, bold=True)
        c.text(col2, y, told, size=6)
        sentence = q(why)
        c.text(col3, y, sentence, size=6)
        if labeled:
            label = q(labeled)
            lx = col3 + tw(sentence, 6) + 7
            lw = tw(label, 6) + 8
            assert lx + lw <= W - 0.5, "the labeled line does not fit beside the sentence"
            c.rect(lx, y - 6.8, lw, 9.8, fill="white", stroke=OCHRE, sw=0.8, rx=1.5)
            c.text(lx + 4, y, label, size=6)
        c.end()
    table_bottom = hy + 18 + (len(GROUPS) - 1) * pitch
    c.end()

    # ---- Panel b: the choices in the break --------------------------------------------------------------------------
    py = table_bottom + 32
    c.text(0, py, "b", size=8, bold=True)
    c.text(11, py, "The choices in the break", size=7)

    sy = py + 18                # baseline of the three step titles
    gy = sy + 8                 # top of the drawings of the three steps
    gh = 18.0                   # height of a tile, a slot and an outcome box
    gutter = 24.0               # the space for each arrow
    c1x, c1w = 0.8, 180.0       # 0.8 pt in, so that the 1.2 pt outline of a picked tile is not cut by the page edge
    c2x, c2w = c1x + c1w + gutter, 140.0
    c3x = c2x + c2w + gutter
    c3w = W - 0.5 - c3x
    for k, (cx, title) in enumerate(((c1x, "Pick 5 of 10 videos"), (c2x, "Each 2-minute slot"),
                                     (c3x, "Two outcomes")), start=1):
        stage_mark(c, cx + 3.8, sy - 2.4, str(k))
        c.text(cx + 11, sy, title, size=7)
    c.arrow(c1x + c1w + 4, gy + gh / 2, c2x - 4)
    c.arrow(c2x + c2w + 4, gy + gh / 2, c3x - 4)

    # Step 1: ten tiles in two rows of five, in one mixed order; the five picks are outlined.
    c.group("n:step-1-pick")
    gap, row_gap = 5.0, 4.0
    tile_w = (c1w - 4 * gap) / 5
    for k, kind in enumerate(TILES):
        x = c1x + (k % 5) * (tile_w + gap)
        y = gy + (k // 5) * (gh + row_gap)
        chosen = k in PICKED
        c.rect(x, y, tile_w, gh, fill=FUN_FILL if kind == "F" else SELF_FILL,
               stroke=INK if chosen else EDGE, sw=1.2 if chosen else 0.4, rx=1.2)
        play_icon(c, x + tile_w / 2, y + gh / 2, INK if chosen else "#5f5f5f")
    assert TILES.count("F") == 5 and sum(TILES[k] == "F" for k in PICKED) == 3 and len(PICKED) == 5
    # Key: the fills are added for the reader; the page shows titles only.
    ky = gy + 2 * gh + row_gap + 11
    kx = c1x
    for fill, stroke, sw, lab in ((FUN_FILL, EDGE, 0.4, "Fun"), (SELF_FILL, EDGE, 0.4, "Self-improvement"),
                                  ("white", INK, 1.2, "Picked")):
        c.rect(kx, ky - 5.6, 8, 6, fill=fill, stroke=stroke, sw=sw, rx=0.8)
        c.text(kx + 11, ky, lab, size=6)
        kx += 11 + tw(lab, 6) + 9
    assert kx <= c1x + c1w + 9
    c.end()

    # Step 2: the five slots of the example (slots 2 and 5 to paid typing), and the other way out.
    c.group("n:step-2-slots")
    slot_gap = 3.0
    slot_w = (c2w - 4 * slot_gap) / 5
    for k in range(1, 6):
        x = c2x + (k - 1) * (slot_w + slot_gap)
        work = k in WORK_SLOTS
        c.rect(x, gy, slot_w, gh, fill=NAVY if work else FREE_FILL, stroke=NAVY if work else TEAL, sw=0.6)
        boxed_text(c, x, x + slot_w, gy + gh / 2 + 2.1, "Typing" if work else "Video",
                   fill="white" if work else INK)
    choice = "A video or paid typing (0.10 USD an item)"
    assert tw(choice, 6) <= c2w
    c.text(c2x, gy + gh + 10, choice, size=6)
    leave = "Or leave now, with full pay"
    lt = gy + gh + 15.5
    c.rect(c2x, lt, tw(leave, 6) + 8, 10.0, fill="white", stroke=INK, sw=0.5, rx=1.5)
    c.text(c2x + 4, lt + 7.0, leave, size=6)
    c.end()

    # Step 3: the two outcomes, with the values of the example.
    c.group("n:outcomes")
    outcomes = [("Share of fun videos among the picks", "60%"), ("Minutes of paid extra work", "4 min")]
    for k, (lab, value) in enumerate(outcomes):
        bt = gy + k * (gh + row_gap)
        c.rect(c3x, bt, c3w, gh, fill="white", stroke=EDGE, sw=0.5, rx=1.5)
        assert tw(lab, 6) + tw(value, 7, bold=True) + 18 <= c3w, lab
        c.text(c3x + 5, bt + gh / 2 + 2.1, lab, size=6)
        c.text(c3x + c3w - 5, bt + gh / 2 + 2.5, value, size=7, bold=True, anchor="end")
    c.end()

    check_quotes()
    return c


def print_pdf() -> None:
    """Print the SVG to a one-page PDF of the same size with headless Edge, ended by its own process id."""
    edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    if not edge.exists():
        edge = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not edge.exists():  # macOS or Linux: Chrome, Chromium or Edge, wherever it is installed
        found = [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")]
        edge = Path(next((f for f in found if f), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"))
    svg_text = SVG.read_text(encoding="utf-8")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><style>@page {{ size: {W:.0f}pt {H:.0f}pt; "
            f"margin: 0 }} html, body {{ margin: 0; padding: 0 }} svg {{ display: block }}</style></head><body>"
            f"{svg_text}</body></html>")
    work = Path(tempfile.mkdtemp(prefix="figure2-"))
    try:
        page = work / "figure2.html"
        page.write_text(html, encoding="utf-8")
        out = work / "figure2.pdf"
        args = [str(edge), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                f"--user-data-dir={work / 'profile'}", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                f"--print-to-pdf={out}", page.as_uri()]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                else:
                    proc.kill()
        for _ in range(20):
            if out.exists() and out.stat().st_size > 0:
                break
            time.sleep(0.25)
        shutil.copyfile(out, PDF)
    finally:
        for _ in range(10):  # Edge's helper processes can hold its profile for a moment after it exits
            shutil.rmtree(work, ignore_errors=True)
            if not work.exists():
                break
            time.sleep(0.5)
        if work.exists():
            print(f"warning: the temporary folder {work} could not be removed; it holds only Edge's profile")


def printed_at_drawing_size() -> bool:
    """True when Edge printed the page at the drawing's own size, so that nothing was scaled."""
    import pymupdf

    rect = pymupdf.open(PDF)[0].rect
    return abs(rect.width - W) < 0.05 and abs(rect.height - H) < 0.05


def check_pdf_and_render_png() -> None:
    import pymupdf

    doc = pymupdf.open(PDF)
    assert doc.page_count == 1, f"the PDF has {doc.page_count} pages"
    page = doc[0]
    width_mm, height_mm = page.rect.width / 72 * 25.4, page.rect.height / 72 * 25.4
    # The page must be the drawing's own size, so that nothing was scaled when it was printed.
    assert printed_at_drawing_size(), (page.rect, W, H)
    fonts = page.get_fonts()  # (xref, file type, font type, base font, name, encoding); file type "n/a" = not embedded
    assert fonts and all(f[2] != "Type3" for f in fonts), fonts
    assert all(f[1] in ("ttf", "cff", "otf") for f in fonts), f"a font is not embedded: {fonts}"
    words = page.get_text("words")
    assert any("minutes" in w[4] for w in words), "no selectable text in the PDF"
    page.get_pixmap(dpi=600, alpha=False).save(PNG)
    print(f"figure2.pdf: {width_mm:.1f} x {height_mm:.1f} mm, fonts {[f[3] for f in fonts]}, {len(words)} words")


def main() -> None:
    global H
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-only", action="store_true", help="print the existing figure2.svg again")
    args = parser.parse_args()
    if args.pdf_only:
        H = float(re.search(r'height="([\d.]+)pt"', SVG.read_text(encoding="utf-8")).group(1))
        print_pdf()
    else:
        canvas = build()
        # Edge rounds some page heights (471 pt printed as 470.88 pt), so the height is the first whole point at least
        # 3 pt below the lowest element that Edge prints exactly.
        for extra in range(6):
            H = float(math.ceil(canvas.bottom + 3) + extra)
            SVG.write_text(canvas.svg(H), encoding="utf-8")
            print_pdf()
            if printed_at_drawing_size():
                break
    check_pdf_and_render_png()


if __name__ == "__main__":
    main()
