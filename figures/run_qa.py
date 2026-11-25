"""Run the figure audits of figures/vendor/ on every figure and keep their JSON in figures/qa/.

Run from the repository's root, after the figures are built:  python figures/run_qa.py

For each figure: audit_figure_collisions.py and audit_pdf_text.py on the PDF, validate_figure.py on the script that
draws it, and for Figure 2 also audit_svg_schematic.py on the SVG. The render-time alignment check
(audit_panel_alignment.py) runs inside figure1.py and figure3.py and writes qa/figure1.alignment.json and
qa/figure3.alignment.json; this script reads those files back. It exits 1 if any audit fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QA = HERE / "qa"
sys.path.insert(0, str(HERE))
from style import SKILL_SCRIPTS  # noqa: E402

FIGURES = {
    "figure1": {"source": "figure1.py", "alignment": True},
    "figure2": {"source": "figure2_build.py", "alignment": False, "svg": "figure2.svg"},
    "figure3": {"source": "figure3.py", "alignment": True},
}
# A validate_figure.py FAIL that cannot apply to a script, with the reason and the check that replaces it.
EXEMPT_FAILS = {
    ("figure2", "EDITABLE-TEXT"): (
        "The check looks for matplotlib's svg.fonttype and pdf.fonttype settings; this script checks the embedded fonts of every PDF again below."),
}
# Why each remaining WARN is accepted (they are reported, not hidden).
WARN_REASONS = {
    "EXPORT-RASTER": "PDF is the submission file; the PNG is a 600 dpi preview.",
    "FONT-SIZE": "figure2_build.py sets sizes in SVG attributes (6, 7 and 8 pt), which audit_svg_schematic.py reads.",
    "FINAL-WIDTH": "figure2.svg is 179.9 mm wide, which audit_svg_schematic.py reads.",
    "canvas.width (SVG audit)": "figure2.svg is 510 pt (179.9 mm) wide, inside the 180 mm double column; Edge prints "
                                "a page in whole points, so the drawing is set to 510 pt and is printed unscaled.",
    "text.large (SVG audit)": "the panel letters are 8 pt and the group names, headers and axis title 7 pt, the "
                              "upper two of the three type sizes; the audit's 7 pt ceiling is for ordinary text, and "
                              "7.00 pt is flagged only by rounding.",
}


def embedded_fonts(pdf: Path) -> str:
    import pymupdf

    fonts = pymupdf.open(pdf)[0].get_fonts()  # (xref, file type, font type, base font, ...); "n/a" = not embedded
    bad = [f[3] for f in fonts if f[1] not in ("ttf", "cff", "otf") or f[2] == "Type3"]
    names = sorted({f[3].split("+")[-1] for f in fonts})
    return ("embedded: " + ", ".join(names)) if fonts and not bad else f"NOT EMBEDDED OR TYPE 3: {bad}"


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=HERE)


def main() -> int:
    QA.mkdir(exist_ok=True)
    rows, failed = [], False
    for name, spec in FIGURES.items():
        pdf = HERE / f"{name}.pdf"
        if not pdf.exists():
            rows.append((name, "missing PDF", "", "", "", ""))
            failed = True
            continue
        col = run([str(SKILL_SCRIPTS / "audit_figure_collisions.py"), pdf.name,
                   "--json-out", str(QA / f"{name}.collision-audit.json")])
        collision = json.loads((QA / f"{name}.collision-audit.json").read_text())["verdict"]

        txt = run([str(SKILL_SCRIPTS / "audit_pdf_text.py"), pdf.name, "--min-pt", "5", "--json"])
        (QA / f"{name}.pdf-text-audit.json").write_text(txt.stdout)
        t = json.loads(txt.stdout)
        text = f"min {t['minimum_found_pt']} pt, {t['below_minimum_count']} below 5 pt" if t["auditable"] else "NOT AUDITABLE"

        val = run([str(SKILL_SCRIPTS / "validate_figure.py"), spec["source"], "--json"])
        (QA / f"{name}.validate.json").write_text(val.stdout)
        report = json.loads(val.stdout)
        v = report["summary"]["counts"]
        fails = [f["check_id"] for f in report["findings"] if f["level"] == "FAIL"]
        warns = [f["check_id"] for f in report["findings"] if f["level"] == "WARN"]
        exempt = [c for c in fails if (name, c) in EXEMPT_FAILS]
        open_fails = [c for c in fails if c not in exempt]
        validate = (f"{v['PASS']} pass, {v['WARN']} warn ({', '.join(warns)}), {len(open_fails)} fail"
                    + (f", {len(exempt)} exempt ({', '.join(exempt)})" if exempt else ""))
        fonts = embedded_fonts(pdf)
        failed |= fonts.startswith("NOT")

        alignment = "not applicable (single schematic)"
        if spec["alignment"]:
            a = json.loads((QA / f"{name}.alignment.json").read_text())
            alignment = a["verdict"]

        svg = ""
        if spec.get("svg"):
            s = run([str(SKILL_SCRIPTS / "audit_svg_schematic.py"), spec["svg"], "--json"])
            (QA / f"{name}.svg-audit.json").write_text(s.stdout)
            kinds = sorted({f["kind"] for f in json.loads(s.stdout)["findings"] if f["severity"] == "WARN"})
            svg = (("pass with warnings (" + ", ".join(kinds) + ")") if kinds else "pass") if s.returncode == 0 \
                else f"FAIL (exit {s.returncode})"
            failed |= s.returncode != 0

        failed |= col.returncode != 0 or txt.returncode != 0 or bool(open_fails) or collision != "PASS"
        failed |= spec["alignment"] and alignment != "PASS"
        rows.append((name, collision, text, validate, alignment, svg, fonts))

    keys = ("figure", "collisions", "pdf_text", "validate_source", "alignment", "svg_schematic", "fonts")
    summary = {"figures": [dict(zip(keys, r)) for r in rows],
               "exempt_fails": {f"{n}: {c}": why for (n, c), why in EXEMPT_FAILS.items()},
               "accepted_warnings": WARN_REASONS,
               "verdict": "ALL AUDITS PASS" if not failed else "AT LEAST ONE AUDIT FAILED"}
    (QA / "summary.json").write_text(json.dumps(summary, indent=2))
    summary = summary["figures"]
    for r in summary:
        print(r)
    print("ALL AUDITS PASS" if not failed else "AT LEAST ONE AUDIT FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
