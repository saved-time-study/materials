"""Download every public input of the ATUS analysis and write data/atus/MANIFEST.csv.

Run from the repository root:  python -I data/atus/scripts/01_download.py
(with the environment variable BLS_CONTACT_EMAIL set to your e-mail address).
Files land in data/atus/raw/ (kept out of git). A file already present is not downloaded again,
but its size and SHA-256 are always recomputed for the manifest.
"""
import csv
import hashlib
import os
import pathlib
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
# The Bureau of Labor Statistics refuses a download whose User-Agent holds no contact e-mail. Set the environment
# variable BLS_CONTACT_EMAIL to your own address (the authors' address is removed from this copy for blind review).
UA = "Mozilla/5.0 (academic research; " + os.environ.get("BLS_CONTACT_EMAIL", "") + ")"

FILES = [
    # BLS American Time Use Survey, multi-year files 2003-2025 (released June 25, 2026)
    ("atusresp-0325.zip", "https://www.bls.gov/tus/datafiles/atusresp-0325.zip"),
    ("atussum-0325.zip", "https://www.bls.gov/tus/datafiles/atussum-0325.zip"),
    ("atusrost-0325.zip", "https://www.bls.gov/tus/datafiles/atusrost-0325.zip"),
    ("atuscps-0325.zip", "https://www.bls.gov/tus/datafiles/atuscps-0325.zip"),
    ("atusact-0325.zip", "https://www.bls.gov/tus/datafiles/atusact-0325.zip"),
    ("atusintcodebk0325.pdf", "https://www.bls.gov/tus/dictionaries/atusintcodebk0325.pdf"),
    ("atussmcodebk0525.pdf", "https://www.bls.gov/tus/dictionaries/atussmcodebk0525.pdf"),
    ("atususersguide.pdf", "https://www.bls.gov/tus/atususersguide.pdf"),
    ("lexiconnoex0325.pdf", "https://www.bls.gov/tus/lexicons/lexiconnoex0325.pdf"),
    # Census occupation code lists with their crosswalks to SOC
    ("2018-occupation-code-list-and-crosswalk.xlsx",
     "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx"),
    ("2010-occ-codes-with-crosswalk-from-2002-2011.xls",
     "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2010-occ-codes-with-crosswalk-from-2002-2011.xls"),
    # BLS crosswalk from 2010 SOC to 2018 SOC
    ("soc_2010_to_2018_crosswalk.xlsx", "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx"),
    # Eloundou, Manning, Mishkin and Rock (2024, Science), exposure scores by O*NET-SOC occupation
    ("gpts_occ_level.csv", "https://raw.githubusercontent.com/openai/GPTs-are-GPTs/main/data/occ_level.csv"),
    # Felten, Raj and Seamans: AI occupational exposure (2021, Strategic Management Journal) and its
    # language-modeling version (2023), by SOC 2010 occupation
    ("AIOE_DataAppendix.xlsx", "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx"),
    ("Language_Modeling_AIOE_and_AIIE.xlsx",
     "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/Language%20Modeling%20AIOE%20and%20AIIE.xlsx"),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("BLS_CONTACT_EMAIL") and not all((RAW / name).exists() for name, _ in FILES):
        sys.exit("Set the environment variable BLS_CONTACT_EMAIL to your e-mail address: the Bureau of Labor "
                 "Statistics refuses a download whose User-Agent holds no contact address.")
    rows = []
    for name, url in FILES:
        dest = RAW / name
        if not dest.exists():
            print("downloading", url, flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
        rows.append({"file": name, "url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)})
        print(f"{name}: {dest.stat().st_size:,} bytes")
        if name.endswith(".zip"):
            folder = RAW / name[:-4]
            with zipfile.ZipFile(dest) as z:
                for member in z.namelist():
                    target = (folder / member).resolve()
                    if folder.resolve() not in target.parents:
                        raise SystemExit(f"refused: {name} holds a path outside its folder: {member}")
                    if not target.exists():
                        z.extract(member, folder)
    with open(ROOT / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "url", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("manifest written:", ROOT / "MANIFEST.csv")


if __name__ == "__main__":
    sys.exit(main())
