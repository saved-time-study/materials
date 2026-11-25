"""Rerun the whole ATUS analysis from nothing: download, build, exposure, estimates, figures, checks.

Run from the repository root:  python -I data/atus/scripts/run_all.py
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STEPS = ["test_fe.py", "01_download.py", "02_build_sample.py", "03_exposure.py", "04_analysis.py",
         "05_figures_tables.py", "06_checks.py", "07_jpxz_replication.py"]

for step in STEPS:
    print(f"== {step}", flush=True)
    r = subprocess.run([sys.executable, "-I", str(HERE / step)])
    if r.returncode != 0:
        sys.exit(f"{step} failed with exit code {r.returncode}")
print("all steps finished")
