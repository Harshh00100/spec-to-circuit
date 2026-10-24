"""Runs the validated inverter template through REAL ngspice for a light and a heavy load.

    python check_template.py

Needs NGSPICE_BIN (or ngspice on PATH). Prints the measurements ngspice reports. It
exists because the time-window scaling for heavy loads can only be proven by a real run.
Good result: tphl, tplh, tpd, voh, vol all show finite numbers for BOTH loads.
Bad result: any line saying 'failed' (the edge fell outside the simulated window).
"""
import os
import shutil
import subprocess
import sys
import tempfile

from app.netlists import inverter_netlist
from app.schemas import DesignProposal, TransistorSize

binary = os.environ.get("NGSPICE_BIN") or shutil.which("ngspice") or shutil.which("ngspice_con")
if not binary:
    sys.exit("ngspice not found: set NGSPICE_BIN to the ngspice_con.exe path")

design = DesignProposal(
    circuit_type="inverter", stage_count=1,
    transistors={"mn": TransistorSize(w_um=0.5, l_um=0.18), "mp": TransistorSize(w_um=1.0, l_um=0.18)},
)
for label, cload_ff in (("20 fF (default)", 20.0), ("10 pF (heavy)", 10000.0)):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "t.cir")
        with open(path, "w") as handle:
            handle.write(inverter_netlist(design, cload_ff=cload_ff))
        run = subprocess.run([binary, "-b", path], capture_output=True, text=True, timeout=120)
    print(f"=== load {label} (exit code {run.returncode})")
    shown = [ln for ln in (run.stdout + run.stderr).splitlines()
             if any(k in ln.lower() for k in ("tphl", "tplh", "tpd", "voh", "vol", "fail", "error"))]
    print("\n".join(shown) if shown else "(no measurement lines found)")
