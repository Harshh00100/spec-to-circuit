"""Stand-in for ngspice, used ONLY to test the runner plumbing (timeout, exit
codes, parsing, log-file reading) on machines without ngspice.

It mimics the *assumed* shape of ngspice batch output. It proves nothing about
real circuit behaviour -- that is what test_inverter_integration.py is for.

Mode comes from FAKE_NGSPICE_MODE: ok | fail_measure | log_only | sleep | crash | silent
"""
import os
import re
import sys
import time
from pathlib import Path

VALUES = {"tphl": 1.0e-10, "tplh": 1.2e-10, "tpd": 1.1e-10, "voh": 1.8, "vol": 0.0}


def main(argv):
    mode = os.environ.get("FAKE_NGSPICE_MODE", "ok")
    log_path = argv[argv.index("-o") + 1] if "-o" in argv else None
    netlist = Path(argv[-1]).read_text()
    names = re.findall(r"^\s*\.meas\s+\w+\s+(\w+)", netlist, re.I | re.M)

    if mode == "sleep":
        time.sleep(60)
    if mode == "crash":
        print("Error: simulation aborted", file=sys.stderr)
        return 3

    lines = ["Circuit: fake", "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000", ""]
    if mode != "silent":
        for name in names:
            if mode == "fail_measure" and name.lower() == "tphl":
                lines.append(f"{name:<20}=  failed")
            else:
                lines.append(f"{name:<20}=  {VALUES.get(name.lower(), 1.0):e} targ=  6.1e-10 trig=  5.1e-10")
        lines.append("Note: unrelated_setting = 42")
    text = "\n".join(lines) + "\n"

    if mode == "log_only" and log_path:
        Path(log_path).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
