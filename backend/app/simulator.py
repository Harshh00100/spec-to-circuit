"""Run ngspice in batch mode and collect .measure results.

The simulator is the only source of evidence in the system. If it times out,
crashes, or a measurement cannot be evaluated, that is reported honestly and
the verifier turns it into UNVERIFIED -- never into PASS.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

from .schemas import SimulationResult

DEFAULT_TIMEOUT_S = 20.0

_MEAS_DECL = re.compile(
    r"^\s*\.meas(?:ure)?\s+(?:tran|ac|dc|op|tf|noise|sp)\s+(\w+)", re.IGNORECASE | re.MULTILINE
)
# ngspice prints e.g. "tpd                 =  1.100000e-10 targ= ... trig= ..."
_RESULT_LINE = re.compile(r"^\s*(\w+)\s*=\s*(\S+)", re.MULTILINE)
# Netlists come from our templates, but ngspice can run shell commands from
# .control blocks, so refuse anything that could do that or pull in files.
_FORBIDDEN = re.compile(
    r"^\s*(\.control|\.endc|\.inc(?:lude)?|\.lib|shell|system)\b", re.IGNORECASE | re.MULTILINE
)


def declared_measures(netlist: str) -> list[str]:
    seen: list[str] = []
    for match in _MEAS_DECL.finditer(netlist):
        name = match.group(1).lower()
        if name not in seen:
            seen.append(name)
    return seen


def parse_measurements(text: str, declared: Sequence[str]) -> tuple[dict[str, float], list[str]]:
    """Extract declared .measure results. Returns (values, failed_names).

    Only names declared in the netlist are accepted, so unrelated "x = y" lines
    in ngspice's output can never be mistaken for measurements. A declared
    measure that is missing, non-numeric ("failed") or non-finite is "failed".
    """
    wanted = {d.lower() for d in declared}
    values: dict[str, float] = {}
    failed: set[str] = set()
    for match in _RESULT_LINE.finditer(text):
        name = match.group(1).lower()
        if name not in wanted or name in values or name in failed:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            failed.add(name)
            continue
        if math.isfinite(value):
            values[name] = value
        else:
            failed.add(name)
    failed |= wanted - set(values)
    return values, sorted(failed)


def _resolve_cmd(ngspice_bin: str | Sequence[str] | None) -> list[str] | None:
    if ngspice_bin is None:
        ngspice_bin = os.environ.get("NGSPICE_BIN") or shutil.which("ngspice")
    if not ngspice_bin:
        return None
    if isinstance(ngspice_bin, str):
        return [ngspice_bin]
    return list(ngspice_bin)


def run_ngspice(
    netlist: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    ngspice_bin: str | Sequence[str] | None = None,
) -> SimulationResult:
    started = time.perf_counter()

    def finish(**kwargs) -> SimulationResult:
        return SimulationResult(runtime_s=time.perf_counter() - started, **kwargs)

    forbidden = _FORBIDDEN.search(netlist)
    if forbidden:
        return finish(ok=False, error=f"netlist rejected: forbidden directive {forbidden.group(1)!r}")

    declared = declared_measures(netlist)
    if not declared:
        return finish(ok=False, error="netlist declares no .measure statements")

    cmd = _resolve_cmd(ngspice_bin)
    if cmd is None:
        return finish(
            ok=False, error="ngspice not found (install it or set NGSPICE_BIN)"
        )

    with tempfile.TemporaryDirectory(prefix="spec2circuit_") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "circuit.cir").write_text(netlist)
        try:
            proc = subprocess.run(
                [*cmd, "-b", "-o", "ngspice.log", "circuit.cir"],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return finish(ok=False, error=f"ngspice timed out after {timeout_s:g}s")
        except OSError as exc:
            return finish(ok=False, error=f"could not start ngspice: {exc}")
        log_path = tmp / "ngspice.log"
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""

    console = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_tail = "\n".join(console.strip().splitlines()[-15:])
    values, failed = parse_measurements((proc.stdout or "") + "\n" + log_text, declared)

    if proc.returncode != 0:
        return finish(
            ok=False,
            measurements=values,
            failed_measures=failed,
            returncode=proc.returncode,
            error=f"ngspice exited with code {proc.returncode}",
            log_tail=log_tail,
        )
    if not values:
        return finish(
            ok=False,
            failed_measures=failed,
            returncode=proc.returncode,
            error="ngspice produced no .measure results",
            log_tail=log_tail,
        )
    return finish(
        ok=True,
        measurements=values,
        failed_measures=failed,
        returncode=proc.returncode,
        log_tail=log_tail,
    )
