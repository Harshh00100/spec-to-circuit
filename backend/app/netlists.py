"""Netlist builders.

* hardcoded_inverter_netlist: the original demo netlist (unchanged; Step 1 tests use it).
* inverter_netlist: the validated template. It takes a DesignProposal plus the user's
  conditions (supply, load), re-checks every number against the allowed ranges below,
  and scales the simulation time base with the expected delay so a heavy load does
  not push the edge outside the simulated window. Only finite numbers ever reach the
  netlist text, so a model can never inject SPICE directives through it.
"""
from __future__ import annotations

import math

from .schemas import DesignProposal

INVERTER_MEASURES = ("tphl", "tplh", "tpd", "voh", "vol")

# Allowed ranges (single source of truth; design.py clamps to these, the builder enforces them).
W_RANGE_UM = (0.18, 50.0)
L_RANGE_UM = (0.18, 2.0)
VDD_RANGE_V = (1.0, 3.3)
CLOAD_RANGE_FF = (1.0, 100_000.0)  # 1 fF to 100 pF
DEFAULT_VDD = 1.8
DEFAULT_CLOAD_FF = 20.0

# Level-1 model constants used by the delay estimate (must match the .model lines below).
_KP_N, _KP_P, _VT = 200e-6, 80e-6, 0.5
_BASE_DELAY_S = 100e-12  # at or below this estimate the original 10 ns window is kept


def hardcoded_inverter_netlist(
    wn_um: float = 0.5,
    wp_um: float = 1.0,
    l_um: float = 0.18,
    cload_ff: float = 20.0,
    vdd: float = 1.8,
) -> str:
    half = vdd / 2
    return f"""* CMOS inverter - hardcoded demo netlist (Level-1 models)
.model nmos_l1 nmos (level=1 vto=0.5 kp=200u lambda=0.05)
.model pmos_l1 pmos (level=1 vto=-0.5 kp=80u lambda=0.05)
vsupply vdd 0 {vdd:g}
vin in 0 pulse(0 {vdd:g} 0.5n 0.1n 0.1n 2n 5n)
mp out in vdd vdd pmos_l1 w={wp_um:g}u l={l_um:g}u
mn out in 0 0 nmos_l1 w={wn_um:g}u l={l_um:g}u
cl out 0 {cload_ff:g}f
.tran 5p 10n
.meas tran tphl trig v(in) val={half:g} rise=1 targ v(out) val={half:g} fall=1
.meas tran tplh trig v(in) val={half:g} fall=1 targ v(out) val={half:g} rise=1
.meas tran tpd param='(tphl+tplh)/2'
.meas tran voh max v(out)
.meas tran vol min v(out)
.end
"""


def _check(label: str, value: object, bounds: tuple[float, float]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{label}={value:g} is outside the allowed range {low:g} to {high:g}")
    return float(value)


def estimate_delay_s(wn_um: float, ln_um: float, wp_um: float, lp_um: float,
                     cload_ff: float, vdd: float) -> float:
    """Rough saturation-current estimate of the 50% delay, used only to size the time window."""
    overdrive = vdd - _VT
    i_n = 0.5 * _KP_N * (wn_um / ln_um) * overdrive**2
    i_p = 0.5 * _KP_P * (wp_um / lp_um) * overdrive**2
    return cload_ff * 1e-15 * (vdd / 2) / min(i_n, i_p)


def inverter_netlist(
    design: DesignProposal, vdd: float = DEFAULT_VDD, cload_ff: float = DEFAULT_CLOAD_FF
) -> str:
    """Build the inverter netlist from a validated design. Raises ValueError on anything off-template."""
    if design.circuit_type != "inverter" or design.stage_count != 1:
        raise ValueError("the inverter template needs circuit_type 'inverter' and stage_count 1")
    if set(design.transistors) != {"mn", "mp"}:
        raise ValueError("the inverter template needs exactly the transistors 'mn' and 'mp'")
    mn, mp = design.transistors["mn"], design.transistors["mp"]
    wn = _check("mn.w_um", mn.w_um, W_RANGE_UM)
    ln = _check("mn.l_um", mn.l_um, L_RANGE_UM)
    wp = _check("mp.w_um", mp.w_um, W_RANGE_UM)
    lp = _check("mp.l_um", mp.l_um, L_RANGE_UM)
    vdd = _check("vdd", vdd, VDD_RANGE_V)
    cload_ff = _check("cload_ff", cload_ff, CLOAD_RANGE_FF)

    estimate = estimate_delay_s(wn, ln, wp, lp, cload_ff, vdd)
    scale = max(1.0, estimate / _BASE_DELAY_S)

    def t(value: float, unit: str) -> str:
        return f"{value * scale:.6g}{unit}"

    half = vdd / 2
    return f"""* CMOS inverter - validated template (Level-1 models, time scale x{scale:.4g})
.model nmos_l1 nmos (level=1 vto=0.5 kp=200u lambda=0.05)
.model pmos_l1 pmos (level=1 vto=-0.5 kp=80u lambda=0.05)
vsupply vdd 0 {vdd:g}
vin in 0 pulse(0 {vdd:g} {t(0.5, 'n')} {t(0.1, 'n')} {t(0.1, 'n')} {t(2, 'n')} {t(5, 'n')})
mp out in vdd vdd pmos_l1 w={wp:g}u l={lp:g}u
mn out in 0 0 nmos_l1 w={wn:g}u l={ln:g}u
cl out 0 {cload_ff:g}f
.tran {t(5, 'p')} {t(10, 'n')}
.meas tran tphl trig v(in) val={half:g} rise=1 targ v(out) val={half:g} fall=1
.meas tran tplh trig v(in) val={half:g} fall=1 targ v(out) val={half:g} rise=1
.meas tran tpd param='(tphl+tplh)/2'
.meas tran voh max v(out)
.meas tran vol min v(out)
.end
"""
