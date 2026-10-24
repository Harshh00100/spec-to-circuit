"""Unit handling for the verifier.

ngspice reports everything in SI base units (seconds, volts, amps, ...).
Criterion thresholds arrive in whatever unit the spec used ("ns", "mV").
This module converts both sides to SI so comparisons are exact, and so a
unit mistake (e.g. a voltage threshold on a delay metric) is caught instead
of silently compared.
"""
from __future__ import annotations


class UnitError(ValueError):
    """Raised for a unit string we do not recognise."""


_PREFIXES: dict[str, float] = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
    "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9,
}

_BASE_UNITS: dict[str, str] = {
    "s": "time",
    "V": "voltage",
    "A": "current",
    "W": "power",
    "F": "capacitance",
    "Hz": "frequency",
    "ohm": "resistance",
    "Ω": "resistance",
}


def _build_table() -> dict[str, tuple[str, float]]:
    table: dict[str, tuple[str, float]] = {}
    for base, dimension in _BASE_UNITS.items():
        for prefix, factor in _PREFIXES.items():
            table[prefix + base] = (dimension, factor)
    table[""] = ("ratio", 1.0)
    table["x"] = ("ratio", 1.0)
    table["%"] = ("ratio", 0.01)
    table["dB"] = ("db", 1.0)
    return table


_UNIT_TABLE = _build_table()


def unit_info(unit: str) -> tuple[str, float]:
    """Return (dimension, factor_to_SI) for a unit string, e.g. "ns" -> ("time", 1e-9)."""
    try:
        return _UNIT_TABLE[unit.strip()]
    except KeyError:
        raise UnitError(f"unknown unit {unit!r}") from None


def to_si(value: float, unit: str) -> float:
    _, factor = unit_info(unit)
    return value * factor


# Which physical dimension each simulator metric has. A criterion whose unit
# has a different dimension than its metric is reported UNVERIFIED, not compared.
METRIC_DIMENSIONS: dict[str, str] = {
    "tphl": "time",
    "tplh": "time",
    "tpd": "time",
    "voh": "voltage",
    "vol": "voltage",
    "avg_power": "power",
}


def metric_dimension(metric: str) -> str | None:
    return METRIC_DIMENSIONS.get(metric.strip().lower())
