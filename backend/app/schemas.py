"""Pydantic schemas for every stage of the pipeline.

LLM outputs (IntentSpec, DesignProposal) are validated against these.
Verdicts and reports are produced only by deterministic Python.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Operator = Literal["<", "<=", ">", ">="]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNVERIFIED = "UNVERIFIED"


class OverallStatus(str, Enum):
    COMPLETE = "COMPLETE"      # every criterion PASS, backed by simulator evidence
    FAILED = "FAILED"          # at least one criterion FAIL
    INCOMPLETE = "INCOMPLETE"  # no FAIL, but something is UNVERIFIED (or nothing to check)


# --------------------------------------------------------------------------
# Stage 1: intent
# --------------------------------------------------------------------------
class Criterion(StrictModel):
    name: str = Field(min_length=1)
    metric: str = ""
    operator: Operator | None = None
    threshold: float | None = None
    unit: str = ""
    measurable: bool
    unverifiable_reason: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> "Criterion":
        if self.measurable:
            missing = [f for f in ("operator", "threshold") if getattr(self, f) is None]
            if missing or not self.metric.strip():
                raise ValueError(
                    "a measurable criterion needs metric, operator and threshold "
                    f"(missing: {missing or ['metric']})"
                )
            if not math.isfinite(self.threshold):  # type: ignore[arg-type]
                raise ValueError("threshold must be a finite number")
        elif not self.unverifiable_reason:
            self.unverifiable_reason = "not expressible as a simulator measurement"
        return self


class ClarifyingQuestion(StrictModel):
    question: str = Field(min_length=1)
    reason: str = ""


class IntentSpec(StrictModel):
    circuit_type: str = Field(min_length=1)
    parameters: dict[str, float | str] = Field(default_factory=dict)
    criteria: list[Criterion] = Field(default_factory=list)
    understood_summary: str = ""
    questions: list[ClarifyingQuestion] = Field(default_factory=list, max_length=2)


# --------------------------------------------------------------------------
# Stage 2: design
# --------------------------------------------------------------------------
class TransistorSize(StrictModel):
    w_um: float = Field(gt=0)
    l_um: float = Field(gt=0)


class DesignProposal(StrictModel):
    circuit_type: str
    stage_count: int = Field(ge=1)
    transistors: dict[str, TransistorSize]
    rationale: str = ""


# --------------------------------------------------------------------------
# Stage 3: simulation
# --------------------------------------------------------------------------
class SimulationResult(StrictModel):
    ok: bool
    measurements: dict[str, float] = Field(default_factory=dict)  # SI units, lowercase names
    failed_measures: list[str] = Field(default_factory=list)
    returncode: int | None = None
    runtime_s: float = 0.0
    error: str | None = None
    log_tail: str = ""


# --------------------------------------------------------------------------
# Stage 4: verification
# --------------------------------------------------------------------------
class CriterionVerdict(StrictModel):
    criterion: Criterion
    verdict: Verdict
    measured: float | None = None      # in the criterion's own unit
    margin: float | None = None        # in the criterion's own unit; > 0 means headroom
    margin_pct: float | None = None    # margin relative to |threshold|
    reason: str = ""


class VerificationReport(StrictModel):
    verdicts: list[CriterionVerdict]
    overall: OverallStatus
    counts: dict[str, int]
    evidence_ok: bool  # did the simulator actually produce results?

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved(self) -> list[str]:
        return [v.criterion.name for v in self.verdicts if v.verdict != Verdict.PASS]


# --------------------------------------------------------------------------
# Loop / reporting
# --------------------------------------------------------------------------
class LLMCallRecord(StrictModel):
    purpose: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retries: int = 0
    ok: bool = True
    error: str | None = None


class IterationRecord(StrictModel):
    index: int
    design: DesignProposal | None = None
    simulation: SimulationResult
    report: VerificationReport


class RunReport(StrictModel):
    run_id: str
    spec_text: str
    intent: IntentSpec | None = None
    iterations: list[IterationRecord] = Field(default_factory=list)
    final_report: VerificationReport | None = None
    status: OverallStatus = OverallStatus.INCOMPLETE
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
