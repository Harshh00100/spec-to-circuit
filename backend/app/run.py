"""Run the full pipeline once: python -m app.run "your requirement"

Needs NGSPICE_BIN and LLM_MODEL set (see PROGRESS.md). Prints each attempt as it happens.
"""
from __future__ import annotations

import sys

from .loop import run_spec
from .simulator import run_ngspice
from .verifier import verify


def show(event: dict) -> None:
    kind = event.get("type")
    if kind == "intent":
        crit = [f'{c["metric"]} {c["operator"]} {c["threshold"]} {c["unit"]}'
                for c in event["spec"]["criteria"] if c.get("measurable")]
        print(f"[intent] measurable criteria: {crit}")
    elif kind == "conditions":
        print(f"[conditions] applied={event['applied']} assumed={event['assumed']} unapplied={event['unapplied']}")
    elif kind == "design":
        print(f"[design {event['index']}] {event['design']}")
    elif kind == "iteration":
        print(f"[result {event['index']}] {event['overall']}")
        for v in event["report"]["verdicts"]:
            c = v["criterion"]
            label = f"{c['metric']} {c['operator']} {c['threshold']} {c['unit']}" if c.get("metric") else c.get("name")
            print(f"    {label}: "
                  f"{v['verdict']}  measured={v.get('measured')}  margin={v.get('margin')}  {v.get('reason') or ''}")
    elif kind == "error":
        print(f"[error] {event}")
    elif kind == "done":
        print(f"[done] status={event['status']} after {event['iterations']} iteration(s)")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.run "CMOS inverter, propagation delay under 2 ns"')
    report = run_spec(sys.argv[1], simulate=run_ngspice, verify=verify, on_event=show)
    calls = report.llm_calls
    print(f"\nFINAL: {report.status.value}   model calls: {len(calls)}   "
          f"total model time: {sum(c.latency_s for c in calls):.0f}s")


if __name__ == "__main__":
    main()
