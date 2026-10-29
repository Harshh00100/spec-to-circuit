# Benchmark: plain LLM answer vs Spec-to-Circuit

Date 2026-10-29. Model `llama3.2:3b`. ngspice Level-1 inverter. One run per spec.

| Spec | Baseline claimed | Baseline real delay | Baseline outcome | System verdict | System real delay | Attempts | System outcome |
|---|---|---|---|---|---|---|---|
| CMOS inverter, propagation delay under 40 ps at 20 fF load | met | 246.73 ps | false_claim | COMPLETE | 37.34 ps | 3 | verified_and_met |
| CMOS inverter, propagation delay under 1 ps at 20 fF load | met | 405.57 ps | false_claim | FAILED | 6.4 ps | 3 | failed_honestly |
| CMOS inverter, propagation delay under 0.1 ns driving 50 fF | met | 2220.54 ps | false_claim | COMPLETE | 87.45 ps | 2 | verified_and_met |

## Totals

- Specs run: 3
- Baseline: 0 design(s) actually met the spec; 3 time(s) it said the spec was met when the simulation says it was not; 0 unusable answer(s).
- System: 2 COMPLETE with a measured pass; 0 wrong COMPLETE; 1 FAILED correctly; 0 INCOMPLETE.

## How to read this

- The baseline's claim is scored against a real ngspice run of the sizes it chose; it never saw that result.
- The system's verdict comes from the verifier. "Wrong COMPLETE" means COMPLETE although the final simulated delay misses the limit; it should be 0.
- One run per spec on a small local model is an anecdote, not a statistic. Results change from run to run.
- The baseline gets the allowed size ranges but no feedback. Out-of-range sizes are counted as invalid, not clamped.
- Both sides use the same simulator, the same Level-1 models, and the stated load and supply.
