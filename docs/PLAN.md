# Build plan (5 days)

Order: verifier -> Token Factory client -> intent compiler -> design proposer ->
repair loop -> API -> UI -> trap case -> benchmark -> README.

- **Day 1:** scaffold, schemas, verifier, ngspice runner + tests; Token Factory client; intent compiler.
- **Day 2:** inverter template with allowed ranges, design proposer, repair loop, run history.
- **Day 3:** FastAPI with streamed iteration events; UI: intent diff, live evidence table, latency/token panel.
- **Day 4:** waveform plot with delay marker, trap mode, benchmark script.
- **Day 5:** README, license check, secrets scan, demo script, buffer.

Run tests (from `backend/`): `python -m unittest discover -s tests -t . -v`
