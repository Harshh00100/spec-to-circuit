# Spec-to-Circuit Agent

**"It won't say done without proof."**

Type a circuit requirement in plain English. The system turns it into measurable criteria, proposes a design, simulates it with ngspice, and reports **PASS**, **FAIL** or **UNVERIFIED** for each criterion. A language model suggests; deterministic Python decides. The model's words are never treated as evidence.

> Status: working prototype. It handles **one circuit type (a CMOS inverter)** with simplified Level-1 transistor models. See [Limits](#limits-read-this-first).

## How it works

```
requirement text
      |
      v
 1. Intent compiler   LLM drafts criteria loosely; Python validates every one.
      |               Anything it cannot measure is kept but marked UNVERIFIABLE.
      v
 2. Design proposer   LLM picks only transistor widths and lengths.
      |               The netlist comes from a fixed template; values are range-checked.
      v
 3. Simulator         ngspice in batch mode, with a timeout.
      |               Only the .measure results are used.
      v
 4. Verifier          Python compares each measurement with its limit.
      |               PASS / FAIL / UNVERIFIED, with the measured value and margin.
      v
 5. Repair loop       On FAIL, Python builds feedback from the measured numbers
                      and the model tries again (up to 4 attempts).
```

**Overall verdict** (from the verifier only):

| Verdict | When |
|---|---|
| COMPLETE | every criterion is PASS |
| FAILED | any criterion is FAIL |
| INCOMPLETE | nothing failed, but something could not be verified (or there was nothing to verify) |

A criterion is PASS or FAIL only if ngspice produced a finite measurement for exactly that metric with consistent units. Anything else is UNVERIFIED. A requirement like "low power" that cannot be expressed as a measurement is shown as unverifiable, never silently passed.

Other rules that keep the model honest:

- **Conditions are grounded.** A load or supply is applied only if it appears in your text. A value like "above 1.7 V" that is an output-level limit is not mistaken for a supply voltage.
- **Your words beat the model's label.** If you wrote "inverter" and the model calls the circuit something else, Python uses "inverter" and shows a note.
- **Unsupported requests are refused, not faked.** An op-amp or "make me a sandwich" ends INCOMPLETE with the reason, and nothing is simulated.
- **Unapplied conditions are not ignored.** If a stated condition cannot be put in the netlist, it becomes an unverifiable criterion, so the run cannot end COMPLETE on something that was never simulated.

### Trap mode

Tick **Trap mode** to see the idea in action. The model is not asked for a design. Python builds a deliberately too-slow inverter and attaches a confident false claim ("Expected tpd about 20 ps ... Spec met, status COMPLETE."). The design goes through the normal path (real netlist, real ngspice, real verifier). The verifier never reads the claim, so it reports FAIL with the measured number, shown next to the struck-through claim.

Trap mode needs a delay limit that a reasonable design can reach (at 20 fF, roughly 8 ps to 400 ps). Limits that no design can reach are refused, because a trap that fails for that reason proves nothing.

## What you see in the page

- A verdict stamp taken only from the verifier's report.
- **What I understood**: what will be measured, what cannot be verified and why, which conditions were applied or assumed, and what Python dropped.
- **Evidence**, one block per attempt: circuit drawing with W and L, the exact netlist sent to ngspice, a gauge for each criterion (limit tick and measured dot), margin, PASS / FAIL. The model's note is labelled "not evidence".
- A results table across attempts, a model-calls table (latency, tokens, retries), a history of past runs, and report downloads (Markdown and JSON).

## Requirements

- Python 3.11 or newer
- [ngspice](https://ngspice.sourceforge.io/) (tested with ngspice-47)
- A model behind an OpenAI-compatible chat endpoint. The default is local [Ollama](https://ollama.com) (no API key); tested with `llama3.2:3b`.
- Python packages: `pip install -r backend/requirements.txt` (pydantic, fastapi, uvicorn)

No Node.js and no build step: the web page is one HTML file served by the API.

## Setup and run (Windows PowerShell)

```powershell
# 1. get the model (one time)
ollama pull llama3.2:3b

# 2. install Python packages (one time)
cd spec-to-circuit\backend
pip install -r requirements.txt

# 3. tell the app where ngspice and the model are (per PowerShell window)
$env:NGSPICE_BIN = "C:\ngspice\Spice64\bin\ngspice_con.exe"
$env:LLM_MODEL   = "llama3.2:3b"

# 4. start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. API docs are at <http://127.0.0.1:8000/docs>.

Command-line runs, without the page:

```powershell
python -m app.run "CMOS inverter, propagation delay under 40 ps at 20 fF load"
```

### Using a different model

Any OpenAI-compatible endpoint works. Set these in the PowerShell window (never put a key in a file):

```powershell
$env:LLM_BASE_URL = "https://your-provider/v1"
$env:LLM_API_KEY  = "your key"
$env:LLM_MODEL    = "model-id"
python -m app.llm models      # list the model ids the endpoint offers
```

See `.env.example` for all settings. Hosted providers such as Nebius Token Factory use this same interface, but **this project has only been run against local Ollama**. No hosted model has been tried.

### Run the tests

```powershell
cd spec-to-circuit\backend
$env:NGSPICE_BIN = "C:\ngspice\Spice64\bin\ngspice_con.exe"
python -m unittest discover -s tests -t .
```

226 tests pass. Without `NGSPICE_BIN`, the integration tests that need real ngspice are skipped. The unit tests use a scripted fake model (`FakeLLM`) and never need Ollama. `FakeLLM` is a test fixture, not a model, and passing tests with it show that the rules work, not that any real model is good.

## Demo script (about 5 minutes)

1. **A pass.** `CMOS inverter, propagation delay under 2 ns at 20 fF load`. One attempt, COMPLETE, with the measured delay shown.
2. **The repair loop.** `CMOS inverter, propagation delay under 40 ps at 20 fF load`. Attempts 1 and 2 FAIL with measured numbers and margins; the model widens the transistors; attempt 3 PASSES. In our recorded run: 71.7 ps, 46.1 ps, then 38.4 ps.
3. **Honesty about the unmeasurable.** `CMOS inverter, propagation delay under 2 ns, and it should be low power`. INCOMPLETE: the delay passes, "low power" is shown as unverifiable.
4. **A refusal.** `Two-stage op-amp with 60 dB gain`. INCOMPLETE, no design template, nothing simulated.
5. **Trap mode.** Same 40 ps requirement with Trap mode ticked. The false claim is shown next to the ngspice measurement, and the verdict is FAILED ("Caught").

Timing on a CPU-only PC with a 3B model: each model call takes roughly 15 to 90 seconds, so a run can take from one to several minutes. Close other programs first; the model needs about 2 GB of free memory.

## Benchmark: plain model versus this system

`python -m benchmark.run_benchmark` runs each spec twice on the same model. The **plain model** is asked once for transistor sizes and whether its design meets the spec; no tools, no feedback. Its claim is then checked by a real ngspice run of the sizes it chose. The **system** is the normal pipeline. Results from one run on `llama3.2:3b` (29 Oct 2026):

| Spec | Plain model said | Its real delay | System verdict | System real delay | Attempts |
|---|---|---|---|---|---|
| tpd < 40 ps at 20 fF | met | 246.73 ps | COMPLETE | 37.34 ps | 3 |
| tpd < 1 ps at 20 fF (impossible) | met | 405.57 ps | FAILED | 6.4 ps | 3 |
| tpd < 0.1 ns driving 50 fF | met | 2220.54 ps | COMPLETE | 87.45 ps | 2 |

The plain model said "met" three times out of three, and none of its designs met the spec (its real delays were about 6, 22 and 405 times over the limit). The system produced two designs with a measured pass and reported the impossible spec as FAILED. A cross-check found no COMPLETE that missed its limit.

Read this carefully: 3 specs, one run each, one small model and a simple baseline. The comparison is not purely "verifier versus no verifier". The system's design prompt also gives sizing hints, and after a FAIL it feeds back measured numbers built by Python; the plain model got the allowed size ranges but no hints and no feedback. Part of the gap comes from that help. A fairer follow-up would give the plain model the same hints and the same number of attempts with feedback. A stronger model or a better-engineered baseline would likely do better, and results vary from run to run. Raw numbers are saved in `backend/benchmark/results/`. Two more specs are available with `--limit 5` and have not been run.

## Project layout

```
backend/app/
  schemas.py     data models for every stage (unknown fields are rejected)
  units.py       unit conversion to SI
  verifier.py    deterministic PASS / FAIL / UNVERIFIED
  simulator.py   runs ngspice, parses .measure results, blocks .control / shell / .include
  netlists.py    the inverter template and its measurements
  llm.py         provider-agnostic client (stdlib urllib) and FakeLLM
  intent.py      requirement text -> criteria
  design.py      criteria -> transistor sizes -> netlist
  loop.py        the repair loop and its event stream
  trap.py        planted-claim trap mode
  runs.py        background runs, live events, saved history
  main.py        FastAPI endpoints
  run.py         command-line runner
backend/benchmark/   plain-model-versus-system benchmark and its saved results
backend/tests/   unit and integration tests
frontend/index.html   the web page
```

API: `GET /api/health`, `POST /api/runs` (`{"text": "...", "trap": false}`), `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/events` (Server-Sent Events).

## Limits (read this first)

- **One circuit type.** Only a CMOS inverter has a template. Other circuits are refused, not approximated. Measurable metrics: propagation delay (`tpd`, `tphl`, `tplh`) and output levels (`voh`, `vol`).
- **Simplified models.** Transistors use Level-1 models, not a real process design kit. A PASS means "passes in this simulation" at an assumed 1.8 V and 20 fF unless your requirement says otherwise. It is not sign-off.
- **Small local model.** Every live result here came from `llama3.2:3b` on a CPU. Expect slow calls, some poor first drafts and occasional retries. That is why Python validates and decides everything.
- **Benchmark is small.** 3 specs, one run each, one 3B model, one simple baseline prompt (see below). It is an anecdote, not a statistic.
- **Hosted models untested.** The client is built for OpenAI-compatible endpoints, but only local Ollama has been run.
- **History is partial.** A reopened past run shows its report, but not the netlist or the load and supply labels on the diagram (those exist only in live events).
- **Trap mode** covers delay limits only (`<`, `<=`), and stops after one attempt.
- **Web fonts need internet.** The page loads Space Grotesk, IBM Plex Sans and JetBrains Mono from Google Fonts, so your browser makes a request to Google when you open it. Offline, it falls back to system fonts and works the same.
- **Single user, local only.** One run at a time, no login. Keep it bound to `127.0.0.1`; do not expose it to a network.
- **Page tested lightly:** the page was tested in headless Chromium against recorded events and on one PC. The circuit drawing and results table have had less testing than the rest.

## Security notes

- The simulator rejects `.control`, `shell` and `.include` directives, so a model (or a user) cannot make ngspice run commands.
- Text from the user or the model is inserted into the page as plain text, never as HTML.
- Secrets belong only in environment variables. `.env`, run history, zip files and `.bak` files are in `.gitignore`.

## License

MIT. See `LICENSE`.
