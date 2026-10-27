---
name: debug-method
description: A method for fixing a bug when the cause is unknown - reproduce with the smallest input, read the state before the symptom, bisect by halves (input, code, commits), name one cause, change one thing, re-run the same reproduction.
---
# Debugging method

Rules:
1. Reproduce before anything: the exact command or input that fails, written as one line, with the observed and the expected output. No reproduction, no fix.
2. Shrink: halve the input until the failure stays with the smallest case (one row, one key press, one value).
3. Read the state just before the symptom: a print, a log line, a debugger break, the `__state()` dump. Do not guess from the code.
4. Bisect when the cause is far from the symptom: by code (comment out or stub half), by data (half the rows), or by history (`git bisect` between a good and a bad commit).
5. Name the one cause in one sentence. Then change one thing. Re-run the exact reproduction, then the full tests.

What goes wrong:
- Fixing the symptom (a `try/except`, a default value) and the cause stays.
- Changing three things at once; the bug disappears and nobody knows why, then returns.
- Evidence from before the last edit used as proof; always re-run after the fix.

Check: the reproduction command fails before the fix and passes after, in the same session, both outputs in the result; the full suite still passes.

Done when:
- [ ] The one-line reproduction with its failing output before the change.
- [ ] The one-sentence cause and the single change (file:line).
- [ ] The same reproduction passing after, plus the full test summary line.

## Reference

Reproduction line: `python -m mytool.cli samples/bad.csv  # expected 3 rows, got "KeyError: 'amount'"`.

Shrink a file: `head -n 2 big.csv > /tmp/small.csv` (header plus one row); then one column.

State before the symptom:
```python
import logging; log = logging.getLogger(__name__)
log.warning("before parse: row=%r keys=%s", row, list(row))    # one line, remove after
breakpoint()   # pdb: p row, pp locals(), n, c
```
Browser: `evaluate_script window.__state()` after a real input; `debugger;` with DevTools open; `console.table(entities)`.

Bisect history:
```bash
git bisect start; git bisect bad; git bisect good v1.2.0
git bisect run pytest -q tests/test_parse.py -x     # or a script that exits non-zero on failure
git bisect reset
```
Bisect code: stub half the pipeline (`return input` at a midpoint) and see on which side the symptom stays.

Common causes to check in one minute each: a stale build or cached `.pyc`/`dist`; a different environment (`which python`, `node -v`); an off-by-one at a boundary (empty, 0, last element); shared mutable state (a default `[]`, a module-level dict); an unawaited promise or coroutine; a time zone or encoding; a race between two async tasks (add an ordering log line).

Write-up format: `Repro: <cmd> -> <wrong>. Cause: <sentence>. Fix: <file:line> <one change>. Proof: <cmd> -> <right>; suite N passed.`

More rules:
- After two failed fixes on one bug: stop, re-read the function from the top, write the bug down again, and question the reproduction.

More pitfalls:
- Trusting an error message's line number from a stale build or a cached module.
- Reproducing in a different environment (prod data, another Python) than the one that fails.
