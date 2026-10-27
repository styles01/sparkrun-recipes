---
name: refactor-safely
description: Changing the structure of existing code without changing its behavior - characterization tests first, small verified steps, renames and extractions, keeping the public API, and proving nothing changed.
---
# Refactoring safely

Rules:
- Behavior stays the same. If the task also changes behavior, do the refactor first in its own commit, then the behavior change.
- Before touching anything: run the tests (baseline), and if the area has none, write a characterization test that records the current outputs for 5 to 10 real inputs.
- One mechanical step at a time: rename, extract function, move file, inline, replace a conditional. Run the tests after each step. Commit each step.
- Keep the public API: names that other files or users import stay, or get a deprecation shim that forwards.

What goes wrong:
- A rename missed in a string (a route, a config key, a template) that no type checker sees.
- The refactor and a bug fix mixed in one diff; the reviewer cannot tell which line changed behavior.
- Extracting a function that captured a variable by closure and now reads a stale value.

Check: the same test suite passes before and after with the same count, the characterization outputs are byte-identical, and the diff shows only structure (no new conditions, no changed constants).

Done when:
- [ ] Test summary line before and after with equal counts and 0 failed.
- [ ] Characterization outputs (or a checksum of them) equal before and after.
- [ ] `git diff --stat` and a one-line list of the mechanical steps taken.

## Reference

Characterization test:
```python
import json, pytest
from mymod import process
CASES = ["", "a", "a,b", "x;y", "ünï", "1,2,3,4,5", "  spaced  ", "a,,b", "\n", "0"]
@pytest.mark.parametrize("inp", CASES)
def test_characterize(inp, snapshot_path=__import__("pathlib").Path("tests/snap_process.json")):
    snaps = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    out = repr(process(inp))
    if inp not in snaps:
        snaps[inp] = out; snapshot_path.write_text(json.dumps(snaps, indent=1, ensure_ascii=False))
    assert snaps[inp] == out
```
Run once to record, then it guards. Delete the file only when a behavior change is intended.

Rename across the repo (check first, then apply):
```bash
grep -rn "old_name" --include='*.py' --include='*.html' --include='*.yaml' --include='*.md' . | grep -v .venv
grep -rl "old_name" --include='*.py' . | grep -v .venv | xargs sed -i.bak 's/\bold_name\b/new_name/g' && find . -name '*.bak' -delete
pytest -q
```
Deprecation shim: `def old_name(*a, **k): warnings.warn("old_name is deprecated; use new_name", DeprecationWarning, 2); return new_name(*a, **k)`.
Extract function: copy the block into a function with explicit parameters for every variable it reads and a return for every one it writes; replace the block with the call; run; then simplify.
Order of steps: rename, extract, move, delete dead code; one commit each, `git log --oneline` as the record.

More rules:
- Use the tool for renames when one exists (IDE, `rope`, `ts-morph`, `gofmt -r`); otherwise grep every call site and change them all in one step.
- Stop when the goal is met. Do not "clean up" neighboring code that nobody asked about.

More pitfalls:
- Tests rewritten "to match" the new code; they then test nothing.
- Behavior changed in an edge (None handling, order of a dict) and no test covered it.
