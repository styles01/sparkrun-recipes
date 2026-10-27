---
name: read-codebase
description: Getting oriented in an unfamiliar repository before changing it - project rules files, entry points, build and test commands, the module map, data flow for one feature, and writing down what was learned.
---
# Reading an unfamiliar codebase

Rules:
- Read the rules first: `README`, `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING`, `Makefile`, `package.json` scripts, `pyproject.toml`. They say how to build and test and what is forbidden.
- Map the tree: `ls` top level, `find . -maxdepth 2 -type d | grep -v node_modules`, then file counts per language. Find the entry points (`main`, `app`, `index`, `cli`).
- Follow one feature end to end: route or command, handler, data model, storage, test. Read the test for it; tests show intent and the real call shapes.
- Run the existing tests before touching anything; note the summary line. That is the baseline.

What goes wrong:
- Rebuilding a helper that exists two directories up; grep for the concept first (`grep -rn "def parse_date\|parseDate"`).
- Changing a function with five callers and testing one.
- Running a different build or test command than the project's; "it fails" when it is just the wrong command.

Check: the rules file read (quote the relevant line), the test baseline run, the call-site list for the target symbol, and the 10-line note.

Done when:
- [ ] The build and test commands quoted from the project's own files, and the baseline test summary line.
- [ ] Call sites of the symbol to change listed with file:line.
- [ ] The orientation note (10 lines) in the message before the first edit.

## Reference

```bash
cat README.md | head -80; ls -a; cat AGENTS.md CLAUDE.md CONTRIBUTING.md 2>/dev/null | head -120
find . -maxdepth 2 -type d -not -path '*/node_modules*' -not -path '*/.git*' -not -path '*/.venv*'
find . -type f \( -name '*.py' -o -name '*.ts' -o -name '*.js' -o -name '*.go' \) -not -path '*/node_modules/*' -not -path '*/.venv/*' | sed 's/.*\.//' | sort | uniq -c
grep -rn "scripts" package.json | head; grep -n "\[tool\|\[project" pyproject.toml; grep -n '^[a-z-]*:' Makefile
grep -rn "app = \|createServer\|def main\|if __name__" --include='*.py' --include='*.js' --include='*.ts' . | grep -v node_modules | head
grep -rn "parse_date" --include='*.py' . | grep -v '.venv' | grep -v 'test_'      # call sites
git log --oneline -15; git log --oneline -- path/to/file | head        # recent history of the area
pytest -q 2>&1 | tail -3     # or: npm test 2>&1 | tail -5 ; go test ./... 2>&1 | tail -5
```
Orientation note template:
```
Build: <cmd>   Test: <cmd>   Run: <cmd>
Entry: <file>  Feature path: <route/cmd> -> <handler file:line> -> <model> -> <store>
Tests for it: <file>   Baseline: <N passed>
Change: <symbol> at <file:line>, callers: <n> (<files>)
Rules: <quoted line from README/AGENTS>
```

More rules:
- Grep for the symbol you will change and list every call site before editing.
- Write a 10-line orientation note (commands, entry points, where the feature lives, what the tests cover) in the message, then start.

More pitfalls:
- Reading 40 files and writing nothing down; the orientation evaporates after a compaction.
