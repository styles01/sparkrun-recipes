---
name: python-cli
description: Writing a Python command-line tool with argparse or typer - subcommands, flags, stdin and stdout, exit codes, error messages, --dump-state, and a main() that tests can call.
---
# Python CLI

Rules:
- `main(argv=None) -> int` returns the exit code; `if __name__ == "__main__": sys.exit(main())`. Tests call `main([...])` directly.
- argparse for no dependencies, typer when the project already has it. Subcommands for more than one action. Every option has `help=`.
- Data to stdout, messages and errors to stderr. `--json` for machine output. Exit 0 on success, 1 on a user error, 2 on bad arguments (argparse does this).
- Read from a file argument or `-` for stdin. Write to `--output` or stdout. Never print inside library functions; return values.

What goes wrong:
- Business logic inside `main`; untestable. Put it in functions that take and return values.
- Printing progress to stdout breaks piping; progress goes to stderr.
- `BrokenPipeError` when piped into `head`; catch it and exit 0.

Check: run on a real input file with known content, compare the output number by number; run the error paths (missing file, bad flag, empty stdin) and read the exit codes with `echo $?`.

Done when:
- [ ] A run on real input with its output in the result and one value checked by hand.
- [ ] Missing file, bad argument, empty input: exit codes 1, 2, 1 (or as designed) shown.
- [ ] `--dump-state` output and `--help` output in the result.

## Reference

```python
#!/usr/bin/env python3
"""count: count lines and words in files."""
import argparse, json, sys
from pathlib import Path
__version__ = "0.1.0"
def count(text: str) -> dict:
    return {"lines": text.count("\n"), "words": len(text.split())}
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="count", description=__doc__)
    p.add_argument("path", help="input file, or - for stdin")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("--dump-state", action="store_true", help="print internal state and exit")
    p.add_argument("--version", action="version", version=__version__)
    return p
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dump_state:
        print(json.dumps({"tandem_state": 1, "version": __version__, "args": vars(args)})); return 0
    try:
        text = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"error: file not found: {args.path}", file=sys.stderr); return 1
    if not text.strip():
        print("error: empty input", file=sys.stderr); return 1
    r = count(text)
    try:
        print(json.dumps(r) if args.json else f"{r['lines']} {r['words']}")
    except BrokenPipeError:
        return 0
    return 0
if __name__ == "__main__":
    sys.exit(main())
```
Probe:
```bash
printf 'a b\nc\n' > /tmp/in.txt; python count.py /tmp/in.txt; echo "exit $?"
python count.py missing.txt; echo "exit $?"; python count.py --bogus; echo "exit $?"
printf '' | python count.py -; echo "exit $?"; python count.py --dump-state /tmp/in.txt
```
typer: `app = typer.Typer(); @app.command() def count(path: Path, json_: bool = typer.Option(False, "--json")): ...`

More rules:
- Errors are one line: `error: file not found: x.csv`. No traceback for expected failures; catch them in `main`.
- Add `--dump-state` (prints `{"tandem_state": 1, ...}`) and `--version`.

More pitfalls:
- Relative paths resolved against the wrong cwd; resolve with `Path(...).resolve()` and say which path failed.
