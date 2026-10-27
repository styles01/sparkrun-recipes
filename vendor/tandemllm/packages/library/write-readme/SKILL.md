---
name: write-readme
description: Writing or updating a README for a project or tool - what it is in two lines, install and run commands that were actually executed, configuration, a usage example with real output, and what is not covered.
---
# Writing a README

Rules:
- First line: the name and what it does in one sentence. Second: who it is for or when to use it. No marketing words.
- Sections in this order: Install, Run (or Usage), Configuration, Development (build, test), Limits or Known issues, License. Skip a section only when it is empty.
- Every command in the README has been run in this session and its output read. Paste a short real output under the usage example.
- Configuration as a table: variable, default, meaning. Secrets described, never shown.
- Requirements stated with versions (Python 3.11+, Node 22, a database).
- Keep it under 150 lines. Details go to `docs/` with a link.

What goes wrong:
- Commands that worked on the author's machine with an activated environment; the README omits the activation step.
- An example whose output no longer matches the code.
- A wall of features and no "run this first" line.
- Placeholders left in (`<your-key>` with no explanation of where to get it).

Check: run each README command in order from a clean shell (fresh clone or a temp copy) and compare the output; a reader with no context must reach a running example.

Done when:
- [ ] Each command in the README executed in this session with exit 0 (listed).
- [ ] The usage example's output pasted from a real run.
- [ ] Line count under 150 (`wc -l README.md`).

## Reference

```markdown
# count

Counts lines and words in text files, with JSON output for scripts.
For shell pipelines and quick checks; not a replacement for `wc` on binary files.

## Requirements
Python 3.11 or newer. No other dependencies.

## Install
    python -m venv .venv && . .venv/bin/activate
    pip install -e .

## Usage
    count path/to/file.txt
    cat file.txt | count - --json

Example:
    $ count README.md
    42 310

## Configuration
| Variable | Default | Meaning |
|-|-|-|
| COUNT_ENCODING | utf-8 | Encoding used to read files |

## Development
    pip install -e ".[dev]"
    pytest -q          # 12 passed

## Limits
Reads the whole file into memory; files over 1 GB are not supported.

## License
MIT
```
Verify commands: `rm -rf /tmp/rc && cp -r . /tmp/rc && cd /tmp/rc && <each command from the README>`.
