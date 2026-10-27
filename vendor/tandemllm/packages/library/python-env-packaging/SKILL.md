---
name: python-env-packaging
description: Setting up a Python project - virtual environment with venv or uv, pyproject.toml, pinned dependencies, src layout, entry points, and installing it in editable mode.
---
# Python environments and packaging

Rules:
- One environment per project: `uv venv` or `python -m venv .venv`, then `. .venv/bin/activate`. Never install into the system Python.
- Dependencies in `pyproject.toml` (`[project] dependencies`). Lock them: `uv lock` or `pip freeze > requirements.txt` after a working install.
- src layout: `src/pkg/__init__.py`, tests in `tests/`. Install editable: `uv pip install -e .` or `pip install -e .`.
- Console scripts via `[project.scripts] name = "pkg.cli:main"`, so the tool runs as a command after install.

What goes wrong:
- `ModuleNotFoundError` in tests: the package is not installed editable, or tests import from the wrong root. Install with `-e`.
- Two environments: the shell runs one `python`, the editor or runner another. Print `sys.executable`.
- A dependency added with pip but not written to `pyproject.toml`; the next install breaks.

Check: fresh environment, install from the project file, run `python -c "import pkg; print(pkg.__file__)"`, run the console script with `--help`, run the test suite.

Done when:
- [ ] `python --version` and `sys.executable` printed from inside the environment.
- [ ] A fresh install from `pyproject.toml` or the lock file exits 0 (output in the result).
- [ ] The console script runs with `--help`; the suite summary line shows 0 failed.

## Reference

`pyproject.toml`:
```toml
[project]
name = "mytool"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["typer>=0.12", "httpx>=0.27"]
[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.6"]
[project.scripts]
mytool = "mytool.cli:main"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/mytool"]
```
Commands with uv:
```bash
uv venv && . .venv/bin/activate
uv pip install -e ".[dev]"
uv lock                       # writes uv.lock when using uv's project mode (uv init / uv add)
python -c "import sys, mytool; print(sys.executable, mytool.__file__)"
mytool --help && pytest -q
```
With pip only: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]" && pip freeze > requirements.txt`.
Layout:
```
pyproject.toml  src/mytool/__init__.py  src/mytool/cli.py  tests/test_cli.py  README.md
```
Check which Python runs: `which python; python -c "import sys; print(sys.version, sys.executable)"`.

More rules:
- State the Python version (`requires-python = ">=3.11"`) and use the one the project says. Check `python --version` before anything.
- `.venv/`, `__pycache__/`, `dist/`, `.env` in `.gitignore`.

More pitfalls:
- Building wheels for a package with no binary for this platform (ARM Mac, aarch64 DGX); pick a version with a wheel or install the build tools.
