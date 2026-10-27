# Contributing

Thank you for helping. Bug reports, measurements from other boards and pull requests are welcome.

## Before a pull request

- Run the CPU test suite: every `tests/test_*.py` is a standalone script (`python tests/test_cache.py`), and all of them must pass: `python tests/script_suite.py --all` runs each one and fails a file that ran no test. A pytest-style module ends with `from scriptmode import run; run(globals())` under `if __name__ == "__main__":`. The GPU tests in `tests/gpu/` need a DGX Spark.
- Run the dashboard tests if you touched `dashboard/`: `npm ci && npm test` in that folder.
- A change must not change the text the engine writes. [docs/exactness.md](docs/exactness.md) lists the checks, and a speed claim needs the method in [docs/measurement.md](docs/measurement.md).

## Commit messages and docs

Commits, docs, the CHANGELOG, the README and model cards are written for a reader of the public repository.

- Commit subject: `KEY: imperative summary`, at most 72 characters, with the ticket key (`ENG-287: Keep attention split buffers per class`).
- Commit body: after a blank line, at most 5 lines wrapped at 72: what changed and why, in technical terms, and at most one evidence line (a test name, or a number with its conditions).
- Docs are reference text: facts, terms, tables, commands, and each number with its conditions on the same line (board, profile, request shape, date).
- No session context, run diaries or long measurement logs; raw runs go in a measurement file under `models/<model>/measurements/`.
- No people: no names of who asked or decided, no "as requested". Upstream authors are credited by name where their work is used.
- No private hostnames, home paths or personal notes.

## Community weight packs

To list weights you published on Hugging Face in the setup guide, send a pull request that adds one file, `models/<model>/community/<id>.json`. [docs/community-weights.md](docs/community-weights.md) explains each field, and a maintainer runs the pack on a DGX Spark before it shows as verified. The agreement below applies to that file too.

## Contributor License Agreement

TandemLLM is dual-licensed: AGPL-3.0 for everyone, and a commercial license from the copyright holder ([COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). To keep that possible, every contributor must agree to the Contributor License Agreement below before a pull request is merged. To agree, add this sentence to your pull request: "I have read the Contributor License Agreement in CONTRIBUTING.md and I agree to it."

### Contributor License Agreement

1. "You" means the person or legal entity submitting a contribution. "Contribution" means any code, documentation or other material you submit to this repository. "Maintainer" means Khaled Bakeer, the copyright holder of TandemLLM.
2. You keep the copyright in your contribution.
3. You grant the Maintainer and anyone who receives software from the Maintainer a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, copy, modify, publish, distribute, sublicense and relicense your contribution, under the AGPL-3.0 or under any other license, including commercial licenses.
4. You grant the same parties a perpetual, worldwide, non-exclusive, royalty-free, irrevocable patent license for any patent claims you can license that your contribution needs, to make, use, sell and distribute the contribution alone or as part of TandemLLM.
5. You confirm that the contribution is your own work, or that you have the right to submit it under these terms. If your employer has rights in your work, you confirm that it allowed you to submit it.
6. You are not expected to support your contribution. It is provided "as is", without warranties of any kind.
