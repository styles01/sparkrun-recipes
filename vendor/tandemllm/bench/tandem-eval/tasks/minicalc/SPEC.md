# minicalc

Build `minicalc`, a small calculator language in pure Python 3 (standard library only).

## Layout
- `minicalc/__init__.py` exports `evaluate(source: str, env: dict | None = None) -> float | int`
  and `CalcError` (the one exception type every error raises).
- `minicalc/lexer.py`, `minicalc/parser.py`, `minicalc/interp.py`: tokenizer, recursive-descent
  parser producing an AST, and the evaluator.
- `minicalc/__main__.py`: `python -m minicalc "expr"` prints the result; with no argument it reads
  lines from stdin (a REPL without a prompt) and prints one result per non-empty line, keeping
  variables between lines. Errors print `error: <message>` and the exit code is 1 for the
  one-shot form.
- `tests/test_minicalc.py`: your own unittest tests (at least 15).

## Language
- Numbers: integers and decimals (`3`, `2.5`, `.5`). Integer arithmetic stays int where exact.
- Operators, by precedence low to high: `+ -`, `* / %`, unary `-`, `^` (power, right-assoc).
  `/` is true division; `%` is modulo. Parentheses group.
- Variables: `x = 3` assigns and the expression's value is 3; names are `[A-Za-z_][A-Za-z0-9_]*`.
  Several statements on a line are separated by `;`, the value is the last one.
- Constants `pi` and `e`; functions `sqrt`, `abs`, `min`, `max` (min/max take 1+ args), `round`
  (1 or 2 args). Calling an unknown function or using an unknown name is an error.
- Errors (all `CalcError`, message names the problem): division or modulo by zero, unknown name,
  unknown function, wrong argument count, unexpected character (with its column, 1-based),
  unbalanced parentheses, trailing input.

## Checker
`python3 check.py` runs the acceptance cases and prints `RESULT: PASS` or
`RESULT: FAIL <- ...` with the failing groups. Done means `RESULT: PASS` and your own tests pass
with `python3 -m unittest discover -s tests`.
