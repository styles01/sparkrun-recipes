# csvq

Build `csvq`, a small query tool for CSV files, in pure Python 3 (standard library only).

## Layout
- `csvq/__init__.py` exports `query(text: str, sql: str) -> list[dict]` and `QueryError` (the one
  exception type every error raises). `text` is the CSV content with a header row.
- `csvq/parser.py` (the query language) and `csvq/engine.py` (filtering, sorting, aggregates).
- `csvq/__main__.py`: `python -m csvq FILE "QUERY"` prints the result as CSV (header row first,
  `\n` line ends) to stdout. Errors print `error: <message>` to stderr and exit with code 2.
- `tests/test_csvq.py`: your own unittest tests (at least 12).

## Query language
`SELECT cols FROM t [WHERE cond] [GROUP BY col] [ORDER BY col [ASC|DESC] [, ...]] [LIMIT n]`

- Keywords are case-insensitive; the table name after `FROM` is always `t`.
- `cols` is `*`, or a comma list of column names and aggregates `COUNT(*)`, `SUM(col)`,
  `MIN(col)`, `MAX(col)`, `AVG(col)`. An aggregate's output column is named as written, upper
  case function, e.g. `SUM(price)`.
- Values: a cell that parses as an int is an int, else as a float a float, else a string. String
  literals use single quotes (`'Berlin'`), `''` inside is one quote.
- `WHERE`: comparisons `= != < <= > >=` between a column and a literal, joined by `AND` and `OR`
  (AND binds tighter) with parentheses. Comparing a number with a string is false, never an error.
- `GROUP BY` one column: every selected non-aggregate column must be that column. Without
  `GROUP BY`, aggregates over the whole (filtered) table give one row; mixing plain columns and
  aggregates without `GROUP BY` is an error. `AVG` is a float; `SUM` of ints is an int.
- `ORDER BY` sorts stably; numbers sort before strings; several keys allowed. Without `GROUP BY`
  it may name any column of the table, selected or not; with `GROUP BY` it names the group
  column or a selected aggregate by its output name (`ORDER BY COUNT(*) DESC`).
- `LIMIT n` keeps the first n rows.
- Errors (`QueryError`, message names the problem): unknown column, syntax error (with the
  1-based position of the bad token), aggregate of an unknown column, bad `GROUP BY` use.

## Checker
`python3 check.py` runs the acceptance cases and prints `taskcheck csvq`, then `RESULT: PASS` or
`RESULT: FAIL <- ...` with the failing groups. Done means `RESULT: PASS` and your own tests pass
with `python3 -m unittest discover -s tests`.
