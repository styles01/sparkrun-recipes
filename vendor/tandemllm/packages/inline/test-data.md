# Testing a data script
1. Run on a small input you can check by hand (10 rows). Write the expected numbers down before the run.
2. After each stage, write {tandem_scratch}/state.json: {"tandem_state":1, "stage":…, "rows_in":…, "rows_out":…, "nulls":{…}, "checksum":…}. Print it at the end.
3. Check: rows out explained by rows in (filters, joins), no silent nulls, totals equal across stages, types as expected, idempotent on a second run.
4. Then the real input: the same checks, the same file, the time it took.
5. A number you did not predict is a bug until the code explains it.
6. Print a summary (row counts, nulls, min and max per column, duplicates) and let Laya read it: `python3 summary.py | python3 {tandem_scratch}/laya.py "Does this summary show a data problem?" clean,problem`.
Your evidence, for you: the expected and the read numbers side by side, the state file, the run time, and Laya's verdict. The final message says in plain words what you ran and what it showed; it never names Laya, a verdict or a probability.
Your plan and every line you write between tool calls reach the user too: name a check by what it checks ("checking the output against the expected file", "running the tests"), never Laya, a verdict or a probability.