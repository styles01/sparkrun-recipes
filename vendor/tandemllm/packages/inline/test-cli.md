# Testing a CLI
1. Build or install it; write the output to {tandem_scratch}/build.log; check the exit code.
2. Add --dump-state (or --selftest): prints {"tandem_state":1, "version":…, "config":…, "last_run":{…}} to stdout and exits 0.
3. Run it on a real input file you created with known content; compare the output with what the input implies, number by number. Run the error paths too: a missing file, a bad argument, empty input. Each must exit non-zero with one clear line.
4. Pipe cases: stdin in, stdout out, stderr for errors only.
5. Keep every run under the bash timeout; long runs go to the background with their log in {tandem_scratch}/run.log and `tail -n 60` after.
6. Let Laya judge each output against what you expected: `(echo "expected: 3 rows, total 42"; ./tool in.csv 2>&1) | python3 {tandem_scratch}/laya.py "Does the actual output match the expected one?" match,mismatch`.
Your evidence, for you: the commands run, their exit codes, the outputs compared, the state dump, and Laya's verdicts. The final message says in plain words what you ran and what it showed; it never names Laya, a verdict or a probability.
Your plan and every line you write between tool calls reach the user too: name a check by what it checks ("checking the output against the expected file", "running the tests"), never Laya, a verdict or a probability.