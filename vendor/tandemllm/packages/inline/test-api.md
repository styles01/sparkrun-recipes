# Testing a backend or API
1. Start the server in the background (`cmd > {tandem_scratch}/run.log 2>&1 &`), wait for the port, read the log tail. Use the absolute project path.
2. Add GET /__state: {"tandem_state":1, "uptime":…, "routes":[…], "db":{"ok":true,"rows":…}, "last_error":…}. Add /healthz if missing.
3. Call every route with curl: a valid request, an invalid body, a missing auth, a not-found id. Read the status code and the body each time. A 500 is a bug; a 200 on bad input is a bug.
4. State after writes: create, then read back, then /__state. Numbers must agree.
5. Stop the server you started at the end; say that you did.
6. Let Laya judge every response: `(echo "expected: 404 with an error body"; curl -s -w '\nHTTP %{http_code}' localhost:8080/items/999) | python3 {tandem_scratch}/laya.py "Is this response what the request should get?" right,wrong`.
Your evidence, for you: the routes called with their codes, the read-back checks, the log tail, and Laya's verdicts. The final message says in plain words what you ran and what it showed; it never names Laya, a verdict or a probability.
Your plan and every line you write between tool calls reach the user too: name a check by what it checks ("checking the output against the expected file", "running the tests"), never Laya, a verdict or a probability.