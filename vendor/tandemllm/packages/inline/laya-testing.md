# Laya: the test judge you can call

TandemLLM serves Laya, a fast decision model (about 30 ms). It reads a short text (about 1,100 characters, head and tail) and answers a typed question with a probability. It writes nothing. Call it from your scripts through the provider:

```
curl -sS --max-time 60 "${TANDEM_URL:-{tandem_url}}/v1/laya/ask" -H 'Content-Type: application/json' \
  -d '{"question":"Does this output show a failure?","choices":["failure","clean"],"text":"<log tail>"}'
# {"label":"failure","p":0.93,"sure":true,...}
```

Also `scale` (levels, lowest first), or no choices for a yes-no question; `texts: [...]` judges up to 16. No key. `curl -s .../v1/laya` shows the contract.

You must use it:
- After every run that produced evidence (a log, an output, the console, a state dump), your test script asks Laya and prints the verdict. A page checked with the kit's page check has it already (the report's `laya` field): read it, write no Laya script of your own for it.
- Before "done", run the check script and read Laya's verdicts. They are for you: the final message says in plain words what was checked, never a label or a probability.
- Your plan and every line you write between tool calls reach the user too: name a check by what it checks ("checking the output against the expected file", "running the tests"), never Laya, a verdict or a probability.
- failure, no or mismatch is a bug to explain before the next edit; `"sure": false` means read the evidence yourself.
- A call that fails (connection refused, a TLS or certificate error, a timeout, a 5xx) is this machine's network, not your code: print "Laya unavailable" once in your check output and go on with your own evidence; it is not news for the user. At most one retry; never debug the transport (no openssl, no CA hunting, no -k).
- A call refused with a 4xx (the helper prints "LAYA refused this request" or "LAYA not asked" and exits 2) is your request, not the network: read the message, fix the call (an empty text, a missing question, a bad choices list) and ask again.
- An interactive build (a game, a UI flow, a CLI): Laya also plays it, step by step, and reports every death and stuck spot (laya-playtest).