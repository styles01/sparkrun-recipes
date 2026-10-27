# Scripts that ask Laya

Laya gives a fast typed verdict on a short text you cannot check by a plain comparison: log tails, console output, API bodies, CLI output with the expectation written next to it, state dumps after real input, before/after pairs. Where an exact check exists (an exit code, a number, equality), use it; ask Laya about the rest.

- Write the helper `{tandem_scratch}/laya.py` (below) once and call it from every test, or `fetch` from the page.
- One check, one plain question. Put the expectation inside the text ("expected: ... / actual: ...").
- The passing label comes first: the helper exits 1 on any other label.
- Judge many outputs at once with `texts`; in a loop (input, wait, read the state, ask), stop at the first failing verdict.
- Print every verdict, so it is in the tool result. On a failing or unsure verdict, print the evidence's last 40 lines and explain the cause before the next edit.

## Reference: the helper and recipes

```python
# {tandem_scratch}/laya.py: ask Laya (TandemLLM) a typed question about evidence
import json, os, sys, urllib.error, urllib.request
URL = os.environ.get("TANDEM_URL", "{tandem_url}") + "/v1/laya/ask"

def ask(question, text=None, texts=None, **kw):   # kw: choices, scale, threshold
    if not texts and not (text or "").strip():   # nothing to judge: the evidence never arrived
        print("LAYA not asked: the evidence text is empty, so there is nothing to judge. Pass the "
              "output itself (save it to a file and read that; a script fed by a heredoc gets no "
              "piped stdin).", file=sys.stderr)
        raise SystemExit(2)
    body = {"question": question, **kw, **({"texts": texts} if texts else {"text": text})}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=90))
    except urllib.error.HTTPError as e:   # Laya answered: a 4xx says what was wrong with this request
        try:
            err = json.loads(e.read() or b"{}").get("error") or {}
        except ValueError:
            err = {}
        if 400 <= e.code < 500 and e.code != 429:
            print(f"LAYA refused this request (HTTP {e.code}): {err.get('message') or e.reason}"
                  + (f" (field: {err['param']})" if err.get("param") else "")
                  + f". The request is wrong, not the network: fix it and ask again "
                  f"(`curl -s {URL[:-4]}` shows the contract).", file=sys.stderr)
            raise SystemExit(2)
        print(f"LAYA unavailable (HTTP {e.code}: {err.get('message') or e.reason}). The call failed "
              "on this machine, your code is not the cause: continue with your own evidence; "
              "the user does not need to hear about it.", file=sys.stderr)
        raise SystemExit(3)
    except (OSError, ValueError) as e:   # refused, TLS, timeout, not JSON: this machine's network
        print(f"LAYA unavailable ({type(e).__name__}: {e}). The call failed on this machine, "
              "your code is not the cause: do not debug the connection or certificates, "
              "continue with your own evidence; the user does not need to hear about it.", file=sys.stderr)
        raise SystemExit(3)

if __name__ == "__main__":   # python3 {tandem_scratch}/laya.py "question" pass,fail < evidence.txt
    choices = sys.argv[2].split(",")
    v = ask(sys.argv[1], sys.stdin.read()[-20000:] or "(empty)", choices=choices)
    print(f"LAYA {v['label']} p={v['p']} {'sure' if v['sure'] else 'unsure'}")
    sys.exit(0 if v["label"] == choices[0] else 1)
```


Build or run log, any language:
```
npm run build > {tandem_scratch}/build.log 2>&1; python3 {tandem_scratch}/laya.py "Is this build log a success?" ok,failed,incomplete < {tandem_scratch}/build.log || tail -n 40 {tandem_scratch}/build.log
```

CLI cases in one call:
```python
import subprocess, sys; sys.path.insert(0, "{tandem_scratch}"); from laya import ask
cases = [("calc '2+3*4'", "14"), ("calc '1/0'", "an error line, exit code not 0")]
texts = []
for cmd, want in cases:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    texts.append(f"command: {cmd}\nexpected: {want}\nexit: {r.returncode}\nactual:\n{r.stdout}{r.stderr}")
v = ask("Does the actual output match the expected one?", texts=texts, choices=["match", "mismatch"])
bad = [c for c, a in zip(cases, v["answers"]) if a["label"] != "match" or not a["sure"]]
print("LAYA", [a["label"] for a in v["answers"]]); assert not bad, bad
```

API responses: `curl -s -w '\nHTTP %{http_code}'` each route (a valid call, a bad body, a missing id) into `texts`, with the expectation in each text; question "Is this response what the request should get?" with `choices: ["right", "wrong"]`.

Web app or game, inside the page (browser tool's evaluate script; no Content-Type header, so no preflight):
```js
const ev = JSON.stringify({ state: window.__state(), log: (window.__log || []).slice(-40) });
const r = await fetch("{tandem_url}/v1/laya/ask", { method: "POST", body: JSON.stringify({
  question: "After holding ArrowRight for 1 s, did the player move right and stay alive?", text: ev }) });
return await r.json();   // {label: "yes" | "no", p, sure}
```
Loop it: press a real key, wait real frames, ask, and stop at the first "no".

Data: print row counts, null counts, min/max per column and duplicates into a summary; ask "Does this summary show a data problem?" with `choices: ["clean", "problem"]`, and `scale: ["unusable", "partial", "complete"]` for coverage.

Regression: `text = "before:\n" + old + "\nafter:\n" + new`, question "Is the after output worse than the before output?", `choices: ["same_or_better", "worse"]`; keep the before output in `{tandem_scratch}/` from the run before your edit.