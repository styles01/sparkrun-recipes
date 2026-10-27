# Laya plays your build

Laya can play interactive software: per step the kit reads a small state, asks Laya for an action, applies it as real input and logs it.

Playtest a game, or a UI, CLI or API flow of many steps toward a goal, this way when the kit runs here (a check, not a deliverable; the testing package's budget applies). A one-screen tool (a timer, a form, a calculator) needs only the page check (`laya_drive.py page`):

1. Kit: `mkdir -p {tandem_scratch} && curl -sf {tandem_url}/v1/laya/kit/laya_drive.py -o {tandem_scratch}/laya_drive.py`
2. A short state: what the next move depends on (distance to the next gap or enemy) plus `lives`, `x`, `over`, `won`, `cause`; in a page `window.tandem_state()` and `tandem_reset()`.
3. Actions with meanings, 3+ episodes: `python3 {tandem_scratch}/laya_drive.py browser URL --actions "run=move right,jump=jump forward" --keys run=ArrowRight,jump=Space --episodes 3` (`proc`, `cmd` for programs).
4. Report: deaths with place, cause and the decisions before them; stuck spots. A crash, or a death or stuck spot that a player with real keys cannot avoid either, is a bug: fix the build, rerun, read the report. A spot where only the bot fails is not a bug: report it. Two runs that end at the same spot with no change to the build: stop playtesting and report what works and what is untested.

Recipes: `curl -s {tandem_url}/v1/library/laya-playtest`

## Reference

### A browser game
```js
// in the game: what Laya sees each step, and a fresh start
window.tandem_state = () => ({
  x: Math.round(player.x), y: Math.round(player.y), vx: +player.vx.toFixed(1), on_ground: player.onGround,
  gap_ahead: nextGap ? Math.round(nextGap.x - player.x) : null,
  enemy_ahead: nearest ? { dx: Math.round(nearest.x - player.x), dy: Math.round(nearest.y - player.y) } : null,
  lives, score, over: lives <= 0, won: player.x >= level.goal, cause: lastDeathCause,
});
window.tandem_reset = () => startLevel(1);
```
```
python3 {tandem_scratch}/laya_drive.py browser http://localhost:5173 \
  --actions "run=move right,jump=jump straight up,run_jump=jump forward,back=move left,wait=stand still" \
  --keys run=ArrowRight,jump=Space,run_jump=ArrowRight+Space,back=ArrowLeft,wait= \
  --every-ms 250 --episodes 3 --steps 400 --label "after the gap fix"
```
It starts its own headless Chrome and presses real keys over the DevTools protocol (a page that checks `isTrusted` works). `--cdp http://127.0.0.1:9222` drives a Chrome you started with `--remote-debugging-port=9222`; `--headed` shows it.

### A UI flow
The state lists the controls; Laya picks the next one and the kit clicks it (at x/y, or the selector's center):
```js
window.tandem_state = () => ({ screen: route(), errors: formErrors(),
  won: saved, controls: [...document.querySelectorAll("button, a, [role=button]")].filter(visible)
    .map((e) => ({ id: e.id || e.textContent.trim().slice(0, 30), label: e.textContent.trim().slice(0, 60), selector: cssPath(e) })) });
```
Run with `--question "Which control moves the user toward saving the form?"` (no `--actions`: the controls are the choices).

### A CLI or a simulation (headless)
Give the program a mode that prints one JSON state per line and reads one action per line (`reset` restarts):
```python
for line in sys.stdin:          # game.py --headless
    a = line.strip(); state = reset() if a == "reset" else step(state, a)
    print(json.dumps(view(state)), flush=True)
```
`python3 {tandem_scratch}/laya_drive.py proc "python3 game.py --headless" --actions "..." --episodes 5`

### An API sequence
`python3 {tandem_scratch}/laya_drive.py cmd --state "curl -s localhost:3000/debug/state" --act "curl -s -XPOST localhost:3000/{action}" --reset "curl -s -XPOST localhost:3000/debug/reset" --actions "create=create an item,update=edit it,delete=remove it,list=list items"`; the state says what exists and what the last call returned (`last_status`), `over: true` on a 5xx.

### Only the browser tool
Load the in-page kit and start it in the page (the keys are synthetic; the log goes to a local sink):
```
python3 {tandem_scratch}/laya_drive.py serve-log --port 8765 &      # appends to {tandem_scratch}/drive-log.jsonl
```
```js
eval(await (await fetch("{tandem_url}/v1/laya/kit/laya-drive.js")).text());
return await tandem_drive.start({ actions: {...}, keys: { run: ["ArrowRight"], jump: [" "] }, everyMs: 250, episodes: 3, logUrl: "http://127.0.0.1:8765/log" });
```
then `python3 {tandem_scratch}/laya_drive.py report`.

### Reading the report
`death 2 (episode 1, step 87) at x=412: fell into the gap` and the decisions before it: was the jump picked in time? If Laya jumped and the player still fell, the jump is too short or the gap too wide (the build's bug); if Laya never jumped, `gap_ahead` may be missing or wrong in the state. `stuck ... at x=1204` is a wall or a missing platform. `fallbacks (busy)`: another request held the engine; rerun. Run again after each fix with a `--label`, and compare the reports.