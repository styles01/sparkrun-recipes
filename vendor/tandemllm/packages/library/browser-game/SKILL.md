---
name: browser-game
description: Building a 2D browser game (platformer, shooter, puzzle) - game loop with fixed timestep, keyboard input, tile collisions, camera, HUD, lives and win state.
---
# Browser game

Rules:
- Build in this order, running after each: tile grid rendered; player drawn; input moves it; gravity and ground collision; camera; one enemy; pit death and respawn; goal; HUD. Never all at once.
- Fixed timestep: accumulate real time, step physics at 1/60 s, render once per frame.
- Input is a set of keys from `keydown`/`keyup` on `window`; read the set in the update, never in the handler.
- Collide per axis: move x, resolve against tiles, then move y, resolve. Set `onGround` only from the y resolve.

What goes wrong:
- Player sinks or tunnels: velocity larger than a tile per step, or both axes resolved at once. Cap speed and resolve per axis.
- Lives lost at start: spawn inside a tile, an enemy on the spawn, or death checked before the first step.
- Jump never fires: `onGround` reset before the check, or the key handler consumed the press.

Check: play it through the browser tool: press Right for 1 s, read the state (x grew, y stable, onGround true), jump, read the state at 3 points, walk to the goal with inputs only, screenshot start, middle, end.

Done when:
- [ ] A playthrough from spawn to goal with real key events, with the state read at least 4 times.
- [ ] Lives after 2 s of standing still: unchanged (read from the state).
- [ ] Screenshots at 1280 and 390 attached; console 0 errors.

## Reference

Loop with a fixed step:
```js
const STEP = 1 / 60; let acc = 0, last = performance.now();
function frame(now) {
  acc += Math.min((now - last) / 1000, 0.1); last = now;
  while (acc >= STEP) { update(STEP); acc -= STEP; }
  render(); requestAnimationFrame(frame);
}
```
Input:
```js
const keys = new Set();
addEventListener('keydown', e => { keys.add(e.code); if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault(); });
addEventListener('keyup', e => keys.delete(e.code));
```
Per-axis tile collision (tiles `T` px, `solid(cx, cy)`):
```js
function moveX(p, dx) {
  p.x += dx;
  const top = Math.floor(p.y / T), bottom = Math.floor((p.y + p.h - 1) / T);
  if (dx > 0) { const cx = Math.floor((p.x + p.w - 1) / T); for (let cy = top; cy <= bottom; cy++) if (solid(cx, cy)) { p.x = cx * T - p.w; p.vx = 0; } }
  else if (dx < 0) { const cx = Math.floor(p.x / T); for (let cy = top; cy <= bottom; cy++) if (solid(cx, cy)) { p.x = (cx + 1) * T; p.vx = 0; } }
}
function moveY(p, dy) {
  p.y += dy; p.onGround = false;
  const left = Math.floor(p.x / T), right = Math.floor((p.x + p.w - 1) / T);
  if (dy > 0) { const cy = Math.floor((p.y + p.h - 1) / T); for (let cx = left; cx <= right; cx++) if (solid(cx, cy)) { p.y = cy * T - p.h; p.vy = 0; p.onGround = true; } }
  else if (dy < 0) { const cy = Math.floor(p.y / T); for (let cx = left; cx <= right; cx++) if (solid(cx, cy)) { p.y = (cy + 1) * T; p.vy = 0; } }
}
```
Cap `|vy|` at `T - 1` per step. Camera: `cam.x = clamp(p.x - W/2, 0, levelW - W)`.
Test through the tool: `press_key ArrowRight` hold 1000 ms, `sleep 300`, `evaluate_script window.__state()`.

More rules:
- Camera clamps to the level bounds. Spawn point is on solid ground with 2 free tiles above.
- State dump: `window.__state()` with player x/y/vx/vy, onGround, lives, score, camera, level, status.

More pitfalls:
- Enemy spawns on the player; a win triggered by proximity on frame 1.
- Camera shows outside the level or jitters: no clamp, no rounding.
