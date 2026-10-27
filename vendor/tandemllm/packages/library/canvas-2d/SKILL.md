---
name: canvas-2d
description: Drawing and animating on an HTML canvas with the 2D context - render loop, device pixel ratio, sprites, transforms, text, and keeping 60 fps.
---
# Canvas 2D rendering

Rules:
- Size the canvas for the device: CSS size in px, backing store multiplied by `devicePixelRatio`, then `ctx.scale(dpr, dpr)` once.
- One loop: `requestAnimationFrame(frame)`; compute `dt` from the timestamp; clear, draw, request the next frame.
- Draw order is layer order: background, world, entities, HUD. Clear with `clearRect` every frame unless trails are intended.
- `save()`/`restore()` around every transform; translate to the camera, draw world in world units.
- Load images before the first draw (`await img.decode()`); draw sprites from a sheet with the 9-argument `drawImage`.

What goes wrong:
- Blurry output: the canvas has no dpr scaling, or CSS stretches it.
- Frame rate drops: shadows, `filter`, gradients recreated per frame, thousands of `fillText` calls. Cache static layers in an offscreen canvas.
- Transforms accumulate because a `restore()` is missing; the scene drifts away.

Check: run for 3 s, count frames, read fps (frames/3) and the positions of two entities after the run; screenshot at start and end; console errors 0.

Done when:
- [ ] Measured fps at 1280x800 and at 390x844 (target 55 or more), read through the state hook, no fps readout left on screen.
- [ ] Screenshot after 3 s shows the scene moved as expected, not a blank or static frame.
- [ ] `window.__state()` returns the frame count and entity positions after real time passed.

## Reference

Setup:
```js
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
function fit() {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', fit); fit();
```
Loop:
```js
let last = performance.now(), frames = 0;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05); last = now; frames++;
  update(dt);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.save(); ctx.translate(-camera.x, -camera.y); drawWorld(ctx); ctx.restore();
  drawHUD(ctx);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
window.__state = () => ({ tandem_state: 1, frames, camera: { ...camera }, player: { x: player.x, y: player.y } });
```
Sprite from a sheet: `ctx.drawImage(sheet, sx, sy, sw, sh, Math.round(x), Math.round(y), sw, sh)`.
Static layer cache: draw the background once to `const bg = new OffscreenCanvas(w, h)` and `drawImage(bg, 0, 0)` per frame.
Text: set `ctx.font`, `textBaseline = 'top'`, measure with `measureText(s).width`.

More rules:
- Round positions to whole pixels for crisp sprites and text.

More pitfalls:
- `dt` in milliseconds used as seconds; everything moves 1000x too fast.
- Loop never stops on page hide; `cancelAnimationFrame` on teardown.
