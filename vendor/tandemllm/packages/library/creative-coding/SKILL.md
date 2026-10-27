---
name: creative-coding
description: One-file HTML/JS visuals with sound - canvas or Three.js scenes, particles, simulations, generative art, small animated worlds with WebAudio ambience and effects; plan in three lines, write one file, render, fix the top three findings.
---
# Creative coding

Plan, three lines above the code: the scene in one sentence (what, mood, palette); its systems (every moving part, sound, input); the order (skeleton, one system at a time, a render after each).

Rules:
- One file `index.html`, CSS and JS inside, no build step, no fetch. Canvas 2D or Three.js from the reference's import map, nothing else external.
- `requestAnimationFrame`, `dt` from the timestamp capped at 0.05 s, a resize handler.
- A seed from `?seed=` for every random number. Keys: Space pauses, R restarts with a new seed, M mutes.
- Sound: one AudioContext started on the first click or key, a master gain at 0.3, each voice through its own gain envelope, peak below -12 dBFS.

Defaults: background `#0f1419`, one hue family plus an accent; 3D camera (0, 4, 10) looking at (0, 1, 0), fov 55, fog 20 to 80; 300 to 2,000 particles; a quiet bed plus one short sound per visible event.

Done when:
- [ ] The three plan lines stand above the code.
- [ ] Space, R and M work; `?seed=7` draws the same picture twice.
- [ ] A render after the last change: 0 console errors, frame 2 differs from frame 1, sound after a click, 30 fps or more.

## Reference

Import map (3D only):
```html
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script>
```
Skeleton (2D; the exemplar `exemplars/bouncing-ball.html` is the full 120-line version):
```js
const seed = +(new URLSearchParams(location.search).get('seed') || 1);
function rng(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rand = rng(seed);
const cv = document.querySelector('canvas'), ctx = cv.getContext('2d');
const W = 1280, H = 720; let paused = false, last = 0, frames = 0;
function fit(){ const s = Math.min(innerWidth / W, innerHeight / H); cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio; cv.style.width = W * s + 'px'; cv.style.height = H * s + 'px'; ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); }
addEventListener('resize', fit); fit();
function frame(ts){ const dt = Math.min((ts - last) / 1000, 0.05); last = ts; if (!paused) { update(dt); draw(); frames++; } requestAnimationFrame(frame); }
requestAnimationFrame(frame);
addEventListener('keydown', e => { if (e.code === 'Space') paused = !paused; if (e.key === 'r') location.search = '?seed=' + (seed + 1); if (e.key === 'm') master && (master.gain.value = master.gain.value ? 0 : 0.3); });
window.__state = () => ({ tandem_state: 1, frames, paused, seed });
```
Sound (unlock, bed, event):
```js
let ctx2, master;
function audio(){ if (ctx2) return; ctx2 = new AudioContext(); master = new GainNode(ctx2, { gain: 0.3 }); master.connect(ctx2.destination);
  const lp = new BiquadFilterNode(ctx2, { type: 'lowpass', frequency: 600 });
  for (const f of [110, 110.7]) { const o = new OscillatorNode(ctx2, { type: 'triangle', frequency: f }); const g = new GainNode(ctx2, { gain: 0.08 }); o.connect(g).connect(lp).connect(master); o.start(); } }
function hit(freq = 440){ if (!ctx2) return; const t = ctx2.currentTime, o = new OscillatorNode(ctx2, { type: 'sine', frequency: freq }), g = new GainNode(ctx2, { gain: 0 });
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.15, t + 0.01); g.gain.exponentialRampToValueAtTime(0.001, t + 0.3); o.connect(g).connect(master); o.start(t); o.stop(t + 0.35); }
addEventListener('pointerdown', audio, { once: true }); addEventListener('keydown', audio, { once: true });
```
Speech (a character voice): `speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance('Hello'), { pitch: 1.4, rate: 0.9 }));`
Easing: in `t*t`, out `1-(1-t)*(1-t)`, smooth `t*t*(3-2*t)`, bounce `1-Math.abs(Math.cos(t*Math.PI*2.5))*(1-t)`.
Palettes (background first): dusk `#0f1419 #1c2a3a #3b5a7a #e0a458 #f2d49b`; forest `#0b1310 #1e3b2a #3f7a4f #8fbf72 #e8d66b`; ember `#120c0c #3a1c14 #8a3a22 #e06a2d #f7c67a`.
References on request: `references/particles-boids-noise.md`, `references/webaudio-patterns.md`, `references/pixel-sprites.md`, `references/threejs-skeleton.md`.
