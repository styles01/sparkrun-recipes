---
name: voxel-worlds
description: A Three.js voxel world - an island with a pagoda, gardens, lakes, bridges, creatures with state machines, a day-night cycle, ambient sound, quality presets - built one feature at a time with a render after each, a to-do file and the frame rate held at its baseline.
---
# Voxel worlds

Build order; after each step a render, the fps, a `TODO.md` line:
1. Skeleton: renderer, camera, orbit controls, lights, a flat 64x64 voxel ground, an fps counter: the baseline you hold.
2. Terrain: a seeded height map (`?seed=`) to voxels, one `InstancedMesh` per colour, never one mesh per voxel.
3. The landmark (a pagoda) from `setVoxel(x, y, z, colour)` calls.
4. Water and trees; then one creature with a state machine (`idle`, `walk`, `startle`).
5. Day-night (sun from the time of day, lanterns at night), then sound on the first click, then quality chips low, medium, high.

Hold the frame rate: after every feature compare the fps with the baseline; 5 or more below, fix it first (fewer voxels, instancing, no `new` per frame).

Rules: one seeded generator for all random numbers, the seed in the HUD; `groundTop(x, z)` places everything, nothing floats or sinks; modules as in the reference, or one `index.html` without a file tool; `TODO.md` lists `done` and `todo`, a line per feature with its fps.

Done when:
- [ ] Every feature of the prompt is under `done` with its fps.
- [ ] The last render: fps within 5 of the baseline, 0 console errors.
- [ ] `?seed=7` builds the same island twice; screenshots at noon and midnight.

## Reference

Seeded random and noise (`js/core/rng.js`):
```js
export const SEED = +(new URLSearchParams(location.search).get('seed') || 20260317);
export function mulberry32(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const rand = mulberry32(SEED), TAU = Math.PI * 2;
export const randRange = (a, b) => a + (b - a) * rand(), pick = arr => arr[(rand() * arr.length) | 0];
export const lerp = (a, b, t) => a + (b - a) * t, smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
```
Voxels and the ground (`js/core/voxels.js`): `heights` is a `Float32Array(GRID*GRID)`; `setVoxel(x,y,z,c,b)` puts `{x,y,z,c,b}` in a `Map` keyed `x,y,z` (`b` is the material bucket: 0 terrain, 1 built, 4 glowing, 5 water); `groundTop(x,z)` returns `Math.round(heights[idx])`.
Meshes (`js/scene/meshes.js`): group voxels by bucket, one `InstancedMesh(BoxGeometry(1,1,1), MeshLambertMaterial, n)` per bucket, `setMatrixAt` + `setColorAt` per voxel, `instanceMatrix.needsUpdate = true`, shadows cast by buckets 1 and 2 only.
Day-night (`js/scene/daynight.js`): `ang = (timeOfDay - 0.25) * TAU; sunDir = (cos(ang)*0.78, sin(ang), cos(ang)*0.42)`; `day = smoothstep(-0.06, 0.20, sunDir.y)`; sky top `#04070f -> #2a63c8` by `day`, horizon `#0b1530 -> #a9cdea`; dusk tint `#ffab6a`; `sun.intensity = day * 2.6`; `moon.intensity = (1 - day) * 0.55`; `fog = FogExp2(horizon, 0.0026 + night * 0.0011)`; lanterns `intensity = night * 40 * (1 + 0.14 * sin(11 t + 2.3 i))`.
Creature state machine: `state = 'graze'; timer = randRange(3, 7)`; on `timer <= 0` go `walk` to a new target `(cos a * r, 0, sin a * r)`; within 2 units go back to `graze`; `y = groundTop(round x, round z) + bob`; legs `rotation.x = ±sin(phase) * 0.5` while walking.
Ambient audio: a 2 s white-noise buffer, looped; wind = lowpass 420 Hz at gain 0..0.22 by wind strength; water = bandpass 900 Hz at 0.045; chirp = sine 1800..3400 Hz with an 80 ms ramp up, 160 ms decay, by day at 0.35 a second; bell = sine 520 + 782 Hz, 2.5 s decay, by night at 0.06 a second.
Quality presets: low `{pixel 0.7, shadows off, bloom 0.25, fog 1.25}`, med `{1.0, on, 0.5, 1.0}`, high `{min(dpr, 2), on, 0.72, 0.8}`; apply with `renderer.setPixelRatio(pixel); renderer.setSize(innerWidth, innerHeight)`.
References on request: `references/module-layout.md`, `references/voxel-builder.md`, `references/day-night.md`, `references/creature-state-machines.md`, `references/ambient-audio.md`, `references/quality-presets.md`; exemplars `exemplars/deer.md`, `exemplars/shrine-bell.md` (prompt, plan, diff).
