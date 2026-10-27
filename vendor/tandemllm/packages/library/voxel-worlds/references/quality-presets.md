# Quality presets and holding the frame rate

Read with `package_read("references/quality-presets.md")`.

```js
// js/ui/quality.js
const PRESETS = {
  low:  { pixel: 0.7, shadows: false, bloom: 0.25, fog: 1.25 },
  med:  { pixel: 1.0, shadows: true,  bloom: 0.5,  fog: 1.0 },
  high: { pixel: Math.min(window.devicePixelRatio || 1, 2), shadows: true, bloom: 0.72, fog: 0.8 },
};
export function applyQuality(key){
  const p = PRESETS[key];
  state.pixels = p.pixel; state.shadows = p.shadows; state.bloom = p.bloom; state.fog = p.fog;
  renderer.setPixelRatio(p.pixel); renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = p.shadows;
}
```

Three chips in the panel call `applyQuality`. `state.shadows` gates `sunLight.castShadow` in
day-night; `state.bloom` and `state.fog` are read there every frame, so a chip changes the picture
at the next frame with no rebuild.

## The fps counter and the baseline

```js
let acc = 0, n = 0, timer = 0;
export function updateFps(dt){ acc += dt; n++; timer += dt; if (timer >= 0.5) { fpsEl.textContent = Math.round(n / acc); acc = n = timer = 0; } }
window.__state = () => ({ tandem_state: 1, fps: +fpsEl.textContent, voxels: voxels.size, seed: SEED, time: state.timeOfDay, calls: renderer.info.render.calls });
```

Write the baseline after step 1 (the flat ground) into `TODO.md`: `baseline: 58 fps, 4k voxels,
calls 6`. Every later line carries its fps: `done: deer (graze/walk/startle) - 57 fps`.

## What costs frames (measured on the run, in order)

1. Shadows from more than one light. Only the sun casts; lanterns are `PointLight`s without shadows.
2. Hidden voxels. The column fill goes down only to the lowest neighbour (see voxel-builder.md); a
   full fill triples the count and halves the fps.
3. One mesh per object. 400 petals as 400 meshes: 400 draw calls. One `InstancedMesh`: 1 call.
4. Allocation in the loop. `new THREE.Vector3()` per creature per frame makes the garbage collector
   stall every few seconds (a visible hitch). Keep one `tmpV`, `tmpC`, `tmpM` per module and reuse.
5. Bloom at a high pixel ratio. The composer renders at the full backing size; `pixel 0.7` on low
   is where most of the low preset's gain comes from.
6. `scene.fog = new FogExp2(...)` per frame is fine; `new Material` per frame is not.

## The rule in the build order

After each feature: render, read `__state().fps`, compare with the baseline. More than 5 below:
the feature is not done. Instance it, cut its count, or move it off the shadow path, then render
again. A dip of a few seconds at dusk when lantern lights and sun shadows overlap is known; above
that, fix.
