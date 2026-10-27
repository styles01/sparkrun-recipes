# Module layout

Curated from a finished voxel island (about 95k voxels, 50 fps on a laptop GPU, 1,700 lines of
modules plus one file per creature). Read with `package_read("references/module-layout.md")`.

```
index.html          the import map, a <div id="app">, the HUD, the control panel, <script type="module" src="js/main.js">
css/style.css       the HUD and panel (40 lines)
js/main.js          imports every builder and creature; init() builds in order; animate() updates in order
js/core/rng.js      SEED from ?seed=, mulberry32, rand, randRange, pick, noise (fbm, ridged), hash01, lerp, smoothstep
js/core/world.js    constants: GRID 256, island radius, isle centres, plaza, lake, hill, DAY_LEN 120 s, the palette P
js/core/state.js    one object: timeOfDay, timeScale, bloom, exposure, fog, pixels, shadows, every feature toggle
js/core/voxels.js   heights Float32Array, voxels Map, setVoxel, groundTop, reserves
js/build/terrain.js generateHeights() (noise + island masks), buildTerrain() (columns of voxels)
js/build/pagoda.js  buildPagoda() (tiers, walls, windows, roof plates, corner curls, finial)
js/build/garden.js  lanterns (with lights), torii, trees, paths, flowers
js/build/water.js   the lake, the sea torii, the shrine, a second isle
js/build/bridges.js, pier.js, plaza.js, ruins.js
js/scene/renderer.js  scene, renderer (ACES tone mapping, soft shadows), camera, OrbitControls, onResize
js/scene/lights.js    sun (shadow camera ±110), moon, hemisphere
js/scene/sky.js       sky dome shader, sun and moon discs, stars, clouds, the sea plane, the sun glint
js/scene/daynight.js  updateDayNight(t): colours, lights, fog, exposure, lanterns, stars, moon phase
js/scene/meshes.js    buildMeshes(): one InstancedMesh per bucket
js/scene/post.js      composer with a bloom pass
js/creatures/<one>.js spawnX() and updateX(dt, t) per creature or effect (koi, birds, petals, fireflies, rain, fox, deer, owl, bell, ...)
js/ui/panel.js        sliders and chips bound to state; quality.js; camera.js (presets); minimap.js; share.js (seed URL, PNG)
TODO.md               done / todo lists with fps
```

The import map (unpkg for three, jsdelivr for simplex-noise):

```html
<script type="importmap">{"imports":{
  "three":"https://unpkg.com/three@0.170.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/",
  "simplex-noise":"https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/+esm"}}</script>
```

`main.js` in two parts:

```js
function init(){
  generateHeights(); buildTerrain(); buildPlaza(); buildPagoda(); buildGarden(); buildPier(); buildWater(); buildBridges();
  buildMeshes();                     // voxels become meshes ONCE, after every builder
  buildSky(); buildClouds(); buildSea(); setupLights();
  spawnPetals(); spawnKoi(); spawnBirds(); spawnFireflies(); spawnDeer(); spawnBell();   // creatures after the meshes
  setupPanel(); setupQuality(); overlay.remove(); animate();
}
function animate(){
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  state.timeOfDay = (state.timeOfDay + dt * state.timeScale / DAY_LEN) % 1;
  updateDayNight(t); controls.update();
  updatePetals(dt, t); updateKoi(dt, t); updateBirds(dt, t); updateFireflies(dt, t); updateDeer(dt, t); updateBell(dt, t);
  updateAudio(dt); updateFps(dt); composer.render();
}
```

Rules the layout keeps:
- A builder writes voxels only (`setVoxel`); it never touches the scene. `buildMeshes()` runs once after all builders. A feature that adds voxels after that must go before `buildMeshes()` in `init()`.
- A creature owns its meshes (small groups, or one `InstancedMesh` for many of a kind) and exports `spawn` and `update`.
- `state.js` is the only shared mutable object; the panel writes it, the updaters read it.
- Every module imports `rand` from `rng.js`; `Math.random` appears nowhere.

ES modules are cached by the browser: reload with the cache disabled after every change, or
append `?v=N` to the module URL while building.
