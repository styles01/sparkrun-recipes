---
name: threejs-scene
description: Building a WebGL scene with Three.js - renderer, camera, lights, meshes, controls, resize, animation loop, and performance (draw calls, instancing, disposal).
---
# Three.js scene

Rules:
- One version through an import map. Addons are named exports (`import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'`); merged parts all indexed or all `.toNonIndexed()`. Not in three.js: `Matrix4.makeRotationXYZ`, `Matrix4.rotateY`, `renderer.shadowMap.size` (use `geometry.rotateY(a).translate(...)`, `light.shadow.mapSize`). A geometry has `rotateY`, `translate`, `scale` and no `position`; a mesh has `position`, `rotation`, `scale` and no `translate`.
- ACES tone mapping. Bloom: RenderPass -> UnrealBloomPass (threshold ~0.85) -> OutputPass last, or the frame looks flat.
- Light since r155: hemisphere ~1.5, sun ~3, the sun on the camera's side so the faces it sees are lit. Golden hour: low warm sun 0xffb070, a sky gradient (warm horizon, blue top), fog the horizon colour. Night: sky 0x0b1530, hemisphere 0.3, moon 0.8, emissive lanterns with point lights. Start in the look asked; the Day/Night control jumps there and stops the cycle.
- Saturated colours, `metalness: 0`; foliage clustered icosahedra. Stacked flat surfaces (ground, path, pond) 0.02 apart or they speckle. Water: `Reflector` (`library_read threejs-scene` has it), not a hand-made render target.
- Frame it: `controls.target` at the subject, camera ~2x its size, further back when `aspect < 1`.

Check: screenshots at 1280 and 390 in each mode, compared with the request; console 0 errors; `window.__state()` gives frames, draw calls (under 1,000) and the mode.

## Reference

Import map:
```html
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script>
```
A day scene:
```js
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);
const sky = 0x9fd3ff, scene = new THREE.Scene();
scene.background = new THREE.Color(sky); scene.fog = new THREE.Fog(sky, 30, 120);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 300);
camera.position.set(8, 6, 12);
scene.add(new THREE.HemisphereLight(0xdcefff, 0x8a7350, 1.5));
const sun = new THREE.DirectionalLight(0xfff1d6, 3); sun.position.set(10, 18, 6); sun.castShadow = true; scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x6fa04a }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ color: 0xc8402c, metalness: 0 }));
cube.position.y = 0.5; cube.castShadow = true; scene.add(cube);
const controls = new OrbitControls(camera, renderer.domElement);
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
let frames = 0;
renderer.setAnimationLoop(() => { frames++; controls.update(); renderer.render(scene, camera); });
window.__state = () => ({ tandem_state: 1, frames, calls: renderer.info.render.calls, cam: camera.position.toArray() });
```
Night: background and fog 0x0b1530, hemisphere 0.3, moon 0.8 (0xaac4ff), lanterns as `PointLight`s.

Bloom with tone mapping kept, and a mode switch that moves everything together:
```js
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.6, 0.4, 0.85));
composer.addPass(new OutputPass());          // tone mapping and sRGB; without it the frame is flat
function skyTexture(top, horizon) {          // a vertical gradient for scene.background
  const c = document.createElement('canvas'); c.width = 2; c.height = 256;
  const g = c.getContext('2d'), grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, top); grd.addColorStop(1, horizon); g.fillStyle = grd; g.fillRect(0, 0, 2, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const MODES = { golden: { top: '#6f8fc8', horizon: '#ffb36b', hemi: 1.2, sun: 3, sunColor: 0xffb070, exp: 1.1, lamps: 0 },
                night: { top: '#050a1c', horizon: '#1a2a55', hemi: 0.3, sun: 0.8, sunColor: 0xaac4ff, exp: 0.9, lamps: 2 } };
let mode = 'golden';
function setMode(m) { const v = MODES[mode = m]; scene.background = skyTexture(v.top, v.horizon);
  scene.fog.color.set(v.horizon); hemi.intensity = v.hemi; sun.intensity = v.sun; sun.color.set(v.sunColor);
  renderer.toneMappingExposure = v.exp; lanternMat.emissiveIntensity = v.lamps; }
// render with composer.render() in the loop; composer.setSize() on resize
window.__state = () => ({ tandem_state: 1, frames, mode, calls: renderer.info.render.calls });
```
Water that mirrors the scene:
```js
import { Reflector } from 'three/addons/objects/Reflector.js';
const pond = new Reflector(new THREE.CircleGeometry(6, 48), { textureWidth: 512, textureHeight: 512, color: 0x6b8a99 });
pond.rotation.x = -Math.PI / 2; pond.position.y = 0.03; scene.add(pond);   // 0.03 above the ground
// tint and ripples: a transparent plane (opacity ~0.35, MeshStandardMaterial, roughness 0.1) 0.01 above it
```
Merging placed parts: `const g = new THREE.BoxGeometry(w, h, d); g.rotateY(a); g.translate(x, y, z); parts.push(g);` then `mergeGeometries(parts)`. With a helper object instead: `const o = new THREE.Object3D(); o.position.set(x, y, z); o.rotation.y = a; o.updateMatrix(); g.applyMatrix4(o.matrix);` (`o` is an Object3D, `g` the geometry).

What goes wrong:
- Dull, grey or dark: lights at 1 or less (dim since r155), a dark or grey background, fog nearer than the subject, metalness above 0. Black: no light, or the camera looks away.
- A TypeError on a three.js call: the name does not exist in this version. List the class's methods from its readable source of the same version, `curl -s https://unpkg.com/three@0.170.0/src/math/Matrix4.js | grep -nP '^\t\w+\('` (addons under `examples/jsm/`), never by grepping a minified build.
- Frame rate collapses: shadows on every light, `new` objects each frame, thousands of meshes instead of an `InstancedMesh`; dispose what you remove.
