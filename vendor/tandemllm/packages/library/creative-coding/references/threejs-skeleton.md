# Three.js skeleton for a creative scene

Read with `package_read("references/threejs-skeleton.md")`. The threejs-scene skill holds the
general rules (camera first, instancing, disposal); this is the one-file skeleton with the
creative-coding defaults (seed, pause, mute, fog, a sound bed).

```html
<!doctype html><meta charset="utf-8"><title>Scene</title>
<style>html,body{margin:0;height:100%;background:#0f1419;overflow:hidden}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"https://unpkg.com/three@0.170.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
const seed = +(new URLSearchParams(location.search).get('seed') || 1);
function rng(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rand = rng(seed);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1419); scene.fog = new THREE.Fog(0x0f1419, 20, 80);
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 200);
camera.position.set(0, 4, 10); camera.lookAt(0, 1, 0);
scene.add(new THREE.HemisphereLight(0xa9cdea, 0x6b5a3a, 1.5));
const sun = new THREE.DirectionalLight(0xfff3e0, 3); sun.position.set(5, 10, 4); scene.add(sun);
const controls = new OrbitControls(camera, renderer.domElement); controls.target.set(0, 1, 0);
// --- the scene's systems go here: build once, animate in update(dt)
const N = 400, geo = new THREE.SphereGeometry(0.08, 6, 6), mat = new THREE.MeshStandardMaterial({ color: 0xe0a458 });
const mesh = new THREE.InstancedMesh(geo, mat, N); scene.add(mesh);
const pos = Array.from({ length: N }, () => new THREE.Vector3((rand() - 0.5) * 20, rand() * 10, (rand() - 0.5) * 20));
const m = new THREE.Matrix4();
function update(dt){ for (let i = 0; i < N; i++) { pos[i].y -= dt * 1.5; if (pos[i].y < 0) pos[i].y = 10; m.setPosition(pos[i]); mesh.setMatrixAt(i, m); } mesh.instanceMatrix.needsUpdate = true; }
// --- loop, keys, resize, state
let paused = false, frames = 0, last = performance.now();
renderer.setAnimationLoop(() => { const now = performance.now(), dt = Math.min((now - last) / 1000, 0.05); last = now; if (!paused) { update(dt); frames++; } controls.update(); renderer.render(scene, camera); });
addEventListener('resize', () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
addEventListener('keydown', e => { if (e.code === 'Space') paused = !paused; if (e.key === 'r') location.search = '?seed=' + (seed + 1); });
window.__state = () => ({ tandem_state: 1, frames, paused, seed, calls: renderer.info.render.calls });
</script>
```

Add the sound block from the skill's reference (`audio()`, `hit()`, the M key) after the loop.

Defaults that look right at once: fog 20 to 80 for a 20-unit scene; `MeshStandardMaterial` with
`roughness 0.8` and `metalness 0`; one hemisphere light (~1.5) plus one directional (~3: lights at 1 or
less are dim since r155); the dark background is a mood for particles at night: a daylight scene
gets a sky colour, its fog the same (the threejs-scene skill's day scene); `InstancedMesh` for anything repeated;
a ground plane `new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: 0x1c2a3a }))` rotated `-Math.PI / 2`.
