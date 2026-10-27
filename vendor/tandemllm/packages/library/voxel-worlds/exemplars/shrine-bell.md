# Exemplar: a shrine bell you can ring

A finished feature of the overnight build as prompt, plan and diff. Read with
`package_read("exemplars/shrine-bell.md")`.

## Prompt

> Put a bell in a wooden frame on the hill. When I click the bell it swings like a pendulum and
> rings (a real bell sound, two partials, long decay).

## Plan

1. New file `js/creatures/bell.js`: `spawnBell()` builds a frame (two posts and a beam from
   `box`) at the hill top and a `Group` for the bell with its pivot at the beam; the bell is an
   open cylinder (`CylinderGeometry(0.5, 1.0, 1.6, 10, 1, true)`), a lip and a clapper.
2. A raycaster on `pointerup` (ignored when the pointer moved more than 6 px, so orbiting does
   not ring): hit on the bell meshes calls `ringBell()`.
3. `ringBell()` adds angular velocity; `updateBell(dt)` integrates a damped pendulum
   `accel = -swing * 14 - swingVel * 1.2` and sets `bell.rotation.z = swing * 0.3`.
4. Sound: `playBell()` in `audio.js`, two sine partials at 520 and 782 Hz, gains 0.12 and 0.05,
   decays 2.5 and 1.8 s, through the master.
5. `main.js`: import, spawn, update. Render, click through the browser tool, read the state.

## Diff

```diff
--- a/js/creatures/audio.js
+++ b/js/creatures/audio.js
+export function playBell(){
+  if(!ctx){ setupAudio(); }
+  if(!ctx) return;
+  ctx.resume();
+  const t=ctx.currentTime;
+  for(const [f,a,d] of [[520,0.12,2.5],[782,0.05,1.8]]){
+    const o=ctx.createOscillator(); o.type='sine'; o.frequency.value=f;
+    const g=ctx.createGain(); g.gain.setValueAtTime(a,t); g.gain.exponentialRampToValueAtTime(0.0001,t+d);
+    o.connect(g); g.connect(master); o.start(); o.stop(t+d+0.1);
+  }
+}
--- /dev/null
+++ b/js/creatures/bell.js
+import * as THREE from 'three';
+import { scene, camera, renderer } from '../scene/renderer.js';
+import { HILL } from '../core/world.js';
+import { playBell } from './audio.js';
+
+let bell, bellMeshes=[], swing=0, swingVel=0, rings=0;
+const GOLD=0xffcf6a, WOOD=0x5a3d28;
+function box(w,h,d,color,x,y,z,parent){
+  const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshLambertMaterial({color}));
+  m.position.set(x,y,z); m.castShadow=true; (parent||scene).add(m); return m;
+}
+export function spawnBell(){
+  const x=HILL.x, z=HILL.z+10, gy=HILL.top;
+  box(0.6,6,0.6,WOOD,x-3,gy+3,z); box(0.6,6,0.6,WOOD,x+3,gy+3,z); box(7,0.5,0.5,WOOD,x,gy+6,z);   // frame
+  bell=new THREE.Group(); bell.position.set(x,gy+6,z);                                              // pivot at the beam
+  const metal=new THREE.MeshStandardMaterial({color:GOLD,metalness:0.6,roughness:0.3,side:THREE.DoubleSide});
+  const body=new THREE.Mesh(new THREE.CylinderGeometry(0.5,1.0,1.6,10,1,true),metal); body.position.y=-1.0; body.castShadow=true; bell.add(body);
+  const lip=new THREE.Mesh(new THREE.CylinderGeometry(1.05,1.05,0.15,10),metal); lip.position.y=-1.8; bell.add(lip);
+  const clapper=new THREE.Mesh(new THREE.SphereGeometry(0.18,8,8),new THREE.MeshStandardMaterial({color:0x3a2a1a})); clapper.position.y=-1.9; bell.add(clapper);
+  scene.add(bell); bellMeshes=[body,lip,clapper];
+  const ray=new THREE.Raycaster(), ndc=new THREE.Vector2(); let dx=0,dy=0;
+  renderer.domElement.addEventListener('pointerdown',e=>{ dx=e.clientX; dy=e.clientY; });
+  renderer.domElement.addEventListener('pointerup',e=>{
+    if(Math.hypot(e.clientX-dx,e.clientY-dy)>6) return;                                           // a drag is not a click
+    ndc.x=(e.clientX/window.innerWidth)*2-1; ndc.y=-(e.clientY/window.innerHeight)*2+1;
+    ray.setFromCamera(ndc,camera);
+    if(ray.intersectObjects(bellMeshes,false).length>0) ringBell();
+  });
+}
+export function ringBell(){ swingVel+=6; rings++; playBell(); }
+export function updateBell(dt){
+  if(!bell) return;
+  const accel=-swing*14-swingVel*1.2;                                                              // damped pendulum
+  swingVel+=accel*dt; swing+=swingVel*dt; bell.rotation.z=swing*0.3;
+}
+export function bellState(){ return { rings, swing:+swing.toFixed(3) }; }
--- a/js/main.js
+++ b/js/main.js
+import { spawnBell, updateBell, bellState } from './creatures/bell.js';
 ...
+  spawnBell();
 ...
+  updateBell(dt);
 ...
-window.__state=()=>({tandem_state:1,fps:+$('fps').textContent,voxels:voxels.size,seed:SEED});
+window.__state=()=>({tandem_state:1,fps:+$('fps').textContent,voxels:voxels.size,seed:SEED,bell:bellState()});
--- a/TODO.md
+++ b/TODO.md
+- done: shrine bell (click to ring, damped swing, two-partial bell sound) - 50 fps (baseline 50)
```

## What was checked

A click through the browser tool at the bell's screen position: `__state().bell.rings` went from
0 to 1 and `swing` was non-zero for the next 4 s; the sound probe after the click read RMS -26 dBFS
and not a single tone; fps 50 against the baseline 50; 0 console errors.
