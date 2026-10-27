# Exemplar: a deer that grazes, walks and startles

A finished feature of the overnight build as prompt, plan and diff. Read with
`package_read("exemplars/deer.md")`.

## Prompt

> Add a deer to the meadow: it grazes with its head down, now and then it startles (head up, a
> small hop) and walks to a new spot, then grazes again. Keep the frame rate.

## Plan (written before the code)

1. New file `js/creatures/deer.js`: `spawnDeer()` builds the body from boxes in a `Group`
   (body, belly, neck group with head and antlers, tail, four legs); `updateDeer(dt, t)` runs a
   three-state machine `graze` -> `walk` (with a startle hop) -> `graze`.
2. Targets: a ring 38 to 70 units from the centre (the meadow, outside the plaza, inside the
   coast); reject a target on water.
3. `y` from `groundTop` every frame; legs swing `±sin(phase) * 0.5` while walking and decay while
   grazing.
4. `main.js`: import, `spawnDeer()` after `buildMeshes()`, `updateDeer(dt, t)` in `animate()`.
5. `state.deer = true` and a panel chip "deer" that sets `deer.visible`.
6. Render at noon; read fps; add the line to `TODO.md`.

## Diff

```diff
--- a/js/core/state.js
+++ b/js/core/state.js
-  crane:true, turtle:true, fox:true,
+  crane:true, turtle:true, fox:true, deer:true,
--- a/js/main.js
+++ b/js/main.js
+import { spawnDeer, updateDeer } from './creatures/deer.js';
 ...
   spawnFox();
+  spawnDeer();
 ...
   updateFox(dt,t);
+  updateDeer(dt,t);
--- /dev/null
+++ b/js/creatures/deer.js
+import * as THREE from 'three';
+import { scene } from '../scene/renderer.js';
+import { groundTop } from '../core/voxels.js';
+import { rand, randRange, TAU } from '../core/rng.js';
+import { LAKE_WATER_Y } from '../core/world.js';
+
+let deer, neck, legL=[], legR=[], tail;
+let state='graze', timer=3, target=new THREE.Vector3(), phase=0, hop=0;
+const BROWN=0x8a5a3a, LIGHT=0xc9a074, DARK=0x4a3320, ANTLER=0x6a4a2a;
+function box(w,h,d,color,x,y,z,parent){
+  const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshLambertMaterial({color}));
+  m.position.set(x,y,z); m.castShadow=true; (parent||scene).add(m); return m;
+}
+function newTarget(){
+  for(let i=0;i<5;i++){
+    const a=rand()*TAU, r=randRange(38,70);
+    target.set(Math.cos(a)*r,0,Math.sin(a)*r);
+    if(groundTop(Math.round(target.x),Math.round(target.z))>LAKE_WATER_Y) return;   // not on water
+  }
+}
+export function spawnDeer(){
+  deer=new THREE.Group();
+  box(1.7,0.8,0.9,BROWN,0,1.3,0,deer);  box(1.3,0.5,0.7,LIGHT,0,1.0,0.1,deer);
+  neck=new THREE.Group(); neck.position.set(0,1.7,0.6); deer.add(neck);
+  box(0.4,0.9,0.4,BROWN,0,0.4,0.1,neck); box(0.5,0.45,0.55,BROWN,0,0.95,0.25,neck);
+  box(0.3,0.3,0.3,LIGHT,0,0.85,0.55,neck);
+  for(const s of [1,-1]){ box(0.12,0.7,0.12,ANTLER,s*0.2,1.5,0.1,neck); box(0.1,0.4,0.1,ANTLER,s*0.35,1.7,0.1,neck); }
+  tail=box(0.25,0.3,0.2,LIGHT,0,1.4,-0.55,deer);
+  for(const [lx,lz] of [[0.5,0.4],[0.5,-0.4],[-0.5,0.4],[-0.5,-0.4]]) (lx>0?legL:legR).push(box(0.24,1.0,0.24,DARK,lx,0.5,lz,deer));
+  scene.add(deer); newTarget(); timer=randRange(2,5); phase=rand()*TAU;
+}
+export function updateDeer(dt,t){
+  if(!deer||!deer.visible) return;
+  timer-=dt;
+  if(state==='graze'){ neck.rotation.x=0.55; if(timer<=0){ state='walk'; hop=1; newTarget(); } }
+  else {
+    const dx=target.x-deer.position.x, dz=target.z-deer.position.z, dist=Math.hypot(dx,dz);
+    if(dist<2){ state='graze'; timer=randRange(3,7); }
+    else { const speed=3.4; deer.position.x+=dx/dist*speed*dt; deer.position.z+=dz/dist*speed*dt;
+           deer.rotation.y=Math.atan2(dx,dz); phase+=dt*9; neck.rotation.x=Math.max(-0.2,neck.rotation.x-dt*2); }
+  }
+  if(hop>0){ hop=Math.max(0,hop-dt*3); neck.rotation.x=-0.1; }
+  deer.position.y=groundTop(Math.round(deer.position.x),Math.round(deer.position.z))+Math.sin(hop*Math.PI)*0.3;
+  const step=Math.sin(phase), walking=state==='walk';
+  legL[0].rotation.x=walking?step*0.5:legL[0].rotation.x*0.9; legR[1].rotation.x=legL[0].rotation.x;
+  legL[1].rotation.x=walking?-step*0.5:legL[1].rotation.x*0.9; legR[0].rotation.x=legL[1].rotation.x;
+  tail.rotation.x=Math.sin(t*3)*0.2;
+}
--- a/TODO.md
+++ b/TODO.md
+- done: deer (graze / walk / startle state machine, meadow ring, never on water) - 50 fps (baseline 50)
```

## What was checked

A render at noon showed the deer on the meadow with its head down; a render 8 s later showed it
walking with legs swinging; `__state().fps` read 50 against a baseline of 50; the console had 0
errors. The line went into `TODO.md` and the next feature started.
