# Creature state machines

Read with `package_read("references/creature-state-machines.md")`. A creature is a small group of
boxes, a state, a timer and a target. One file per creature with `spawnX()` and `updateX(dt, t)`.

## The box helper every creature uses

```js
function box(w, h, d, color, x, y, z, parent){
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshLambertMaterial({ color }));
  m.position.set(x, y, z); m.castShadow = true; (parent || scene).add(m); return m;
}
```

## A grazer (deer): graze, walk, startle

```js
let deer, neck, legL = [], legR = [], tail, state = 'graze', timer = 3, target = new THREE.Vector3(), phase = 0, hop = 0;
function newTarget(){ const a = rand() * TAU, r = randRange(38, 70); target.set(Math.cos(a) * r, 0, Math.sin(a) * r); }   // inside the meadow ring
export function spawnDeer(){
  deer = new THREE.Group();
  box(1.7, 0.8, 0.9, BROWN, 0, 1.3, 0, deer); box(1.3, 0.5, 0.7, LIGHT, 0, 1.0, 0.1, deer);          // body, belly
  neck = new THREE.Group(); neck.position.set(0, 1.7, 0.6); deer.add(neck);
  box(0.4, 0.9, 0.4, BROWN, 0, 0.4, 0.1, neck); box(0.5, 0.45, 0.55, BROWN, 0, 0.95, 0.25, neck);   // neck, head
  for (const s of [1, -1]) box(0.12, 0.7, 0.12, ANTLER, s * 0.2, 1.5, 0.1, neck);                     // antlers
  tail = box(0.25, 0.3, 0.2, LIGHT, 0, 1.4, -0.55, deer);
  for (const [lx, lz] of [[0.5, 0.4], [0.5, -0.4], [-0.5, 0.4], [-0.5, -0.4]]) (lx > 0 ? legL : legR).push(box(0.24, 1.0, 0.24, DARK, lx, 0.5, lz, deer));
  scene.add(deer); newTarget(); timer = randRange(2, 5); phase = rand() * TAU;
}
export function updateDeer(dt, t){
  if (!deer || !deer.visible) return;
  timer -= dt;
  if (state === 'graze') { neck.rotation.x = 0.55; if (timer <= 0) { state = 'walk'; hop = 1; newTarget(); } }   // startle: head up and a hop
  else {
    const dx = target.x - deer.position.x, dz = target.z - deer.position.z, dist = Math.hypot(dx, dz);
    if (dist < 2) { state = 'graze'; timer = randRange(3, 7); }
    else { const speed = 3.4; deer.position.x += dx / dist * speed * dt; deer.position.z += dz / dist * speed * dt; deer.rotation.y = Math.atan2(dx, dz); phase += dt * 9; neck.rotation.x = Math.max(-0.2, neck.rotation.x - dt * 2); }
  }
  if (hop > 0) { hop = Math.max(0, hop - dt * 3); neck.rotation.x = -0.1; }
  deer.position.y = groundTop(Math.round(deer.position.x), Math.round(deer.position.z)) + Math.sin(hop * Math.PI) * 0.3;   // always on the ground
  const step = Math.sin(phase), walking = state === 'walk';
  legL[0].rotation.x = walking ? step * 0.5 : legL[0].rotation.x * 0.9; legR[1].rotation.x = legL[0].rotation.x;
  legL[1].rotation.x = walking ? -step * 0.5 : legL[1].rotation.x * 0.9; legR[0].rotation.x = legL[1].rotation.x;
  tail.rotation.x = Math.sin(t * 3) * 0.2;
}
```

## The pattern, for any creature

| part | rule |
|-|-|
| states | 2 to 4 names (`idle`, `walk`, `startle`; `perch`, `turn`, `hoot`; `swim`, `leap`); one `state` string, one `timer` |
| transitions | on `timer <= 0` or on arrival (`dist < 2`); a new target from `rand()` inside the creature's area (a ring, the lake ellipse, the plaza radius) |
| movement | `speed * dt` toward the target, `rotation.y = atan2(dx, dz)`; `y` from `groundTop` for walkers, a fixed water level for swimmers, a `sin` bob for flyers |
| animation | `phase += dt * k`; legs `±sin(phase) * 0.5`, wings `sin(phase) * 0.8`, tail `sin(t * 3) * 0.2`; decay toward 0 when idle (`*= 0.9`) |
| gating | `if (!x || !x.visible) return` first; `visible = state.deer` from the panel toggle |
| many of a kind | one `InstancedMesh` and a data array (`koiData[i] = {a, r, speed, phase}`), matrices composed per frame, `instanceMatrix.needsUpdate = true` |
| sound | an event sound at a state change (`playChime()` on a leap, `hoot()` on `hoot`), never per frame |

## Koi, a swimmer with instancing (14 fish)

```js
koi = new THREE.InstancedMesh(new THREE.BoxGeometry(0.55, 0.6, 1.55), new THREE.MeshLambertMaterial({ color: 0xffffff }), 14);
for (let i = 0; i < 14; i++) { koiData.push({ a: rand() * TAU, f: randRange(0.22, 0.74), w: (rand() < 0.5 ? -1 : 1) * randRange(0.3, 0.55), phi: rand() * TAU }); koi.setColorAt(i, col.set(pick(KOI_COLORS))); }
// per frame: a += w * dt; x = LAKE.x + cos(a) * LAKE.rx * f; z = LAKE.z + sin(a) * LAKE.rz * f; y = LAKE_WATER_Y + 0.2 + sin(t * 2 + phi) * 0.1
// m.compose(pos, quatFromHeading(a + sign(w) * PI/2), scale); koi.setMatrixAt(i, m); koi.instanceMatrix.needsUpdate = true
```

Feeding: a click on the lake sets `food = {x, z}`; every fish sets `seeking = true` and steers
toward it for 6 s, then back to its circle.

## What went wrong in the run, and the fix

- A creature floated at dusk: `y` was set once at spawn. Fix: `groundTop` every frame.
- Legs kept swinging while grazing: no decay. Fix: `rotation.x *= 0.9` in the idle branch.
- A walker crossed the lake: the target ring overlapped the water. Fix: reject a target whose
  `groundTop` is at the water level and pick again (at most 5 tries).
