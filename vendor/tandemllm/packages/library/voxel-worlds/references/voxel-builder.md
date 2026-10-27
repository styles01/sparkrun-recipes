# Voxel builder

Read with `package_read("references/voxel-builder.md")`.

## The store

```js
// js/core/voxels.js
import { GRID, GC } from './world.js';          // GRID 256, GC 128 (the grid is -128..127)
export const heights = new Float32Array(GRID * GRID);
export const hIdx = (cx, cz) => (cz + GC) * GRID + (cx + GC);
export const voxels = new Map();                 // 'x,y,z' -> {x, y, z, c, b}
export function setVoxel(x, y, z, c, b){ voxels.set(x + ',' + y + ',' + z, { x, y, z, c, b }); }
export function groundTop(x, z){ if (x < -GC || x >= GC || z < -GC || z >= GC) return 0; return Math.round(heights[hIdx(x, z)]); }
```

Buckets `b`: 0 terrain (no shadow cast, receives), 1 built (casts and receives), 2 trees (casts),
3 water edge, 4 glowing (`MeshBasicMaterial`, colour x1.9, for lanterns and gold), 5 water
(transparent, opacity 0.72, a 1.7-tall box pushed down 0.65 so the surface sits at the voxel top).
Setting a voxel twice keeps the last one: build the terrain first, then carve (a lake sets its
cells to water), then build on top.

## Heights

```js
export function generateHeights(){
  for (let cz = -GC; cz < GC; cz++) for (let cx = -GC; cx < GC; cx++) {
    let base = 3.0 + fbm(cx * 0.019, cz * 0.019, 4) * 2.1 + fbm(cx * 0.062 + 40, cz * 0.062 - 17, 3) * 0.7;   // rolling ground
    const mount = smoothstep(0.04, 0.62, ridged(cx * 0.013 - 9, cz * 0.013 + 31, 4));                           // a ridge
    base += mount * smoothstep(-24, 56, -cz) * 28;                                                                 // the ridge to the north
    base = lerp(base, 0.35, smoothstep(-8, 2.5, islandSDF(cx, cz)));                                              // fall to sea outside the coast
    const rp = Math.hypot(cx, cz);
    if (rp < PLAZA_R) base = PLAZA_TOP; else if (rp < PLAZA_R + 10) base = lerp(PLAZA_TOP, base, smoothstep(PLAZA_R, PLAZA_R + 10, rp));   // a flat plaza
    const ls = lakeSigned(cx, cz);
    if (ls < -0.06) base = 1.0; else if (ls < 0) base = Math.min(base, 1.6); else if (ls < 0.05) base = Math.max(base, 3.0);   // a lake with a rim
    heights[hIdx(cx, cz)] = base;
  }
}
```

`islandSDF(x, z) = hypot(x, z) - R * coastR(atan2(z, x))` with `coastR(a) = 1 + 0.13 sin(3a + 0.9) + 0.08 sin(5a + 2.1) + 0.05 sin(8a + 0.3)`: a wobbly coast from three sines. The same shape makes a lake (`lakeSigned`) and extra isles with their own centres.

## Columns

```js
export function buildTerrain(){
  for (let cz = -GC; cz < GC; cz++) for (let cx = -GC; cx < GC; cx++) {
    const top = Math.round(heights[hIdx(cx, cz)]); if (top <= 0) continue;
    let ground = Infinity;                                   // fill down only to the lowest neighbour: hidden voxels cost fps
    for (const [nx, nz] of [[cx + 1, cz], [cx - 1, cz], [cx, cz + 1], [cx, cz - 1]]) ground = Math.min(ground, groundTop(nx, nz));
    ground = Math.max(ground - 1, 0); if (top - ground > 26) ground = top - 26;
    for (let y = ground; y <= top; y++) setVoxel(cx, y, cz, y === top ? topColor(cx, cz, top) : subColor(y, top), 0);
  }
}
```

`topColor`: snow above 22, rock above 16, sand near the coast and the lake rim, else grass in
three greens by low-frequency noise with 1.6 % flowers by `hash01(x, z)`. `subColor`: dirt within 4
of the top, rock below, dark rock under 6. This fill rule is why a 256x256 island is about 95k
voxels and not 1.5 million.

## A landmark from loops

```js
function hollowBox(x0, y0, z0, x1, y1, z1, c){ for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) if (x === x0 || x === x1 || z === z0 || z === z1) setVoxel(x, y, z, c, 1); }
export function buildPagoda(){
  for (let i = 0; i < TIERS; i++) {                                   // 7 tiers, 5 apart, radius 15 - 2 i
    const yb = PAGODA_BASE + TIER_SPACING * i, r = TIER_R0 - 2 * i, wallR = r - 1;
    hollowBox(-wallR, yb + 1, -wallR, wallR, yb + 3, wallR, P.plaster);                       // walls
    for (const o of [-6, -5, -4, 4, 5, 6]) if (wallR >= 6) for (const [x, z] of [[o, wallR], [o, -wallR], [wallR, o], [-wallR, o]]) setVoxel(x, yb + 2, z, P.window, 1);
    for (const [he, py, pc] of [[r + 3, yb + 4, P.jade], [r + 2, yb + 5, P.jade], [r + 1, yb + 6, P.jadeLight]])      // roof plates
      for (let dz = -he; dz <= he; dz++) for (let dx = -he; dx <= he; dx++) setVoxel(dx, py, dz, pc, 1);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) setVoxel(sx * (r + 5), yb + 6, sz * (r + 5), P.gold, 4);   // glowing corner curls
  }
  for (let dz = -3; dz <= 3; dz++) for (let dx = -3; dx <= 3; dx++) if (Math.hypot(dx, dz) <= 3) setVoxel(dx, 44, dz, P.gold, 4);   // the finial
}
```

A lantern: a 3x3 stone base at `groundTop - 1`, a two-voxel post, a 3x3 cap, one glowing voxel
(bucket 4) inside, a 5x5 roof with the corners cut; `withLight` adds a `PointLight(0xffd08a, 0, 14)`
whose intensity day-night sets. A tree: a trunk of 3 to 5 wood voxels and a sphere of leaves
`if (dx*dx + dy*dy + dz*dz <= r*r && hash3(x, y, z) < 0.85)`.

## The palette

Name every colour once in `world.js` (`P.grass1 0x4e7d37 ... P.plaster 0xe9e1d2, P.jade 0x2f8f7e,
P.gold 0xffcf6a, P.red 0xb7412f, P.water 0x2f6f9e, P.lanternGlow 0xffd08a`); builders use the
names. Three greens, two sands, two rocks, four blossoms, and the pagoda's plaster, jade, red and
gold read as one place.
