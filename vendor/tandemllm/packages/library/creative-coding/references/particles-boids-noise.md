# Particles, boids, noise

Read with `package_read("references/particles-boids-noise.md")`, or copy from here.

## A particle system (typed arrays, 2,000 particles at 60 fps)

```js
const N = 2000;
const px = new Float32Array(N), py = new Float32Array(N), vx = new Float32Array(N), vy = new Float32Array(N), life = new Float32Array(N);
function spawn(i){ px[i] = rand() * W; py[i] = -10; vx[i] = (rand() - 0.5) * 40; vy[i] = 60 + rand() * 80; life[i] = 1; }
for (let i = 0; i < N; i++) { spawn(i); py[i] = rand() * H; }
function update(dt){
  for (let i = 0; i < N; i++) {
    vx[i] += Math.sin(py[i] * 0.01 + i) * 30 * dt;      // a little wind
    px[i] += vx[i] * dt; py[i] += vy[i] * dt;
    if (py[i] > H + 10) spawn(i);
  }
}
function draw(){
  ctx.fillStyle = '#0f1419'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#e0a458';
  for (let i = 0; i < N; i++) ctx.fillRect(px[i], py[i], 3, 3);   // fillRect is far cheaper than arc()
}
```

Rules: one draw call shape per particle (`fillRect`), no `beginPath()`+`arc()` for thousands, no
object allocation inside `update`, no `ctx.shadowBlur` (10 times slower).

## Boids (separation, alignment, cohesion)

```js
const B = 120, bx = new Float32Array(B), by = new Float32Array(B), bvx = new Float32Array(B), bvy = new Float32Array(B);
for (let i = 0; i < B; i++) { bx[i] = rand() * W; by[i] = rand() * H; bvx[i] = (rand() - 0.5) * 100; bvy[i] = (rand() - 0.5) * 100; }
function boids(dt){
  const R = 60, SPEED = 120;
  for (let i = 0; i < B; i++) {
    let sx = 0, sy = 0, ax = 0, ay = 0, cx = 0, cy = 0, n = 0;
    for (let j = 0; j < B; j++) {
      if (i === j) continue;
      const dx = bx[j] - bx[i], dy = by[j] - by[i], d2 = dx * dx + dy * dy;
      if (d2 > R * R) continue;
      n++; cx += bx[j]; cy += by[j]; ax += bvx[j]; ay += bvy[j];
      if (d2 < 400) { sx -= dx; sy -= dy; }                 // separation inside 20 px
    }
    if (n) { ax = ax / n - bvx[i]; ay = ay / n - bvy[i]; cx = cx / n - bx[i]; cy = cy / n - by[i]; }
    bvx[i] += (sx * 1.5 + ax * 0.05 + cx * 0.01) * dt * 60;
    bvy[i] += (sy * 1.5 + ay * 0.05 + cy * 0.01) * dt * 60;
    const s = Math.hypot(bvx[i], bvy[i]) || 1; bvx[i] *= SPEED / s; bvy[i] *= SPEED / s;   // constant speed
    bx[i] = (bx[i] + bvx[i] * dt + W) % W; by[i] = (by[i] + bvy[i] * dt + H) % H;          // wrap
  }
}
```

Draw a boid as a triangle pointing along its velocity: `ctx.translate(bx[i], by[i]); ctx.rotate(Math.atan2(bvy[i], bvx[i]));`.
120 boids with the O(n²) loop is fine (14,400 pairs); above 400 boids use a grid of cells.

## Value noise and fBm (no library)

```js
function noise2(x, y){                               // smooth value noise, seeded by hash
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
  const h = (a, b) => { let n = Math.imul(a, 374761393) ^ Math.imul(b, 668265263) ^ (seed | 0); n = Math.imul(n ^ n >>> 13, 1274126177); return ((n ^ n >>> 16) >>> 0) / 4294967296; };
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = h(xi, yi), b = h(xi + 1, yi), c = h(xi, yi + 1), d = h(xi + 1, yi + 1);
  return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v;    // 0..1
}
function fbm(x, y, oct = 4){ let s = 0, amp = 0.5, f = 1, norm = 0; for (let i = 0; i < oct; i++) { s += amp * noise2(x * f, y * f); norm += amp; amp *= 0.5; f *= 2; } return s / norm; }
```

Uses: terrain height `fbm(x * 0.01, 0) * 300`, wind field `noise2(x * 0.005, t * 0.1) - 0.5`,
cloud alpha `fbm(x * 0.02, y * 0.02, 5)`. For simplex noise with the import map:
`"simplex-noise": "https://cdn.jsdelivr.net/npm/simplex-noise@4.0.3/+esm"` and
`createNoise2D(rand)`.

## Springs and a soft physics step

```js
function spring(pos, vel, target, k = 40, damp = 6, dt){ const a = (target - pos) * k - vel * damp; vel += a * dt; pos += vel * dt; return [pos, vel]; }
```
A fixed step keeps physics stable: run `update` in steps of 1/120 s until the frame's `dt` is used up.
