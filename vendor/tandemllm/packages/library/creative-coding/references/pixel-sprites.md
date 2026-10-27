# Pixel sprites drawn in code

Read with `package_read("references/pixel-sprites.md")`. No image files: a sprite is a list of
strings, one character per pixel, drawn once to an offscreen canvas and then blitted.

```js
const PAL = { '.': null, 'k': '#1b1b1b', 'o': '#e06a2d', 'w': '#f2f2ef', 'g': '#3f7a4f', 'y': '#f7c67a' };
const FOX = [
  '..oo....oo..',
  '..ooo..ooo..',
  '.oooooooooo.',
  '.ookoooookoo',
  '.oooooooooo.',
  '..oowwwwoo..',
  '...oowwoo...',
  '....oooo....',
];
function sprite(rows, scale = 4){
  const c = document.createElement('canvas'); c.width = rows[0].length * scale; c.height = rows.length * scale;
  const g = c.getContext('2d');
  rows.forEach((row, y) => [...row].forEach((ch, x) => { if (PAL[ch]) { g.fillStyle = PAL[ch]; g.fillRect(x * scale, y * scale, scale, scale); } }));
  return c;
}
const fox = sprite(FOX);
ctx.imageSmoothingEnabled = false;             // keep the pixels crisp
ctx.drawImage(fox, x, y);                      // or drawImage(fox, x, y, w * 2, h * 2) to scale
```

Animation: two or three frames as separate string lists; pick `frames[Math.floor(t * 6) % frames.length]`.
Flip: `ctx.save(); ctx.translate(x + w, y); ctx.scale(-1, 1); ctx.drawImage(fox, 0, 0); ctx.restore();`.

Rules:
- Width 8 to 24 pixels; 3 to 6 colours; an outline colour (`k`) makes a sprite readable on any background.
- Build every sprite once at start, never inside the frame loop.
- A creature is a sprite plus a position, a velocity, a facing and a state (`idle`, `walk`, `startle`); change the frame set with the state.

Tile maps: a string per row, one character per tile; draw tiles the same way to one offscreen canvas and blit it per frame (one `drawImage` for the whole map).
