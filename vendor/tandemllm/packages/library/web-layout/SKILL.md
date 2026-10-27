---
name: web-layout
description: Building an HTML/CSS page layout with flexbox and grid that holds at 1280 px desktop and 390 px phone width - headers, sidebars, cards, columns, spacing.
---
# HTML/CSS layout

Rules:
- Start with semantic boxes: header, nav, main, aside, footer. Style the boxes, not the text.
- One layout tool per level: grid for the page (areas, columns), flex for rows inside (nav items, card footer).
- Mobile first. Write the 390 px layout as the default, add `@media (min-width: 768px)` and `(min-width: 1200px)` for wider.
- Size with `rem`, `%`, `minmax()`, `clamp()`. Fixed `px` widths only for icons and borders.
- `box-sizing: border-box` on everything. `max-width: 100%` on images.
- Never position: absolute for layout; only for badges, tooltips, overlays.

What goes wrong:
- Horizontal scroll at 390 px: a fixed width, a long unbroken string, or a grid with `1fr 1fr 1fr` and no media query.
- Flex children overflow: add `min-width: 0` to the child that holds text.
- Centering with margins and guesses instead of `place-items: center` or `margin-inline: auto`.
- Heights set by hand; content then clips. Let height come from content.

Check: resize the viewport to 1280 and 390, screenshot both, `document.documentElement.scrollWidth <= window.innerWidth` must hold at both.

Done when:
- [ ] Screenshots at 1280 px and 390 px attached, nothing cut or overlapping.
- [ ] No horizontal scroll at 390 (scrollWidth read with evaluate_script).
- [ ] Console has zero errors after load.

## Reference

Page grid:
```css
.page { display: grid; grid-template-areas: "header" "main" "aside" "footer"; gap: 1rem; }
@media (min-width: 768px) {
  .page { grid-template-columns: 1fr 18rem; grid-template-areas: "header header" "main aside" "footer footer"; }
}
```
Card row that wraps:
```css
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem; }
```
Row with text that must not overflow:
```css
.row { display: flex; gap: .5rem; align-items: center; }
.row > .label { flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
```
Fluid type: `font-size: clamp(1rem, 0.9rem + 0.5vw, 1.25rem);`

Resize and check in the browser tool: `resize_page` to 1280x800, screenshot; `resize_page` to 390x844, screenshot; `evaluate_script`: `({w: innerWidth, sw: document.documentElement.scrollWidth})`.

More pitfalls:
- `vh` on phones jumps with the address bar; use `dvh` or avoid full-height.
