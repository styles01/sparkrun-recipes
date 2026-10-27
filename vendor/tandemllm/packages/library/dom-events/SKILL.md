---
name: dom-events
description: Vanilla JavaScript in the browser without a framework - DOM queries, event listeners, delegation, rendering state to HTML, and avoiding leaks.
---
# Vanilla JS: DOM and events

Rules:
- Keep state in one object. Render from state to DOM in one `render()` function. Events change state, then call `render()`.
- `addEventListener`, never `onclick=` in HTML. One listener on a parent for lists (delegation with `e.target.closest('[data-id]')`).
- Query once, keep references (`const els = { list: document.querySelector('#list') }`).
- Build DOM with `textContent` and `createElement`. `innerHTML` only with a template you wrote and no user data in it.
- Script at the end of body or `type="module"`; do not touch the DOM before it exists.

What goes wrong:
- Listeners added inside render; every render adds another. Attach once, outside.
- `querySelector` returns null because the id differs or the script ran too early; read the console.
- `this` inside an arrow function is not the element; use `e.currentTarget`.

Check: open the page, read the console (zero errors), click the main control with the browser tool, read the state object and the DOM text after the click.

Done when:
- [ ] A real click or key through the browser tool changes the state and the DOM; both read back and shown.
- [ ] Console errors: 0 after load and after the interaction.
- [ ] Adding 3 items then removing one leaves the listener count stable (one listener on the parent).

## Reference

Pattern:
```js
const state = { items: [], filter: '' };
const els = { list: document.querySelector('#list'), form: document.querySelector('#add') };
function render() {
  els.list.replaceChildren(...state.items.map(it => {
    const li = document.createElement('li');
    li.dataset.id = it.id; li.textContent = it.text; return li;
  }));
}
els.form.addEventListener('submit', e => {
  e.preventDefault();
  const text = new FormData(e.currentTarget).get('text').trim();
  if (!text) return;
  state.items.push({ id: crypto.randomUUID(), text });
  e.currentTarget.reset(); render();
});
els.list.addEventListener('click', e => {
  const li = e.target.closest('li[data-id]'); if (!li) return;
  state.items = state.items.filter(i => i.id !== li.dataset.id); render();
});
window.__state = () => ({ tandem_state: 1, items: state.items.length, filter: state.filter });
render();
```
Throttle to a frame:
```js
let pending = false;
window.addEventListener('scroll', () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; onScroll(); }); });
```
Custom events between parts: `el.dispatchEvent(new CustomEvent('item:add', { detail, bubbles: true }))`.

More rules:
- `preventDefault()` on form submit and on links you handle yourself.

More pitfalls:
- Scroll and resize handlers that run heavy work; throttle with `requestAnimationFrame`.
- Timers kept after an element is removed; `clearInterval` on teardown.
