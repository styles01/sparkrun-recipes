---
name: web-a11y
description: Accessibility basics for any web page or app - semantic elements, labels, keyboard focus, contrast, ARIA only when needed, screen-reader names.
---
# Accessibility basics

Rules:
- Use the element that does the job: `button` for actions, `a href` for navigation, `input` + `label for`, `h1..h3` in order, `ul` for lists, `table` for tables.
- Every control has a name: label text, `aria-label`, or visible text. Every image has `alt` (empty `alt=""` when decorative).
- Keyboard first: Tab reaches every control in reading order; Enter/Space activate; Escape closes dialogs. Never remove `outline` without a visible replacement.
- Contrast: text 4.5:1 against its background, large text and icons 3:1.
- ARIA is the last resort. No `role="button"` on a div when a `button` works. `aria-live="polite"` for status text that changes.

What goes wrong:
- Click handlers on `div` and `span`: not focusable, not announced, no keyboard.
- Placeholder used as the label; it vanishes on input.
- Focus trapped or lost after a modal opens or closes; return focus to the opener.

Check: Tab through the page in the browser and read `document.activeElement` after each step; run the a11y audit tool if available; read contrast of the two main text colors.

Done when:
- [ ] A Tab walk reaches every control, logged as the sequence of activeElement names.
- [ ] Every form field has a label linked by `for`/`id` (query `input:not([id])` returns 0).
- [ ] Images with missing `alt` return 0 (`img:not([alt])`).

## Reference

Quick audit in the console:
```js
({
  noAlt: document.querySelectorAll('img:not([alt])').length,
  unlabeled: [...document.querySelectorAll('input,select,textarea')].filter(e => !e.labels?.length && !e.getAttribute('aria-label')).length,
  divButtons: [...document.querySelectorAll('div[onclick],span[onclick]')].length,
  h1: document.querySelectorAll('h1').length,
  lang: document.documentElement.lang
})
```
Focus walk: press Tab with the key tool 10 times, after each read `document.activeElement.tagName + ' ' + (document.activeElement.textContent||document.activeElement.ariaLabel||'').trim().slice(0,30)`.

Visible focus style:
```css
:focus-visible { outline: 3px solid #2563eb; outline-offset: 2px; }
```
Skip link: `<a class="skip" href="#main">Skip to content</a>`, positioned off-screen until focused.

Dialog: use `<dialog>` with `showModal()`; it traps focus and handles Escape. On close, call `opener.focus()`.

More rules:
- One `h1` per page. `lang` on `html`. `title` set.

More pitfalls:
- Color as the only signal (red = error); add text or an icon.
- `tabindex` larger than 0; it reorders focus unpredictably.
