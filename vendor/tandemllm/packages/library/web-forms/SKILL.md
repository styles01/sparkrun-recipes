---
name: web-forms
description: HTML forms with client-side validation and submission - input types, required and pattern, custom messages, error display, submit via fetch, and server-side revalidation.
---
# Forms and validation

Rules:
- Use the right input type (`email`, `number`, `date`, `url`) with `required`, `min`, `max`, `pattern`, `maxlength`. The browser validates first.
- Every field has a `label for`. Errors appear next to the field in text, with `aria-describedby`, not only in a color.
- Validate on submit and on `blur`, not on every keystroke for the first entry.
- `novalidate` on the form only when you render your own messages; then call `checkValidity()` and `setCustomValidity()` yourself.
- Read values with `new FormData(form)`. Disable the submit button while a request is in flight. Show success or the server's error.

What goes wrong:
- Submit reloads the page: `preventDefault()` missing.
- Errors shown but focus not moved: focus the first invalid field.
- `type="number"` rejects decimals: set `step="any"`.

Check: through the browser tool, submit empty (error text visible, focus on the first field), fill valid values, submit, read the success text and the request body in the network log.

Done when:
- [ ] Empty submit shows the error text and focuses the first invalid field (activeElement read).
- [ ] A valid submit through real input produces the request with the expected body (network or server log line).
- [ ] Screenshots at 1280 and 390; the form is usable at 390 with no cut fields.

## Reference

```html
<form id="signup" novalidate>
  <label for="email">Email</label>
  <input id="email" name="email" type="email" required aria-describedby="email-err">
  <small id="email-err" class="err" aria-live="polite"></small>
  <label for="age">Age</label>
  <input id="age" name="age" type="number" min="13" max="120" step="1" required>
  <button type="submit">Sign up</button>
  <p id="status" aria-live="polite"></p>
</form>
```
```js
const form = document.querySelector('#signup'), status = document.querySelector('#status');
function showErrors() {
  let first = null;
  for (const el of form.elements) {
    if (!el.name) continue;
    const err = document.querySelector(`#${el.id}-err`);
    const ok = el.checkValidity();
    if (err) err.textContent = ok ? '' : (el.validity.valueMissing ? 'Required' : el.validationMessage);
    el.setAttribute('aria-invalid', String(!ok));
    if (!ok && !first) first = el;
  }
  first?.focus(); return !first;
}
form.addEventListener('submit', async e => {
  e.preventDefault(); if (!showErrors()) return;
  const btn = form.querySelector('button'); btn.disabled = true; status.textContent = 'Sending';
  try {
    const r = await fetch('/api/signup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(Object.fromEntries(new FormData(form))) });
    status.textContent = r.ok ? 'Done' : `Error ${r.status}: ${(await r.json()).error ?? ''}`;
  } catch (err) { status.textContent = 'Network error'; } finally { btn.disabled = false; }
});
form.addEventListener('focusout', e => { if (e.target.name) showErrors(); });
```

More rules:
- The server validates again. The client check is for the user, not for safety.

More pitfalls:
- Error state never clears after the user fixes the value.
- A double submit sends twice; disable the button until the response.
