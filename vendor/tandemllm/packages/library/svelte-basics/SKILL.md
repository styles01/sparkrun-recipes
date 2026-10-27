---
name: svelte-basics
description: Building a small app with Svelte 5 - runes ($state, $derived, $effect), components and props, events, each blocks with keys, stores versus runes, and the Vite project setup.
---
# Svelte 5 basics

Rules:
- Project: `npm create vite@latest app -- --template svelte` (or `svelte-ts`). One component per file, `App.svelte` at the root.
- State with runes: `let count = $state(0)`; derived values with `$derived(expr)`; side effects with `$effect(() => ...)` and a returned cleanup.
- Props: `let { title, items = [] } = $props();`. Children with `{@render children()}`.
- Events are plain attributes: `onclick={() => count++}`. Components pass callbacks as props.
- `{#each items as item (item.id)}` with a key for anything that reorders or is removed.

What goes wrong:
- Reading a `$state` value into a plain `let` copies it once; it never updates. Use `$derived` or read it where needed.
- `$effect` that writes the state it reads loops; compute with `$derived` instead.
- Missing key in `each`; rows keep the wrong input values after a delete.

Check: `npm run dev` in the background, open the URL, console 0 errors, perform the main action through the browser tool, read the DOM; `npm run build` exits 0.

Done when:
- [ ] Dev server log line with the URL, page opened, console 0 errors.
- [ ] A real input through the browser tool updates the rendered value; both read back.
- [ ] `npm run build` exit code 0 with the output size line in the result.

## Reference

`src/App.svelte`:
```svelte
<script>
  import Row from './Row.svelte';
  let items = $state([]);
  let q = $state('');
  let visible = $derived(items.filter(i => i.text.toLowerCase().includes(q.toLowerCase())));
  let text = $state('');
  function add() { if (!text.trim()) return; items.push({ id: crypto.randomUUID(), text }); text = ''; }
  $effect(() => { window.__state = () => ({ tandem_state: 1, items: items.length, q }); });
</script>
<input aria-label="Filter" bind:value={q} />
<form onsubmit={e => { e.preventDefault(); add(); }}>
  <input aria-label="New item" bind:value={text} /><button>Add</button>
</form>
<ul>{#each visible as item (item.id)}<Row {item} onremove={() => items = items.filter(i => i.id !== item.id)} />{/each}</ul>
```
`src/Row.svelte`:
```svelte
<script> let { item, onremove } = $props(); </script>
<li>{item.text} <button onclick={onremove}>x</button></li>
```
Run: `npm install && (npm run dev -- --host --port 5173 > {tandem_scratch}/run.log 2>&1 &)` then `sleep 2; tail -n 20 {tandem_scratch}/run.log`.
Fetch on mount: `$effect(() => { const ctl = new AbortController(); fetch('/api/items', { signal: ctl.signal }).then(r => r.json()).then(d => items = d); return () => ctl.abort(); });`

More rules:
- Deep state: `$state` objects and arrays are reactive on mutation (`items.push(x)` works); replace the array when the source changes entirely.

More pitfalls:
- Stores (`writable`) and runes mixed in one component; pick runes in Svelte 5 files.
- `bind:value` on a derived value; bind only to state.
