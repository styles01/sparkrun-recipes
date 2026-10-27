---
name: react-basics
description: Writing React components with hooks - useState, useEffect, derived values, lists with keys, forms, lifting state, and the common performance and re-render pitfalls.
---
# React with hooks

Rules:
- State is the minimum that cannot be computed. Derived values (filtered lists, totals) are computed in render, not stored.
- State lives in the lowest common parent. Pass data down as props, changes up as callbacks.
- `useEffect` only for syncing with the outside (fetch, subscriptions, DOM APIs). List every dependency. Return a cleanup. No effect for "when X changes set Y"; compute Y instead.
- Lists: stable `key` from the data id, never the index when items reorder or are removed.
- Never mutate state. Copy: `setItems(items => items.map(...))`, `setForm(f => ({ ...f, name }))`.

What goes wrong:
- Infinite loop: an effect sets state that is in its own dependencies, or an object literal in the deps array.
- Stale closure: a timer or listener reads an old state value; use the functional setter or a ref.
- Double fetch in dev StrictMode; abort in cleanup and ignore the first.

Check: render the component, perform the user action through the browser tool, read the DOM text; open the React profiler or log render counts to confirm a list of 100 items re-renders only the changed row.

Done when:
- [ ] A real click or input changes the rendered output; before and after text read from the DOM.
- [ ] Console has 0 errors and 0 warnings (keys, deps) after the interaction.
- [ ] Screenshots at 1280 and 390.

## Reference

```jsx
import { useState, useEffect, useMemo } from 'react';
export function Items({ api }) {
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [status, setStatus] = useState('loading');
  useEffect(() => {
    const ctl = new AbortController();
    api('/items', { signal: ctl.signal }).then(d => { setItems(d); setStatus('ready'); }).catch(e => { if (e.name !== 'AbortError') setStatus('error: ' + e.message); });
    return () => ctl.abort();
  }, [api]);
  const visible = useMemo(() => items.filter(i => i.text.toLowerCase().includes(q.toLowerCase())), [items, q]);
  if (status !== 'ready') return <p role="status">{status}</p>;
  return (
    <>
      <input aria-label="Filter" value={q} onChange={e => setQ(e.target.value)} />
      <ul>{visible.map(i => <Row key={i.id} item={i} onRemove={() => setItems(xs => xs.filter(x => x.id !== i.id))} />)}</ul>
    </>
  );
}
const Row = React.memo(function Row({ item, onRemove }) { return <li>{item.text} <button onClick={onRemove}>x</button></li>; });
```
Form by name: `const onChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));`
Ref for the latest value in a timer: `const latest = useRef(count); latest.current = count;` then read `latest.current` in the callback.
State dump for tests: expose `window.__state = () => ({ tandem_state: 1, items: items.length, q })` inside an effect.

More rules:
- Controlled inputs: `value` + `onChange`. One handler per form, keyed by `name`.

More pitfalls:
- A new function or object prop every render defeats `memo`; `useCallback`/`useMemo` only when a measured re-render is the problem.
- Conditional hook calls; hooks must run in the same order every render.
