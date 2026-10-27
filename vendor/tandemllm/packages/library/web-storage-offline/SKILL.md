---
name: web-storage-offline
description: Persisting browser state with localStorage, sessionStorage and IndexedDB, and a minimal service worker for offline caching - versioned keys, try/catch, quotas, cache strategy.
---
# Browser storage and offline

Rules:
- `localStorage` for small per-device conveniences (settings, a draft, under 100 KB). JSON-encode under one versioned key: `app:v2:settings`.
- Wrap every read and write in try/catch; private mode and blocked storage throw. The page must render with no storage at all.
- Validate what you read: parse, check the shape, fall back to defaults. Never trust the stored version to match the current code.
- IndexedDB for anything large or structured (records, blobs): one database, versioned schema in `onupgradeneeded`.

What goes wrong:
- Storing an object without `JSON.stringify` gives `"[object Object]"`.
- A changed shape crashes the old stored data on load; add a version and defaults.
- Service worker caches a stale `index.html` forever; version the cache name and delete old caches in `activate`.

Check: set a value through the UI, reload in the browser tool, read it back from the UI and from `localStorage.getItem`; with a service worker, go offline and load the page.

Done when:
- [ ] A value survives a reload: written through real input, read back after `navigate` to the same URL.
- [ ] With storage blocked or cleared the page still renders (test with `localStorage.clear()` then reload).
- [ ] Offline load (if a service worker is in scope) returns the shell; screenshot attached.

## Reference

Safe storage helper:
```js
const KEY = 'app:v2:settings', DEFAULTS = { theme: 'dark', volume: 0.3 };
function load() {
  try { const raw = localStorage.getItem(KEY); const o = raw ? JSON.parse(raw) : {}; return { ...DEFAULTS, ...(o && typeof o === 'object' ? o : {}) }; }
  catch { return { ...DEFAULTS }; }
}
let saveTimer;
function save(settings) { clearTimeout(saveTimer); saveTimer = setTimeout(() => { try { localStorage.setItem(KEY, JSON.stringify(settings)); } catch {} }, 200); }
```
Minimal service worker (`sw.js` at the site root):
```js
const CACHE = 'app-v3', SHELL = ['/', '/index.html', '/app.js', '/app.css'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/')) return; // network only
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));
});
```
Register: `if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js');`
IndexedDB wrapper: use the `idb` package (`openDB`, `db.put('store', value)`), or the raw API with `indexedDB.open(name, version)` and `onupgradeneeded` creating `createObjectStore('items', { keyPath: 'id' })`.

More rules:
- Service worker only when offline use is asked for. Register from the page, cache the app shell on install, serve cache-first for static files, network-first for API calls.
- Migrate on version bump: read old key, transform, write new key, delete old.

More pitfalls:
- Service worker registration fails on `http://` other than localhost; it needs https.
- Reading storage in a loop per frame; read once at start, write on change, debounced.
