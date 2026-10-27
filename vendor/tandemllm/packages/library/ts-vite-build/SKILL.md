---
name: ts-vite-build
description: Setting up TypeScript in a frontend or Node project with Vite or esbuild - tsconfig strict mode, types for props and API data, type narrowing, building and type-checking before done.
---
# TypeScript with Vite or esbuild

Rules:
- Scaffold: `npm create vite@latest app -- --template vanilla-ts` (or `react-ts`, `svelte-ts`). Node libs: `tsc` for types, `esbuild` for the bundle.
- `strict: true` in tsconfig. Fix errors, never `// @ts-ignore` and never `any` for API data; write an interface and a type guard for what comes over the wire.
- Type-check separately from the build: Vite and esbuild strip types and do not check. Run `npx tsc --noEmit` before "done".
- Narrow with `typeof`, `in`, discriminated unions (`kind: 'a' | 'b'`), and `satisfies` for config objects.
- Unknown input is `unknown`; parse it into a typed value (a guard or a schema library) at the boundary, then types hold inside.

What goes wrong:
- The dev server works and `tsc` fails: nobody ran `tsc --noEmit`.
- `Object is possibly undefined` fixed with `!` everywhere; handle the undefined case.
- Importing JSON or assets without a type; add `/// <reference types="vite/client" />` or `resolveJsonModule`.

Check: `npx tsc --noEmit` exit 0; `npm run build` exit 0 with output sizes; open the built `dist/` with `npx vite preview` and read the console.

Done when:
- [ ] `tsc --noEmit` output in the result: 0 errors.
- [ ] Build output listed (file names and kB).
- [ ] Preview or run of the built output with console 0 errors (or the Node binary runs on a real input).

## Reference

tsconfig for Vite:
```json
{ "compilerOptions": { "target": "ES2022", "module": "ESNext", "moduleResolution": "Bundler", "strict": true, "noUncheckedIndexedAccess": true, "skipLibCheck": true, "types": ["vite/client"] }, "include": ["src"] }
```
Node library with esbuild: `npx esbuild src/index.ts --bundle --platform=node --format=esm --outfile=dist/index.js` and `tsc --noEmit -p .` for types.

Type guard for API data:
```ts
interface Item { id: string; text: string; done: boolean }
function isItem(x: unknown): x is Item {
  return typeof x === 'object' && x !== null && typeof (x as Item).id === 'string' && typeof (x as Item).text === 'string' && typeof (x as Item).done === 'boolean';
}
const data: unknown = await res.json();
const items: Item[] = Array.isArray(data) ? data.filter(isItem) : [];
```
Discriminated union:
```ts
type Result = { ok: true; value: number } | { ok: false; error: string };
function show(r: Result) { return r.ok ? String(r.value) : r.error; }
```
Alias in both places: tsconfig `"paths": { "@/*": ["./src/*"] }`, vite.config.ts `resolve: { alias: { '@': '/src' } }`.

More rules:
- Keep `allowJs: false` in new projects; convert files instead of mixing.

More pitfalls:
- ESM/CJS mismatch in Node: `"type": "module"` in package.json and `"module": "NodeNext"` in tsconfig, or neither.
- Path aliases set in tsconfig but not in Vite (`resolve.alias`), or the reverse.
