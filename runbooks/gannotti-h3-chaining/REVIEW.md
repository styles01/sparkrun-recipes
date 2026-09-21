# H3 long-form capture — review

Review of `smfworks/h3-longform-capture` as shipped on `main` (`5dccc3b`) and the live demo [h3-longform-capture.vercel.app](https://h3-longform-capture.vercel.app). No product code was changed.

Verified: `cd app && npm test` (30/30 pass) and `npm run build` (tsc + Vite 8, clean). Live SPA loads, Vercel rewrite on `/does-not-exist` still serves the app, first visit is 9/9 green with the Sigils sample.

This app is a client-side form over a generate gate. If the checklist can go green while the pack is still the thing Sigils got wrong, the product has failed even if the zip shape matches `templates/`.

---

## Executive summary

The SPA is a real pack builder: six steps, nine README gates, markdown zip in the `templates/` filename shape, Sigils sample, `localStorage` autosave, no backend, no MP4s. Tests lock template headings and a handful of refuse-paths. Build is clean. Vercel hosting of `app/` works.

The gate is not honest enough to be the thing you trust before a Spark night. `evaluateProps` treats any non-empty overall/haft string as pinned, so the public sample is 9/9 green with haft = `none — not in public example` and the UI prints **Overall vs haft length pinned**. `evaluateEditList` only requires join + one camera verb + a hold string — wipe `songT` / `take` / `locationGrade` / `action` and you can still hit Export pack zip. New pack prefills `DEFAULT_LOOK_STYLE`, so a blank job is already 1/9 green, and Sigils-shaped placeholders (`Sigils in the Steel`, `≥4:12`) look like filled values because `styles.css` never styles `::placeholder`.

Docs drift is smaller than gate drift, but it is real: README still has no demo URL (GitHub homepage does), HOW-TO says `git add packs/my-title` while `.gitignore` is `packs/**`, fill order disagrees across HOW-TO / app steps / README gates, and there is no CI on a repo whose whole claim is “tests cover the gate.”

Do not ship more sample packs or a prompt assembler until the nine lights mean the same thing as `docs/FRAMEWORK.md`.

---

## P0 bugs (must-fix)

### 1. The axe rule is the product, and the sample greens it with `none`

`propUnresolved` in `app/src/lib/gate.ts` is:

```ts
overallHaft: !overall || !haft
```

`filled()` is “trim length > 0”. There is no number check, no reject-list for `none` / `TBD` / `unknown`.

**Reproduced** (node probe + live demo):

| Input | `propGenerateFlags.overallHaftUnresolved` | Gate 6 |
|---|---|---|
| Sigils sample haft = `none — not in public example` | `false` | green |
| overall=`TBD`, haft=`unknown` on the sample | `false` | green |
| blank haft | `true` | red (only this case is tested) |

`sigilsSample()` in `app/src/lib/sample.ts` sets haft to `NONE` (`none — not in public example`) with source `not in public lyric sheet — pin before GPU`. `CardsStep` then renders **Overall vs haft length pinned**. `renderProp` exports:

```
- [ ] overall vs haft length unresolved
```

That is the exact Sigils miss (`docs/FRAMEWORK.md` “Research ≠ pin”, `docs/HOW-TO.md` “Ambiguous prop fields (overall vs haft length) must be resolved or the pack is incomplete”). The demo is green *because* the geometry is still unpinned. `gate.test.ts` “the Sigils sample clears all nine” locks the lie in.

`none + why` is the correct still-file protocol (`stillOk`). It is not a length.

### 2. Edit list can be empty of picture and still export as a complete pack

`evaluateEditList` only fails on: missing join, missing/mush camera verb, hold vs `cut`. It does not require `songT`, `durS`, `take`, `locationGrade`, or `action`.

**Reproduced:** clone the sample, set every edit row’s `songT` / `take` / `locationGrade` / `action` / `notes` to `""`. Gate 3 stays green, `allGatesGreen` stays `true`, Export pack zip stays enabled. The zip is an edit list of joins and camera cells with no clock and no take.

That is a song map with extra columns, which is the other Sigils miss (“Song map ≠ edit list”).

Gate copy says “type + amplitude + speed”. `isSingleOfficialCamera(formatCameraCell("pan", "", ""))` is `true`. Amplitude and speed are not required. `holdOkForJoin("continue", "yes before fade")` is `true`; `holdOkForJoin("fadeblack", "banana")` is `true`. Bible: hold only before `fadeblack`; `cut` = no hold.

### 3. New pack looks loaded; Look is actually loaded

`emptyPack()` in `app/src/lib/pack.ts` prefills `look.styleLine` with `DEFAULT_LOOK_STYLE`. `evaluateLook` only checks `filled(styleLine)`. **New pack → 1 / 9 green, gate 7 “One style line locked.”** Live-reproduced. `gate.test.ts` “an empty pack is all red” only asserts `every(ok) === false` (not all-red), so this passes CI-in-your-head.

Worse UX: `PackStep` placeholders are the Sigils strings (`Sigils in the Steel`, `≥4:12`, `0:18 verse · 0:46 chorus`, the log-line sentence). There is no `input::placeholder` rule in `app/src/styles.css`. After New pack, the form *looks* like Sigils until you notice gate 1 is red. Title is not a gate, so an operator who types a log line and leaves the placeholder title exports `# Capture pack — {TITLE}` / `pack-untitled-pack.zip`.

`Header` **New pack** / **Load Sigils sample** have no confirm. Live: click wipes immediately, toast only. Autosave (`App.tsx` 280 ms debounce into `smf.h3-longform-capture.pack.v1`) then overwrites the previous job. Reload will not save you.

---

## P1 gaps / correctness

### Gate vs bible (the checklist is thinner than the paper)

- **Character lock table is optional.** `evaluateCharacters` wants name + lock paragraph + forbidden + `stillOk`. Clear `ageSex` / `faceHairBeard` / `wardrobe` / … — gate 5 stays green. Template `templates/character-card.md` is a lock table. Sample lock paragraphs are generic (“Same face, hair, and wardrobe every hop”) with lock fields set to `none — not in public example`.
- **Stills table is not a gate.** Empty `pack.stills` while character/prop `stillFile` is filled → still 9/9. Two sources of truth (`CardsStep` character/prop still vs Stills tab) drift with no warning. Capture-pack template has a stills table; hop-1 conditioning is the identity story at `cut` / `fadeblack`.
- **Wikipedia is a still.** `stillOk("https://en.wikipedia.org/wiki/Francisca")` is `true`. `stillOk("none.")` is `true` (a trailing `.` counts as “why”). Template and README: “Wikipedia is not a still.” Only a hint on the Stills tab.
- **Smoke attestation does not leave the building.** Gate 9 requires `t2vPlanned` + `watched` + unique prefix. `renderReadme` smoke section is only `smokeNotes`. The zip has no per-take planned/watched columns. You can tick both boxes before any T2V exists; the pack on disk cannot prove you did. Prefix uniqueness *is* real (`sigils-a` vs `SIGILS-A` correctly fails).
- **Hop-1 seed is required to lock the pack, before generate.** `evaluateTakes` demands `hop1Seed`. README gate 4 is “one location + one grade.” Seed is a generate-time / continuity-log field (`templates/continuity-log.md`). You currently cannot go green without inventing a seed.
- **Duration, song clock, title, speech-vs-8s are not gated.** HOW-TO fill step 1 includes duration + audio. Empty `durationTarget` + `songNarrativeClock` on the sample stays 9/9. Speech is a pack-level toggle; no edit-row check that lines finish by 8.0 s.
- **`continue` does not have to stay in one room.** Sample row 2 can be edited to `locationGrade: "yard / night"` with `join: "continue"` — gate 3 stays green. FRAMEWORK: continue = same room, same grade.
- **Map energy `title` / `outro`** are accepted in `energyOk` (`gate.ts`) but `MapStep` `<select>` only offers verse/chorus/bridge. Sample maps `0:00` title overlay as energy `verse`. Clock `ms` is a false-positive shot (`looksLikeShot` `\bms\b`).
- **`mentionsResearch` is a word-boundary hammer.** Source `research notes` fails the whole prop card. Right instinct, wrong granularity — it should hit lock/action language, not every source cell.

### Export / round-trip

- Zip shape is good: `packs/<slug>/{README,edit-list,look,continuity-log,character-*.md,prop-*.md}` matches HOW-TO copy targets. `fadeblack→A` join cell matches `templates/edit-list.md`.
- Draft vs complete is **filename only** (`pack-draft-*.zip` vs `pack-*.zip`). File bodies are identical; no `DRAFT` banner. Incomplete zip looks like a bible once unzipped.
- There is **no markdown/zip/JSON import**. The app cannot parse `templates/` or a previously exported pack. “Parser robustness” is currently `stillOk` / `looksLikeShot` / `detectCameraVerbs` only. Hand-filled `packs/` jobs cannot enter the SPA.
- `holdCell` in `markdown.ts` is a no-op (`return h` both branches).
- `saveStoredPack` has no `try/catch`. `QuotaExceededError` throws out of the `useEffect` in `App.tsx`. `isPack` does not require `stills`, `continuityRows`, `audioPath`. A v2 field or a hand-edit of localStorage will crash `CardsStep` / `SmokeStep` on `.map`.

### Docs / repo hygiene

- **README has no live demo URL.** GitHub `homepage` is `https://h3-longform-capture.vercel.app`. Quickstart `cd app && npm i && npm run dev` is accurate.
- **HOW-TO “Commit a pack” is impossible in this tree.** `.gitignore` is `packs/**` with `!packs/.gitkeep`. `git add packs/my-title` stages nothing. Fine as a public-tree guard (no stills/MP4s); the HOW-TO does not say “private fork: drop that ignore.”
- **Fill order disagrees three ways.** HOW-TO: pack → map → takes → cards → look → edit → smoke. App `STEPS`: pack → map → takes → **edit** → cards → smoke. README gates: log → map → **edit** → takes → characters → props → look → audio → smoke. Audio lives on step 1, numbered as gate 8.
- **No CI.** No `.github/workflows`. Gate tests exist and they currently certify the sample-is-green behavior. `tsconfig.json` `exclude`s `src/**/*.test.ts`, so `npm run build` does not typecheck tests.
- MIT license present and correct. `examples/sigils-lessons.md` is process-only (good); the loadable pack is hardcoded in `sample.ts` and will drift from the lessons file.
- `vercel.json` rewrite is unnecessary today (no client router, step is `useState`) but harmless and already proven on `/does-not-exist`.

### UX / a11y / mobile (live)

- Desktop gate is sticky and usable. **Mobile (390×844): gate is `position: static` below the editor.** Long Cards/prop grids bury “do not queue Comfy.” Step chips wrap and work.
- Audio/speech are `<button class="seg">`, not a radiogroup. Map table inputs have no per-cell `<label>` / `aria-label`. Step nav has no `aria-current`. `.sr-only` exists in CSS and is unused. Toast has `role="status"` (good).
- Gate items are not buttons; clicking a red row does not jump to the step that would fix it.
- Step 1 is labeled **New pack**, same words as the wipe button.

---

## P2 improvements

- **H3 prompt assembler.** Look card says paste the style line into every `[Shot 1]`. FRAMEWORK lists `integrated_multimodal_description` / `overall_soundscape` / `non_diegetic_music`. The app never emits a hop prompt from lock paragraphs + camera cell + audio path. That is the obvious second product, after the gate is honest.
- **Join math.** Hop-1 10.125 / hop 2+ 9.209 / cut often 5 / fadeblack offset = dur − 0.333 s / speech done by 8.0 s are hints, not checks. `durS: "potato"` passes.
- **Take ↔ edit-list sync.** Location/grade typed twice. No unique-prefix helper on the Takes fields (only fails later on Smoke).
- **JSON schema + storage v2.** Version the pack, migrate, reject incomplete objects. Export `.json` beside the zip so a future importer has a chance.
- **Share URL / PNG of the nine-gate card.** Useful as a Spark-floor Polaroid; not a substitute for the zip. Only after lights mean something.
- **More samples.** A narrative (non-music) pack, a chorus-as-`cut` pack that is *honestly* red until haft is a number. Do not add another 9/9 redacted example.
- **A11y polish.** `aria-current` on steps, fieldset for audio/speech, labeled map cells, `:focus-visible` on buttons, `prefers-reduced-motion` on the toast, 44px remove targets.
- **Bundle.** Vite reports `index-*.js` ~360 kB / 110 kB gzip plus three font families at multiple weights. Fine for a lab tool; subset if this stays public.
- Dead CSS: `.btn-ghost`, `.grid-4`, `.sr-only`. No favicon. No error boundary.
- `research` detector scoped to lock/action; map shot detector should not treat clock `ms` as a medium shot.

---

## Quick wins (≤1 day each)

1. **README App section:** link the live demo, one line. GitHub homepage already has it.
2. **GitHub Actions** on `app/`: `npm ci && npm test && npm run build`. Include test files in `tsc` (drop the exclude, or a second tsconfig).
3. **`input::placeholder { color: var(--muted) }`** and stop using Sigils strings as New-pack placeholders (`Title`, `0:18`, `≥4:12`).
4. **Confirm** before New pack / Load sample. One `window.confirm` is enough.
5. **HOW-TO:** “this public `.gitignore` drops `packs/**`. In a private fork, delete that line before `git add packs/…`.”
6. **Map energy select:** add `title` and `outro` to match `energyOk`. Relabel sample `0:00` as `title`.
7. **`stillOk`:** reject `\bwikipedia\b` and require a real why after `none` (letters, not `none.`).
8. **`holdOkForJoin`:** `continue` must be `no`; `fadeblack` must be `yes` / `yes before fade`. Kill `holdCell`.
9. **`aria-current="step"`** on `StepNav`; clicking a `GatePanel` row sets `step`.
10. **Export smoke:** add T2V planned / watched columns to the README takes table (or a smoke checklist). Filename-only draft is not enough — stamp `> DRAFT — gates red` at the top of `README.md` when incomplete.

---

## Suggested next ship (one PR)

**Scope: Gate honesty — props, edit list, sample, export of what the lights claimed.**

Do not add import, share/PNG, or a prompt assembler in this PR. If the sample must stay “demo green,” it cannot also claim geometry is pinned.

**Files:** `app/src/lib/gate.ts`, `pack.ts`, `sample.ts`, `markdown.ts`, `camera.ts`, `app/src/components/CardsStep.tsx`, `EditListStep.tsx`, `PackStep.tsx`, `Header.tsx`, `app/src/lib/gate.test.ts`, `markdown.test.ts`, plus a short README note that the public sample is a lessons pack, not a generate pack.

**Behavior:**

1. Numeric / pinned overall + haft (and the other unit fields that FRAMEWORK lists). Reject `none`, `TBD`, `unknown`, `n/a` as values. `none + why` stays valid only on still files.
2. Edit row must have `songT`, `take` (or `—` on `cut`), `locationGrade`, `action`, camera verb **and** amplitude **and** speed. `continue` + hold fails. `continue` whose location/grade does not match the take card fails.
3. Sigils sample: haft stays `none — pin before GPU` **and gate 6 is red**, with a visible banner “public example — not generate-ready.” Delete or rewrite the test “the Sigils sample clears all nine.” Optionally keep a second fixture `sigilsGenerateReady()` with a numeric haft for the 9/9 path, clearly fake for the demo (`haft: 32 cm, source: measured stand-in`).
4. New pack: empty look style (or do not count the factory default as locked). Confirm on wipe. Muted placeholders.
5. Zip: persist T2V planned/watched; draft README starts with `DRAFT`.
6. Tests: the probe cases in this review (TBD/unknown, empty edit action, `none.`, Wikipedia URL, empty-pack look, continue+hold, stripped songT).

Done when: live demo can no longer show **Overall vs haft length pinned** next to `none`, and Export pack zip is disabled until the edit list has clocks, takes, and one verb with amp+speed.

---

## What was run

```text
cd app && npm i && npm test && npm run build
# tests 30, fail 0
# vite build clean (2026-09-19)
```

Live: https://h3-longform-capture.vercel.app — sample 9/9, New pack 1/9 (Look green), props tab “Overall vs haft length pinned” on haft `none — not in public example`, mobile gate below the form, no confirm on wipe/load.
