# Framework

Long-form generative video is not a longer prompt. It is a **capture pack**, then many short generations with locked references.

## Why Sigils missed the song

On 2026-09-14 we were given a music-video brief for *Sigils in the Steel*: ≥4:12, title overlay, “research the Merovingian axe,” every camera at once, lyric timings, mute in the NLE later. We collapsed it into six Motion-Context takes (28 windows), joined with 8-frame fadeblack, and landed **262.846 s / 6285 f / 1344×768**. Picture held. It was not a rock montage.

Failures that this pack exists to catch:

1. **Research ≠ pin.** Lyrics already had 45 cm / 600 g / 10 cm / square poll / teardrop eye. We still mixed Wikipedia shape language (S-curve, arch-shaped) with no still.
2. **Song map ≠ edit list.** Clocks told us when the chorus started. They did not say `cut` vs `continue`.
3. **Every camera at once.** Official H3 wants type + amplitude + speed, one verb. We chose slow single-verb windows so takes would hold, and spent the chorus energy.
4. **`non_diegetic_music: N/A` plus a mastered track** is the right music-video path only if picture is cut to the waveform. 10 s holds are not cuts.

Receipts: https://www.smfclearinghouse.com/blog/2026-09-16-h3-sigils-four-minutes

## What we borrowed (and what we measured)

**Film.** A shot list names scene, shot, description, size, movement, subject.[4] A continuity sheet records what *changed* (prop in/out, hair, which hand) and stills beat memory.[5]

**AI video.** Models do not remember the last clip. Character work is a master sheet, identical keywords, a forbidden list, and reference images.[3]

**H3 official.** Fields in order: `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`. `[Shot 1]` has no timestamp. Camera is an English action. FL2VA describes the **path between** stills, not the stills. Prefer one shot so the model interpolates.[1][9]

**H3 chaining (external Comfy pack).** Last frame → next first frame is FL2VA’s trained job. Preview shot 1 before paying for the chain. Long chains accumulate texture; cap extend-takes unless measured.[2]

**Papers.** Subject consistency is identity **grounding**, not local continuation. Next-shot memory without an identity store drifts. Entity-level scheduling (who/what must persist) beats a prompt paragraph.[6][7][8]

**SMF measurement overrides blogs.** Motion-Context 22 holds a take. Fadeblack is a scene cut. Abort ≥85–86°C. One verb. Lyric numbers must be pinned before generate.

**Still factory.** A sheet is the character/prop bible. A plate is the first frame of a window. Generate both on the image Spark (Qwen-Image-2.1) at native 1344×768, then condition hop-1 with `MiniMaxH3ImageToVideo.first_frame`. Hop 2+ is Motion-Context, not a new still. Procedure: [IMAGE-STILLS.md](IMAGE-STILLS.md).[10][11][12]

## The three joins

| Join | Picture | When |
|---|---|---|
| `continue` | Motion-Context hop, trim 22, concat `-c copy` | Same room, same grade, action continues |
| `cut` | New I2VA/T2V (new plate), hard cut, **no last-second hold** | Chorus, snare, new angle |
| `fadeblack` | New take + 8-frame dip | New location / time of day |

Hop-1 is 243 frames / **10.125 s** @ 24 fps. Hop 2+ after trim is **221 f / 9.209 s**. Fadeblack offset = duration − 0.333 s. Speech and chorus hits finish by **8.0 s**.

## Prop card (the axe rule)

If the lyric sheet already has numbers, copy them. Do not average three encyclopedia diagrams.

Must resolve before generate:

- Overall length vs haft length vs edge width (each with a unit)
- Head mass
- Poll / eye / socket
- Wood and finish
- Decoration layout
- Still path or explicit `none`

Forbidden list: bearded blade, double bit, horns, chrome, unless listed.

## Camera

One verb per window. Official vocabulary: push/pull, pan, truck, tilt, pedestal, arc, track, static, shake; plus amplitude and speed. “Pan and zoom and circle” is three rows or it is refused.

## Audio

Pick one:

1. `non_diegetic_music: N/A`, mute AAC in the NLE, lay the mastered track, cut picture to the waveform (music video).
2. Fill the music field (H3 invents *a* score, not *your* track).
3. Silence (`N/A` and no later mix).

Do not do (1) and (2).

## Smoke

Three independent hop-1s + planned fades before a 28-window night. I2VA if a plate exists. Watch identity at the cuts. A hop-1 with no SaveLatent cannot be hopped.

## Sources

See [SOURCES.md](SOURCES.md).
