# Still factory → clip factory

Long-form H3 does not remember the last clip. A pasted wardrobe paragraph is not a face. Identity holds **inside** a Motion-Context take. At `cut` and `fadeblack` it drifts until a **plate still** conditions hop-1.

This file is the image half of the capture bible. Serving pins stay in SMF ops. Prompt field syntax stays in MiniMax `h3-prompt-writing`. Do not queue either Spark until the nine gates in the README are green **and** every hop-1 that must hold a face or prop has a plate (or explicit `none` + why).

## Two boxes, two jobs

SMF occupancy (2026-09-20): one heavy engine per GB10.

| Job | Box | Engine | Canvas |
|---|---|---|---|
| Still factory | spark-d369 | Qwen-Image-2.1 Comfy (INT8 ConvRot) | Generate **1344×768** (H3 native 16:9). Do not stretch 1024². |
| Clip factory | spark-56bc | Comfy native MiniMax H3 + Motion-Context | Hop-1 **1344×768**, 243 f / 10.125 s, 6-step turbo |

Do not stand Image-2.1 next to H3. Do not stand H3 on d369. Copy PNGs across the LAN; do not share the GPU.

Measured still factory (Clearinghouse 2026-09-20, 23/23): 1024² / 25 step **20 s** after warmup; **1344×768 / 25 = 24 s**; edits ~**28 s**; 2048² / 25 = 128 s. Peak die 78°C. H3 on 56bc stayed HTTP 200.

## Sheet vs plate (do not collapse)

| Still | What it is | What it is not |
|---|---|---|
| **Sheet** | Character / prop bible. Front (or hero) view, locked keywords, forbidden list. Lives on the character/prop card. | Not the first frame of a take. |
| **Plate** | The actual first frame of a hop-1, cut, or fadeblack: same identity, **this** location, **this** grade, **this** camera size. Conditions `MiniMaxH3ImageToVideo.first_frame`. | Not a Wikipedia crop. Not a 1024² sheet stretched to 16:9. |

Sigils failed the axe because we had numbers and no still. The next failure mode is the opposite: a beautiful sheet, then thirteen T2Vs that invent thirteen faces.

Copy the lock paragraph **from** the still. Do not generate the still from a lock you wrote after.

## Where stills enter the three joins

| Join | Still | Clip tool |
|---|---|---|
| `continue` | Plate on **hop-1 only**. Hop 2+ is Motion-Context latent (`SaveLatent` → exact filename). | Do **not** re-feed a new Qwen still on hop 2+. That fights the latent. |
| `cut` | New plate (often a Qwen **edit** of the sheet: new angle, same face). | New I2VA hop-1, hard cut, **no hold**. |
| `fadeblack` | New plate in the **new** location/grade (edit the sheet into that set). | New I2VA hop-1 + 8-frame dip. Do not use take A's last frame as take B's first — that is a cut, not a scene change. |

Chorus rows that used to be “independent T2Vs” become independent **plates → I2VA**. Same sheet, new plates, hard cuts.

## Official H3 (what the model was trained to do)

H3-Base-FL2VA accepts zero, one, or two images: none = T2VA, one = first- **or** last-frame, two = first-and-last. Ref2VA (≤9 images) is a different checkpoint. SMF live generate is Comfy native FL2VA + Motion-Context, not Ref2VA.

Comfy node on spark-56bc (verified 2026-09-20 `/object_info`): `MiniMaxH3ImageToVideo` required `clip, vae, prompt, width, height, length`; optional **`first_frame`, `last_frame`**.

Prompt rules (MiniMax `base-en.txt`):

- I2VA / FL2VA / L2VA: **alignment line first**, then a blank line, then the three fields.
- I2VA: develop **forward** from the first frame. Do not re-describe the still.
- FL2VA: the body is a **path** (first-frame state → intermediate changes → last-frame state), not two still descriptions. Prefer **one shot** so the model interpolates.
- `MiniMaxH3AddGuide` is a different product. Do not mix it with Motion-Context in one chain.

The joeynyc `/v1` pin (drained): `task=fl2va` + `input_reference=@first.png`. Measured 2026-09-05: geometric RGB identity held on frame 0 from a clean PNG. That is the **API** first-frame receipt, not a Comfy native hop-1 + SaveLatent measurement.

**Not yet measured on 56bc:** Qwen plate → `MiniMaxH3ImageToVideo.first_frame` → `MiniMaxH3MotionContextSaveLatent` → hop 2+. Prescribe the graph. Do not invent PSNR. Watch hop-1 before hopping.

## Still factory recipe (d369)

Pin: INT8 ConvRot, euler/simple, cfg 1, AuraFlow shift 3.1, 25 steps. Look line = the look card, verbatim.

1. **Sheets.** One T2I per character and per hero prop at **1344×768**. Same look lock as H3. Reject on-screen lettering unless the lyric needs it.
2. **Plates.** For every hop-1 / `cut` / `fadeblack` that must hold identity: edit the sheet into that location and grade. Working graph: `LoadImage` → `TextEncodeQwenImageEdit` + `VAEEncode` → KSampler. **Do not** pass `image_1=` into `TextEncodeQwenImage21` (TypeError; the graph validates then dies at execute).
3. **Lock from pixels.** Write the character/prop lock paragraph from the sheet. Forbidden list from what the still must not grow.
4. **Name files.** `stills/{entity}-sheet.png`, `stills/{take}-hop1-plate.png`, `stills/cut-{n}-plate.png`. Relative paths on the cards.
5. **Copy to 56bc.** Plates land where Comfy `LoadImage` can see them. Sheets stay in the pack even if hop-1 is T2VA (`none` + why).

Do not publish likeness stills or MiniMax MP4s to this public repo. Private fork or sibling private tree.

## Clip factory recipe (56bc)

1. Hop-1 graph: `MiniMaxH3ImageToVideo` with `first_frame` = the plate (empty = T2VA, which is how you get a new face). `width=1344`, `height=768`, length 243. Prompt = I2VA alignment line + one verb + lock paragraph copied from the sheet.
2. Same hop-1 **must** `MiniMaxH3MotionContextSaveLatent` with a unique prefix. A watched I2VA with no latent cannot be hopped.
3. Watch identity at the planned `cut` / `fadeblack` **before** hop 2+.
4. Hop 2+: Motion-Context from the exact latent filename. Trim 22. Concat `-c copy` only inside the take.
5. Optional FL2VA two-still (first + last plates) is for a planned path inside **one** window (orbit, walk-to-mark). It is not a substitute for Motion-Context across windows.

## Research this matches (and what we do not take)

Film continuity stills beat memory. AI character guides: master sheet, identical keywords, forbidden list, reference images — not a new paragraph per shot.

Papers: subject consistency is **identity grounding**, not local continuation. Storyboard / keyframe pipelines treat the still as the visual anchor of each clip, then expand. Entity-level scheduling (who/what must persist) beats a prompt paragraph. SMF measurement still overrides blogs: Motion-Context 22 holds a take; fadeblack is a scene cut; abort ≥86°C on the live sigils run.

We do **not** take: Ref2VA as the local generate path; `MiniMaxH3AddGuide` mixed into Motion-Context; last-frame of take A as hop-1 of take B across a location change; Wikipedia as a still; 1024² sheets scaled to 1344×768.

## Pitfalls

| Fail | Fix |
|---|---|
| Thirteen T2Vs + a pasted lock | Sheets, then plates, then I2VA hop-1 |
| Sheet used as hop-1 plate | Edit into that location/grade; canvas 1344×768 |
| Stretch 1024² | Generate native 1344×768 (~24 s @ 25 step) |
| `image_1=` on `TextEncodeQwenImage21` | `TextEncodeQwenImageEdit` + `VAEEncode` |
| New Qwen still on hop 2+ | Motion-Context latent only |
| Mix AddGuide + Motion-Context | One chain, one method |
| Image-2.1 on 56bc | Split occupancy. Copy PNGs. |
| Wikipedia / research tab as still | `none` + why, or a real PNG |
| Unmeasured I2VA+SaveLatent claimed as a pin | Watch hop-1; log it; then hop |

## Verification

Still factory is done when: every character/prop has a sheet or `none` + why; every hop-1 / cut / fadeblack that must hold identity has a 1344×768 plate (or `none` + why); lock paragraphs were copied from the sheets; look line matches; files are named and on the clip box. Generate without that is a miss, even if ffprobe is green.
