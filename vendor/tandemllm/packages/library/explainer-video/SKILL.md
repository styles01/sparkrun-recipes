---
name: explainer-video
description: A short explainer video with synthesised sound rendered to an mp4 through the render_video tool - a scene script (title cards, numbers with their sources, schematic marks) and a sound cue list (bed, whoosh, chime, optional narration); every number on screen names its source.
---
# Explainer video

You write two lists; `render_video` renders and mixes them.
1. `scenes`: a JSON list in order; kinds `title`, `number`, `schematic`, `compare`; fields `id`, `kind`, `seconds`, `title`, `lines`, `numbers` (each a `value` and a `source`); see the reference.
2. `cues`: `bed` (four chords, seconds per chord), `events` (a `whoosh` per scene change, a `chime` per number), `narration` (only when a voice is asked).
3. Call `render_video(scenes, cues)`; it returns path, seconds, LUFS, peak, black run.
4. Check: seconds in the asked range (30 s: 25 to 35), -19 to -17 LUFS, peak at most -1.5 dBTP, no black run over 1 s; else fix and call again, 3 calls at most.

Honesty: every number on screen has a `source` (the prompt's data or a named file), else it is not shown. A `schematic` scene says "schematic" in a line. Titles 3 to 5 words, lines under 60 characters.

Done when:
- [ ] The tool returned an mp4 with seconds, LUFS and peak in range.
- [ ] Every `number` scene has a `source`; every `schematic` scene says so.
- [ ] The answer names the file, its length and one line per scene.

## Reference

Scene fields:
```
id        "s1".."sN", unique                          seconds  3..10 (defaults by kind: title 3, number 4, schematic 6, compare 5)
kind      title | number | schematic | compare        title    3 to 5 words, shown large
lines     0 to 3 strings, each at most 60 characters  numbers  [{value: "9.6 tok/s", source: "..."}], 1 to 3 per number or compare scene
marks     schematic only: ["box: engine", "box: cache", "arrow: engine -> cache", "label: 1,024-token grid"]
compare   compare only: {"left": {"label": "fixed block", "value": "47.4"}, "right": {"label": "adaptive", "value": "51.9"}, "unit": "req/s", "source": "..."}
```
Cue fields:
```
bed        {"chords": [4 chord names], "seconds_per_chord": 8..32, "gain_db": -24}   a slow pad; it ducks 3 dB under narration
events     [{"at": "<scene id>" | "<scene id>.numbers[i]", "kind": "whoosh" | "chime" | "tock" | "click"}]
narration  [{"at": "<scene id>", "text": "one sentence, at most 120 characters"}]   spoken by the system voice; the scene's seconds grow to fit
```
Return: `{"path": "explainer.mp4", "seconds": 31.2, "lufs": -18.1, "true_peak_dbtp": -1.6, "black_max_s": 0.0, "scenes": 7}`.
A 30-second example (7 scenes: title 3, number 4, schematic 6, number 4, compare 5, number 4, title 4 = 30 s) with its cues is in `references/scene-and-cue-format.md`; the tool's contract is in `references/render-video-interface.md`.
