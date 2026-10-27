# Scene script and cue list: the format, with a 30-second example

Read with `package_read("references/scene-and-cue-format.md")`.

## A complete 30-second script

The prompt gave this data: "decode speed 58.2 tok/s at 4k context and 41.8 tok/s at 128k; the
adaptive block gives 51.9 req/s against 47.4 for the fixed block (table 2)". Every number below
names that source.

```json
[
  {"id": "s1", "kind": "title", "seconds": 3, "title": "Why context costs speed", "lines": ["a 30-second explainer"]},
  {"id": "s2", "kind": "number", "seconds": 4, "title": "Short context", "lines": ["one user, 4k tokens"],
   "numbers": [{"value": "58.2 tok/s", "source": "the prompt's data, 4k"}]},
  {"id": "s3", "kind": "schematic", "seconds": 6, "title": "Where the time goes", "lines": ["schematic, not to scale", "every step reads the whole cache"],
   "marks": ["box: prompt cache", "box: decode step", "arrow: prompt cache -> decode step", "label: grows with context"]},
  {"id": "s4", "kind": "number", "seconds": 4, "title": "Long context", "lines": ["one user, 128k tokens"],
   "numbers": [{"value": "41.8 tok/s", "source": "the prompt's data, 128k"}]},
  {"id": "s5", "kind": "compare", "seconds": 5, "title": "Fixed against adaptive",
   "compare": {"left": {"label": "fixed block", "value": "47.4"}, "right": {"label": "adaptive block", "value": "51.9"}, "unit": "req/s", "source": "table 2"}},
  {"id": "s6", "kind": "number", "seconds": 4, "title": "The difference", "lines": ["adaptive over fixed"],
   "numbers": [{"value": "+9.5 %", "source": "computed from table 2: 51.9 / 47.4"}]},
  {"id": "s7", "kind": "title", "seconds": 4, "title": "Measure, then claim", "lines": ["same build, same hour, both sides"]}
]
```

```json
{"bed": {"chords": ["Dmaj7", "Bm7", "Gmaj7", "Aadd9"], "seconds_per_chord": 8, "gain_db": -24},
 "events": [
   {"at": "s2", "kind": "whoosh"}, {"at": "s2.numbers[0]", "kind": "chime"},
   {"at": "s3", "kind": "whoosh"}, {"at": "s4", "kind": "whoosh"}, {"at": "s4.numbers[0]", "kind": "chime"},
   {"at": "s5", "kind": "whoosh"}, {"at": "s6", "kind": "whoosh"}, {"at": "s6.numbers[0]", "kind": "chime"},
   {"at": "s7", "kind": "tock"}],
 "narration": []}
```

Total: 3 + 4 + 6 + 4 + 5 + 4 + 4 = 30 s. The tool adds a 0.45 s cross-fade between scenes and
reports the real length (about 31 s), which is inside 25 to 35.

## Rules the tool enforces (an error comes back, the model fixes the list)

| rule | error text |
|-|-|
| a `number` or `compare` scene without a `source` | `s4: number without source` |
| a `schematic` scene whose `lines` lack the word "schematic" | `s3: schematic not labelled` |
| a title over 5 words, a line over 60 characters, more than 3 lines | `s1: title too long` |
| an `at` that names no scene | `cue 3: unknown scene s9` |
| total seconds over 10 % off the asked length (the harness passes the asked length) | `length 42 s, asked 30` |
| a chord name outside the 24 major/minor/7th names the bed knows | `bed: unknown chord Xsus` |

## What each kind looks like

- `title`: the title large and centred on the dark ground, the lines small under it, a 0.3 s fade.
- `number`: the title top left, the value very large (tabular digits) with the unit smaller, the
  source in small text at the bottom edge ("source: table 2"), the lines beside the value.
- `schematic`: the marks drawn as boxes and arrows in the house palette (cobalt boxes, orange
  arrows), labels in monospace, the word "schematic" in the corner.
- `compare`: two columns, each a label and a value, the larger value in orange, the unit once, the
  source at the bottom.

## Cues

- `whoosh`: a band-passed noise sweep, 0.7 s, at every scene change by default.
- `chime`: two partials, 1.2 s, when a number appears (`s2.numbers[0]` is the moment the first
  number of s2 lands, 0.6 s into the scene).
- `tock`: a short low click for a title card.
- `click`: a UI click, for a schematic mark that "activates".
- `bed`: a slow four-chord pad from detuned triangle partials through a low-pass sweep; it sits
  at `gain_db` (default -24 LUFS on its own) and ducks 3 dB under narration.
- `narration`: the system voice reads `text` at the scene's start; the scene grows to the
  sentence's length plus 0.5 s when it is longer than `seconds`.

The mix is normalised with two-pass `loudnorm` to -18 LUFS, true peak -1.5 dBTP. The model never
sets levels by hand.
