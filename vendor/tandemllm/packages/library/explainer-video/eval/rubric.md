# Rubric: explainer-video

Automatic checks (pass or fail, from the tool's return and the file):

| check | passes when |
|-|-|
| `file_exists` | the returned path is an mp4 that ffprobe reads |
| `length_range` | the duration is within the asked length ±5 s (±10 % above 60 s) |
| `audio_nonsilent` | RMS over the whole file above -40 dBFS |
| `loudness_range` | integrated loudness between -19 and -17 LUFS |
| `true_peak` | true peak at or below -1.5 dBTP |
| `no_black_second` | the longest black run is at most 1.0 s |
| `sources_named` | every `number` and `compare` scene in the list the model sent has a non-empty `source` |
| `schematic_labelled` | every `schematic` scene's `lines` contain the word "schematic" |
| `narration_present` | ev-09: the cue list has one narration entry per scene and the bed ducks (the tool reports `ducked: true`) |
| `no_number_scenes` | ev-12: no `number` scene and no `chime` cue |

Judge items (on the three returned frames; 1 to 5, one sentence of reason each):

| item | 5 means | 1 means |
|-|-|-|
| match | the frames show the asked topic: the named numbers, the schematic marks | unrelated cards |
| composition | the title readable, the number large, the source visible at the bottom, nothing overlapping | text cut off or overlapping |
| motion | the three frames are three different scenes | the same card three times |
| sound | `null` for the judge; the human listens | `null` |
| bugs | no black frame, no empty card, no placeholder text | an empty or black frame |
| honesty | every visible number has a source line; every schematic says so on screen | a number with no source |

Human sample (the reviewer listens to each): ev-01, ev-02, ev-03, ev-04, ev-05, ev-06, ev-08, ev-09, ev-11, ev-12.
