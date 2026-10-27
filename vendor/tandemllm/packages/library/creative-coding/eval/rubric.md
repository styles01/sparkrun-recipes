# Rubric: creative-coding

Automatic checks (pass or fail, from the render tool's results):

| check | passes when |
|-|-|
| `one_file` | exactly one fenced html block (or one written `index.html`), under 200 KB, no external URL outside the allowlist |
| `runs` | the page loads, no uncaught error and no unhandled rejection in 10 s |
| `no_console_errors` | zero console errors (warnings are counted, not failed) |
| `fps_min` | the median frame rate over seconds 2 to 10 at 640x360 is at or above the manifest's `calibration.fps_min` (set once on the box; not in this file) |
| `audio_nonsilent` | for prompts with `"sound": true`: an AudioContext exists, the 10 s probe has RMS above -40 dBFS, peak below -1 dBFS, and the spectrum is not a single tone |
| `prompt_terms` | every word in the prompt's `terms` appears in the page's text, element ids, class names or comments (the named elements exist in the code) |

Judge items (1 to 5, one sentence of reason each):

| item | 5 means | 1 means |
|-|-|-|
| match | every element the prompt names is visible in the frames | a different scene or an empty page |
| composition | the subject fills the frame with intent, nothing cut off, text readable | a blank or cluttered frame |
| motion | frames differ the way the prompt implies (leaves fall, water runs) | nothing moves |
| sound | the probe is non-silent and not one tone, and the prompt asked for sound (`null` when it did not) | silence where sound was asked |
| bugs | no black frame, no overlap, nothing outside the frame | an error page or a black frame |

Human sample: cc-01, cc-02, cc-05, cc-09, cc-13, cc-14, cc-19, cc-21, cc-23, cc-24, the same ten every run.
