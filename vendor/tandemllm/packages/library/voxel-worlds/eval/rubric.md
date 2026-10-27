# Rubric: voxel-worlds

Automatic checks (pass or fail):

| check | passes when |
|-|-|
| `runs` | the page loads, no uncaught error and no unhandled rejection in 10 s |
| `no_console_errors` | zero console errors |
| `fps_min` | the median fps over seconds 2 to 10 at 640x360 is at or above the manifest's `calibration.fps_min` |
| `fps_baseline_held` | vw-20 only: the fps after the sixth feature is within 5 of the baseline the model wrote after the island (read from `TODO.md` and `__state().fps`) |
| `stable_60s` | vw-20 only: fps and a scene hash at 60 s are within bounds of those at 10 s |
| `audio_nonsilent` | for prompts that ask for sound: an AudioContext exists after a click, RMS above -40 dBFS, peak below -1 dBFS, not a single tone |
| `prompt_terms` | every word in `terms` appears in the page's text, ids, class names or comments, and (vw-20) the first five features are still present in the final page |
| `seed_stable` | vw-01: two loads with `?seed=7` give the same `__state().voxels` and the same first frame hash |
| `todo_file` | vw-20: a `TODO.md` exists with a `done` list of six lines, each with an fps |

Judge items (1 to 5, one sentence of reason each):

| item | 5 means | 1 means |
|-|-|-|
| match | the scene reads as asked (the pagoda is a pagoda, the lake is a lake, the feature is there) | a different scene |
| composition | the island fills the frame from the default camera, the landmark visible, the HUD readable | the camera inside the ground or an empty frame |
| motion | creatures, water, petals or the sun move between frames | nothing moves |
| sound | the probe non-silent when sound was asked, `null` otherwise | silence where sound was asked |
| bugs | no black frame, no z-fighting bands, no HUD overlap | an error page |
| grounded | nothing floats and nothing sinks: every creature, lantern and bridge end sits on the ground or the water | objects in the air or inside the terrain |
| lighting | sun, sky, fog and lanterns agree with the time of day | daylight sky with night lanterns on |

Human sample: vw-01, vw-02, vw-03, vw-04, vw-07, vw-09, vw-11, vw-16, vw-19, vw-20, the same ten every run.
