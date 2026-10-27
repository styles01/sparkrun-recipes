# Judge prompts

The judge turn the eval harness (OPS-37, `tools/package_eval.py`) builds. The system text is the
eval-judge package body; the user turn is this shape. Temperature 0, thinking off, `max_tokens`
600, the served model's image input.

## The user turn

```
Prompt the model received:
<the eval prompt, verbatim>

Package rubric (extra rows, if any):
<the rows of eval/rubric.md below the five standard items, or "none">

Frames: three images follow, at 0.5 s, 2 s and 5 s.
<image part: frame at 0.5 s>
<image part: frame at 2 s>
<image part: frame at 5 s>

Console: errors 0, warnings 2.
Sound probe: AudioContext yes, RMS -23.4 dBFS, peak -7.1 dBFS, single tone no.   (or: "Sound probe: none")

Score now. One JSON object.
```

## The human page

The run view shows the same three frames of a prompt for both sides without saying which side is
on, and the reviewer scores the same items 1 to 5. The reviewer's name and the date are stored
with the scores. Ten fixed prompts per package, the same ten every run.

## Agreement

Cohen's kappa between the judge and the reviewer on the ten prompts, per item and overall, is in
every run's table. Below 0.4 the judge's score is shown but not used for a claim; the human score
decides whether a package ships as measured.

## What the judge must not see

The code, the model's reasoning, the other side's frames, the package name, the run's labels.
