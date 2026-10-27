# Rubric format

Every package's `eval/rubric.md` has this shape, so the judge and the human page read it the same
way:

```markdown
# Rubric: <package>

Automatic checks (pass or fail, from the tool results):

| check | passes when |
|-|-|
| <check id> | <one sentence> |

Judge items (1 to 5, one sentence of reason each):

| item | 5 means | 1 means |
|-|-|-|
| match | ... | ... |
| composition | ... | ... |
| motion | ... | ... |
| sound | ... | ... |
| bugs | ... | ... |
| <extra row> | ... | ... |

Human sample: prompts <ids>, the same ten every run.
```

Rules:
- The five standard items keep their names and their order.
- An extra row has a one-word lower-case id (`numbers`, `grounded`), a "5 means" and a "1 means".
- A check id names a function of the harness (`one_file`, `runs`, `no_console_errors`,
  `fps_min`, `stable_60s`, `audio_nonsilent`, `prompt_terms`, and the package's own); a rubric
  that names an unknown check fails to load.
- A threshold that depends on the box (`fps_min`) is not in the rubric. It is in the package's
  manifest under `calibration`, written once after the calibration run on the box.
