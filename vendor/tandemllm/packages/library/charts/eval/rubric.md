# Rubric: charts

Automatic checks (pass or fail):

| check | passes when |
|-|-|
| `one_file` | exactly one fenced `svg` block (an `html` block for ch-20), no external URL |
| `runs` | the SVG parses and renders with no console error in 10 s |
| `prompt_terms` | every value in the prompt's `terms` appears as a text node of the rendered SVG (thin-space thousands accepted as plain digits too); this is the correctness check and it is automatic |

Judge items (1 to 5, one sentence of reason each):

| item | 5 means | 1 means |
|-|-|-|
| match | the asked form, the asked highlight in orange, the title names the unit | a different form or no highlight |
| composition | axes, ticks, labels and title readable, nothing overlaps, plot fills the frame | overlapping text or an empty frame |
| motion | still is right for a chart: 5 | an animation that distracts |
| sound | `null` | `null` |
| bugs | no clipped label, no bar below zero, no missing axis | a broken frame |
| numbers | every number readable at a glance, tabular, thin-space thousands, the unit once | numbers missing or unreadable |

Human sample: ch-01, ch-02, ch-03, ch-04, ch-06, ch-09, ch-11, ch-14, ch-17, ch-20, the same ten every run.
