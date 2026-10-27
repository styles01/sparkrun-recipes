# Rubric: eval-judge

The judge is tested on described cases (the frames are given as text in these prompts, so the
test runs without images; the harness also runs the same six with real frames once the render
tool exists).

Automatic checks:

| check | passes when |
|-|-|
| `json_only` | the whole answer parses as one JSON object, no text around it |
| `all_items_scored` | `match`, `composition`, `motion`, `sound`, `bugs`, `reasons` (five strings) and `overall` are present; `sound` is `null` exactly when no probe was given |
| `scores_in_range` | every scored item is an integer 1 to 5; `overall` is the mean of the scored items to one decimal |
| `stable_twice` | the same prompt judged twice in the same run gives the same scores |
| `expected_band` | each item named in the prompt's `expect` lies in its band |

Judge items: none (this package is the judge).

Human sample: none. Agreement with the reviewer is measured on the packages' own ten-prompt samples, not here.
