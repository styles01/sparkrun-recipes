# Rubric: render-tools

Automatic checks (from the transcript and the tool results; no judge):

| check | passes when |
|-|-|
| `render_called` | at least one `render` call in the turn |
| `renders_max_3` | at most 3 `render` calls in the answer |
| `audio_probe_called` | at least one `audio_probe` call when the prompt asks for sound |
| `fix_after_error` | after a render whose console had an error, the next render is a changed page (different `html`) and its error count is lower |
| `no_console_errors` | the last render's `error_count` is 0 |
| `allowlist_only` | every `external_urls` entry of the last render is on the package's allowlist |
| `plain_final_answer` | the final answer has no tool JSON and no tool name; it says in words what was seen |

Counts recorded per run: renders per prompt (mean), prompts with a fix after an error, prompts whose
last render has 0 errors. This package is measured with its "off with tools" row (the tool
offered, the package absent) beside the plain off row, because the row that separates the tool's
effect from the instructions' effect is the one that says which lever did the work.
