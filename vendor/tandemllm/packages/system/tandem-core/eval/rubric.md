# Rubric: tandem-core

Automatic checks (each pass or fail, read from the transcript and the tool calls, no judge):

| check | passes when |
|-|-|
| `tool_args_typed` | every tool call's arguments parse as JSON and each argument has the type the tool's schema declares (a number is not a string, a boolean is not a string) |
| `no_think_tags` | the answer text and every tool argument contain neither `<think>` nor `</think>` |
| `one_answer` | one assistant message with content after the last tool result; no second answer, no repeated final paragraph |
| `file_lines_max_200` | no single file-write call carries more than 200 lines; a longer file arrives in two or more calls |
| `plain_final_answer` | the final answer names what was built, where it is and how it was checked, and contains no tool name, score or runtime note (the SRV-229 word list of the harness: Laya, kit, hook, scratch, fps readout) |

Judge: none. This package is scored by the automatic checks alone.

Pass rate = prompts with every check passed / 8. The before side (no package) and the after side (with it) run in the same hour on the same build; the delta is reported with both counts.
