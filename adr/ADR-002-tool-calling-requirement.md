# ADR-002: Tool Calling Requirement for All Flavors

**Date:** July 11, 2026
**Status:** Accepted
**Author:** Oracle

## Context

Hermes is an agent framework. Every profile (Loca, Venus, Kenny, Pedro, etc.) runs as an agent loop that requires tool calls to function. When a vLLM server doesn't have `--enable-auto-tool-choice` and a `--tool-call-parser`, any request that includes `tools` with `tool_choice="auto"` gets HTTP 400. The agent loop dies instantly.

This is what happened when Oracle wired profiles to the DS4 server on July 5: the DS4 serve script had no tool calling flags, so every Hermes agent got 400 errors on every message. The Qwen 122B serve script (written by the Entrpi repo author, verified by Lara) correctly includes these flags.

## Decision

**All three Spark flavors must have tool calling enabled.** This is non-negotiable.

### Required Flags Per Flavor

| Flavor | Tool Parser | Reasoning Parser | Flag |
|---|---|---|---|
| 35b | `qwen3_xml` | `qwen3` | `--enable-auto-tool-choice` |
| 122b | `qwen3_xml` | `qwen3` | `--enable-auto-tool-choice` |
| ds4 | `deepseek_v4` | `deepseek_v4` | `--enable-auto-tool-choice` |

### Parser Selection Rationale

- **qwen3_xml**: Qwen 3.5/3.6 emits XML-format tool calls (`<=function=name>=function<`). The `qwen3_xml` parser reads this format. `hermes` parser only reads JSON format and returns empty tool_calls — verified by the Entrpi serve.sh comments.
- **deepseek_v4**: DS4 uses DSML format (`<｜DSML｜invoke name="...">`). The `deepseek_v4` parser (extends `deepseekv32_tool_parser`) handles this. Confirmed available in vLLM-Moet venv on Spark.

### Verification

Tool parsers confirmed available in `~/venvs/vllm-moet/lib/python3.12/site-packages/vllm/tool_parsers/`:
- `deepseekv4_tool_parser.py` — registered as `deepseek_v4`
- `qwen3_engine_tool_parser.py` — registered as `qwen3_xml`, `qwen3_coder`, `qwen3_engine`

Reasoning parsers confirmed in `~/venvs/vllm-moet/lib/python3.12/site-packages/vllm/reasoning/`:
- `deepseek_v3_reasoning_parser.py` — registered as `deepseek_v4`
- `qwen3_engine_reasoning_parser.py` — registered as `qwen3`

## Consequences

- DS4 serve script must be patched before it can be used as an agent backend
- All future recipes must include tool calling flags before testing
- A recipe without tool calling is automatically rejected (cannot pass Loca agent loop test)
- The tool parser only activates when a request carries `tools`, so plain chat is unaffected