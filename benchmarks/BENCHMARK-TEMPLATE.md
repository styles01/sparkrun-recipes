# Benchmark Template — Spark LLM Flavor Test

**Copy this file for each benchmark run. Name: `benchmarks/<flavor>-<date>.md`**

---

# Benchmark: [FLAVOR] — [DATE]

**Flavor:** [35b / 122b / ds4]
**Date:** [YYYY-MM-DD]
**Tester:** [Oracle / James]
**Recipe:** [link to recipe file]
**Recipe status:** [LOCKED / DRAFT / TESTING]

## Server Configuration

| Parameter | Value |
|---|---|
| Served model name | |
| Docker image / venv | |
| GPU mem util | |
| Max model len | |
| Max num seqs | |
| Max num batched tokens | |
| Tool parser | |
| Reasoning parser | |
| Speculative config | |
| Attention backend | |
| Prefix caching | |
| KV cache dtype | |
| Startup time | [mm:ss] |

## Raw Performance (curl-based)

### Test 1: Short Q&A (128 tokens)
```
Prompt: "What is the capital of France?"
Max tokens: 128
```
| Run | Tokens | Wall time | tok/s | TTFT (ms) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

### Test 2: Code Generation (512 tokens)
```
Prompt: "Write a Python function to reverse a linked list."
Max tokens: 512
```
| Run | Tokens | Wall time | tok/s | TTFT (ms) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

### Test 3: Long Generation (1024 tokens)
```
Prompt: "Explain the history of computing from 1950 to 2020."
Max tokens: 1024
```
| Run | Tokens | Wall time | tok/s | TTFT (ms) |
|---|---|---|---|---|
| 1 | | | | |

### Test 4: Tool Call (if applicable)
```bash
curl -X POST http://<spark-host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<served-name>",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
```
| Result | Pass / Fail |
|---|---|
| HTTP status | |
| tool_calls returned? | |
| Correct function name? | |
| Correct arguments? | |

## Loca Agent Loop Test

| Test | Pass / Fail | Notes |
|---|---|---|
| Smoke ("hello, what model are you?") | | |
| Tool call (read a file) | | |
| File operation | | |
| Terminal command | | |
| Multi-turn (3+ turns) | | |

## Concurrency Test (if max_num_seqs > 1)

| Concurrent | Total tok/s | Per-req tok/s | Peak tok/s |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 4 | | | |
| 8 (if supported) | | | |

## Memory Profile

| Measurement | Value |
|---|---|
| Model weights | |
| GPU memory used | |
| Free RAM | |
| KV cache pool | |

## Verdict

- [ ] PASS — Recipe locked, STATE.md updated
- [ ] FAIL — Issue documented, recipe remains draft
- [ ] PARTIAL — Some tests pass, needs debugging

## Notes

[Any observations, issues, crash logs, or things to try next]