---
name: tandem-library
description: The TandemLLM skill library - short rules, pitfalls and snippets for common stacks and practices, and how to search it and read a skill from the shell before unfamiliar work.
---
# The skill library

The runtime keeps a library of short skills: web pages, canvas, Three.js, games, React, Svelte, FastAPI, Express, SQL, pandas, pytest, bash, git, Docker, debugging, profiling, security and more. It may add one to your prompt; search it yourself before work you are unsure about.

```bash
L=${TANDEM_URL:-{tandem_url}}/v1/library
curl -s "$L/search?q=canvas+game+collisions&k=5"   # [{name, description, score}]
curl -s "$L/browser-game"                          # {body, reference}
```

- No key is needed. Search with the task's own words, read the best one or two, follow their rules and their "Done when".
- The `reference` holds snippets and commands: read it when you start that code.
