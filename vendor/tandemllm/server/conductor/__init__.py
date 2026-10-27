"""The TandemLLM conductor: the second-level harness in the API layer.

The client sees one ordinary OpenAI stream. Behind it, the conductor watches each request at four
safe points and may act from a fixed action set: close a reasoning block, continue a turn that
ended empty, hold a tool call, add a hidden note at the prompt's tail, set the request's thinking
budget, or give up with one visible sentence. Nothing here touches `engine/` (kernels, verify,
drafters, router); the one exception is `ThinkBudget`'s detector arms, which only read tokens.

`monitor/` is the session monitor: its policy file, the events the rules read, the Tier 0 rules,
the per-request supervisor that applies modes and caps, the session state and the journal. With
`--conductor raw` none of it runs and the served output is the raw engine's.
"""
