---
name: plan-feature
description: Planning a multi-file feature before building it - clarify the outcome, list files and steps in build order, a test for each step, the risks, an estimate, and when to stop planning and start.
---
# Planning a multi-file feature

Rules:
- One sentence for the outcome as the user will see it (a route, a screen, a command). If it cannot be written, ask one question, then plan.
- List the files to create or change, in build order: data model, storage, logic, API or UI, wiring, tests. Each step has a check that can be run (a test, a curl, a screenshot).
- Smallest end-to-end slice first: one field, one route, one button, working through all layers. Then widen.
- Name the risks (an unknown library, a migration on live data, a performance limit) and the first step that proves each.
- Estimate by counting: steps x minutes per step, plus 30 % for the unknowns. Write the number.

What goes wrong:
- Planning at the level of "implement backend, implement frontend": no order, no checks, nothing to run after step 1.
- Building all layers of all fields before anything runs end to end.
- A hidden dependency (a schema change, an auth rule) discovered at the last step.

Check: the plan lists files, order and a runnable check per step; step 1 runs within the first 10 minutes and produces evidence.

Done when:
- [ ] The plan (8 to 15 lines) in the message with a check per step and the estimate.
- [ ] Step 1 executed with its evidence (test output, curl, screenshot) in the same session.
- [ ] Risks listed with the step that retires each.

## Reference

Plan template:
```
Outcome: users can mark an item done from the list and see it filtered.
Estimate: 7 steps x 10 min + 30 % = 90 min.
1. migrations/0004_items_done.sql: add done bool default false. Check: migrate, \d items.
2. app/models.py: Item.done. Check: pytest tests/test_models.py.
3. app/routes/items.py: PATCH /items/{id} {done}. Check: curl PATCH -> 200, GET -> done true.
4. app/routes/items.py: GET /items?done=false filter. Check: curl both values, counts differ.
5. web/src/Row.svelte: a checkbox calling PATCH. Check: click in the browser, state read, network 200.
6. web/src/App.svelte: filter toggle. Check: toggle, list length changes, screenshots 1280/390.
7. tests/test_items_api.py + e2e click test. Check: suite summary 0 failed.
Risks: existing rows need default false (step 1 proves it on a copy); the list query must stay under 50 ms with the new filter (step 4: EXPLAIN).
```
Slice rule: after step 3 the feature works end to end through curl; after step 5 through the UI.
Questions worth one line before planning: who can do this (auth), what happens to old data, is there a limit (size, rate), where does it show up (which screen, which command).
Time boxes: a step over 2x its estimate is a signal to stop and report what blocks it, not to push on silently.

More rules:
- The plan is 8 to 15 lines. Then start the first step. The plan changes only when a run shows something new.

More pitfalls:
- Re-planning while thinking instead of running step 1; the plan grows and nothing is built.
