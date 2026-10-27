---
name: git-workflow
description: Day-to-day git for a coding task - small commits with clear messages, a branch per task, reading status and diff before committing, resolving merge conflicts, and never committing secrets or generated files.
---
# git workflow

Rules:
- Read the project's rules first (README, CONTRIBUTING, AGENTS.md): branch names, commit format, whether pushing is allowed. Never push or open a PR unless told to.
- One branch per task from the main branch: `git switch -c feat/short-name`.
- Commit small and often: one logical change per commit, imperative message under 70 characters, a body when the why is not obvious.
- Before each commit: `git status`, `git diff --staged`. Stage files by name, not `git add -A`, so no secret, `.env`, build output or data file goes in.

What goes wrong:
- A secret or a large binary committed once stays in history; stop, remove it, and tell the user before anything is pushed.
- A commit on the wrong branch; `git switch -c right-branch`, then reset the old one to `origin/main`.
- Line endings or formatting churn hiding the real diff; commit formatting separately.

Check: `git status` clean except intended files, `git log --oneline -5` shows the new commits, `git diff main --stat` lists only the files touched, and the test suite passes on the committed state.

Done when:
- [ ] `git log --oneline` of the branch in the result with one-line messages that read as actions.
- [ ] `git diff main --stat` lists only intended files; no `.env`, data or build output.
- [ ] Tests run on the committed tree (summary line) after the last commit.

## Reference

```bash
git status --short && git log --oneline -10
git switch -c feat/items-filter
git add src/items.py tests/test_items.py
git diff --staged --stat
git commit -m "Add done filter to items list" -m "Users asked for open items only; the index on (user_id, done) exists."
```
Conflict:
```bash
git merge main            # or: git rebase main (only on an unpushed branch)
git diff --name-only --diff-filter=U      # files with conflicts
# edit: keep both intents, delete <<<<<<< ======= >>>>>>>
pytest -q && git add <file> && git merge --continue   # or git rebase --continue
```
Undo safely: `git restore <file>` (working copy), `git restore --staged <file>` (unstage), `git revert <sha>` (new commit undoing one), `git reset --soft HEAD~1` (uncommit, keep changes, unpushed only).
Secret check before commit: `git diff --staged | grep -nEi 'api[_-]?key|secret|password|BEGIN (RSA|OPENSSH)'` must print nothing.
Ignore generated files: `printf '.venv/\nnode_modules/\ndist/\n.env\n' >> .gitignore`.

More rules:
- Conflicts: open each file, keep both intents, remove the markers, run the tests, then `git add` and continue. Never resolve by taking one side blindly.
- Never rewrite history that was pushed. `git stash` for a quick context switch; `git log --oneline -20` to orient.

More pitfalls:
- `git add -A` sweeps in `node_modules`, `.venv`, `dist`; check `.gitignore` first.
