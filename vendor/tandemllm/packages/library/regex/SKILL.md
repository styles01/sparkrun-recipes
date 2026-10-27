---
name: regex
description: Writing and testing regular expressions in Python and JavaScript - anchors, groups, character classes, greedy versus lazy, multiline flags, named groups, and testing against positive and negative examples before use.
---
# Regular expressions

Rules:
- Write the examples first: 5 strings that must match, 5 that must not, including the edges (empty, unicode, extra spaces). Test the pattern against them before using it.
- Anchor what should be whole: `^...$` (with `re.fullmatch` in Python). Escape literals: `\.`, `\(`, `\?`.
- Named groups `(?P<year>\d{4})` (Python) or `(?<year>\d{4})` (JS). Non-capturing `(?:...)` for grouping only.
- Lazy `.*?` for "up to the next X" inside a line; better: a negated class `[^"]*`.
- Flags on purpose: `re.I`, `re.M` (anchors per line), `re.S` (dot matches newline), `re.X` for a commented pattern over 40 characters.

What goes wrong:
- `.*` swallows the rest of the line; two fields merge.
- `\d` matches Arabic-Indic digits in Python 3 str mode; use `[0-9]` when ASCII is meant.
- Catastrophic backtracking from `(a+)+` or `(.*,)*` on a long non-matching input; test on a 10k-character miss.

Check: run the positive and negative examples through the pattern and print a table of expected and actual; time one long non-matching input (under 10 ms).

Done when:
- [ ] The example table (10 cases, expected vs actual) in the result, all equal.
- [ ] Timing of the pattern on a 10k-character non-matching string under 10 ms.
- [ ] Each capture group named and shown on one real input.

## Reference

Python harness:
```python
import re, time
PAT = re.compile(r"""^(?P<user>[\w.+-]+)@(?P<host>[\w-]+(?:\.[\w-]+)+)$""", re.X)
yes = ["a@b.io", "first.last+tag@sub.example.com", "x_y@h-h.de", "A@B.CO", "1@2.3x"]
no = ["", "a@b", "@b.io", "a b@c.io", "a@b..io"]
for s, exp in [(s, True) for s in yes] + [(s, False) for s in no]:
    got = PAT.fullmatch(s) is not None
    print(f"{s!r:40} expected={exp} got={got} {'OK' if exp == got else 'FAIL'}")
m = PAT.fullmatch("first.last+tag@sub.example.com"); print(m.groupdict())
t0 = time.perf_counter(); PAT.fullmatch("a" * 10000 + "@" + "b" * 10000); print(f"{(time.perf_counter()-t0)*1000:.2f} ms")
```
Common pieces: ISO date `(?P<y>\d{4})-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])`; key=value lines `^(?P<k>[^=\s]+)\s*=\s*(?P<v>.*)$` with `re.M`; quoted string `"(?P<s>(?:[^"\\]|\\.)*)"`; whole words `\bword\b`.
Replace with a function: `re.sub(r"\d+", lambda m: str(int(m.group()) * 2), text)`.
JS: `const m = s.match(/^(?<y>\d{4})-(?<m>\d{2})$/); m?.groups.y`; all matches: `[...s.matchAll(/\d+/g)].map(m => m[0])`.
Split on a pattern: `re.split(r"[,;]\s*", s)`. Find all non-overlapping: `re.findall(r"\w+", s)`.

More rules:
- A regex is not a parser: HTML, JSON, CSV with quotes, nested brackets get a real parser.

More pitfalls:
- Forgetting raw strings in Python (`"\d"` is fine, `"\b"` is a backspace); always `r"..."`.
- JS `g` flag keeps `lastIndex` across `.test()` calls; use `match`/`matchAll` or reset.
