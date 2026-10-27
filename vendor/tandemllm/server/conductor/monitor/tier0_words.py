"""D1's detector at the word level, for a backend whose tokens the proxy never sees (proxy P4).

The engine's reasoning-loop detector (engine/spec.py `StallArms`) reads token ids. Behind the
proxy only the streamed text exists, so the same three arms run over words: a word is a maximal
run of non-whitespace characters, its id a hash of its text. The arms and their meaning are the
engine's; the numbers are not copied from the token arms, they are measured again on the opencode
database's reasoning by `tools/loop_words_fp.py` and live in the monitor policy's `D1.words`
block (ops/supervisor.yaml), which carries the calibration's date and counts:

    loop      a pattern of `min_size..max_size` words repeated back to back, counted only when
              the repeats span `min_span` words or more
    repeat    every `every` words past `min_words`: the share of the last `window` words'
              `gram`-grams that already occurred earlier in the block; fires at `act` or above on
              `in_a_row` checks in a row, reports `grey..act` once as an observation
    novelty   the share of new `gram`-grams in the last `window` words against the window before;
              fires below `below` past `min_words`

`calibrated: false` in the block (or no block) keeps every verdict in shadow whatever the
switches say: the proxy journals what it would have done and acts on nothing.

Standard library only; text in, a reason out. Per word: a comparison per pattern size, one hash,
one append.
"""

from __future__ import annotations

import re

# The block the proxy uses when the policy document carries none: tools/loop_words_fp.py's run of
# 2026-10-04 over the opencode database (reasoning of at least 200 words: 2,674 finished turns that
# compress under 5x, 20 loops: turns cut by the token limit inside the reasoning or compressing 8x
# or more). The strictest of 32 configurations still fires on 6 of the finished turns and finds 11
# of the 20 loops. Whether those 6 are false alarms or loops the engine's own D1 closed is not
# known from the text alone (the token detector acted on that traffic), so the block ships
# `calibrated: false`: every verdict is journaled in shadow until the Compare view shows the
# precision (docs/proxy.md, "The word-level loop detector").
DEFAULTS = {
    "calibrated": False,
    "loop": {"min_size": 1, "max_size": 12, "min_count": 4, "min_span": 160},
    "repeat": {"gram": 16, "window": 200, "every": 50, "min_words": 2500, "act": 0.85,
               "grey": 0.4, "in_a_row": 3},
    "novelty": {"window": 100, "gram": 6, "min_words": 250, "below": 0.02},
    "measured": {"date": "2026-10-04", "legit": 2674, "loops": 20, "fired_legit": 6,
                 "fired_loops": 11},
}

_WORD = re.compile(r"\S+")


def words_cfg(policy_doc: dict | None) -> dict:
    """The `D1.words` block of a monitor policy document, else the shipped defaults."""
    try:
        w = ((policy_doc or {}).get("tasks") or {}).get("D1", {}).get("words")
    except AttributeError:
        w = None
    return dict(DEFAULTS, **w) if isinstance(w, dict) else dict(DEFAULTS)


class WordArms:
    """The three arms over word ids (the engine's StallArms, words for tokens)."""

    def __init__(self, cfg: dict):
        lp, self.rep, self.nov = cfg.get("loop"), cfg.get("repeat"), cfg.get("novelty")
        self.share: float | None = None
        self.grey_seen = False
        self._sizes: range = range(0)
        if lp:
            self._sizes = range(lp["min_size"], lp["max_size"] + 1)
            self._need = [0] * (lp["max_size"] + 1)
            for s in self._sizes:
                self._need[s] = (max(lp["min_count"], -(-lp["min_span"] // s)) - 1) * s
            self._runs = [0] * (lp["max_size"] + 1)
            self._tail: list[int] = []
            self._keep = lp["max_size"]
        if self.rep:
            self._grams: list[int] = []
            self._last: list[int] = []
            self._seen: set[int] = set()
            self._streak = 0
        if self.nov:
            self._recent: list[int] = []
            self._next = self.nov["min_words"]

    def feed(self, t: int, n: int) -> str | None:
        why = None
        if self._sizes:
            tail, runs, need = self._tail, self._runs, self._need
            k = len(tail)
            for s in self._sizes:
                if s <= k and tail[-s] == t:
                    runs[s] += 1
                    if why is None and runs[s] >= need[s]:
                        why = f"wloop(size={s} count={need[s] // s + 1} span={need[s] + s})"
                else:
                    runs[s] = 0
            tail.append(t)
            if k >= self._keep:
                del tail[0]
        if self.rep:
            r = self._repeat(t, n)
            why = r if why is None else why
        if self.nov:
            w = self.nov["window"]
            rec = self._recent
            rec.append(t)
            if len(rec) > 2 * w:
                del rec[:-2 * w]
            if why is None and n >= self._next and len(rec) >= 2 * w:
                self._next = n + w // 2
                v = self._novelty()
                if v < self.nov["below"]:
                    why = f"wnovelty({v:.2f})"
        return why

    def _repeat(self, t: int, n: int) -> str | None:
        r, last = self.rep, self._last
        g, w = r["gram"], r["window"]
        last.append(t)
        if len(last) > g:
            del last[0]
        if n >= g:
            self._grams.append(hash(tuple(last)))
        j = n - w - g
        if j >= 0:
            self._seen.add(self._grams[j])
        if n < r["min_words"] or n < w or n % r["every"]:
            return None
        window = self._grams[n - w:n - g + 1]
        if not window:
            return None
        v = sum(1 for h in window if h in self._seen) / len(window)
        self.share = v
        if r["grey"] <= v < r["act"]:
            self.grey_seen = True
        self._streak = self._streak + 1 if v >= r["act"] else 0
        if self._streak >= r["in_a_row"]:
            return f"wrepeat{g}(share={v:.2f} at={n})"
        return None

    def _novelty(self) -> float:
        w, g = self.nov["window"], self.nov["gram"]
        last, prev = self._recent[-w:], self._recent[-2 * w:-w]
        n = w - g
        if n <= 0:
            return 1.0
        new = {tuple(last[i:i + g]) for i in range(n)}
        old = {tuple(prev[i:i + g]) for i in range(n)}
        return len(new - old) / n


class WordLoop:
    """One reasoning block's detector: `feed(text)` with each streamed piece of the reasoning;
    returns the reason once an arm fires, else None. `words` is the block's length in words."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or dict(DEFAULTS)
        self.arms = WordArms(self.cfg)
        self.words = 0
        self.reason: str | None = None
        self._part = ""

    def feed(self, text: str) -> str | None:
        if self.reason is not None or not text:
            return None
        buf = self._part + text
        # the last run of non-whitespace may continue in the next piece: keep it back
        cut = len(buf)
        while cut > 0 and not buf[cut - 1].isspace():
            cut -= 1
        done, self._part = buf[:cut], buf[cut:]
        for m in _WORD.finditer(done):
            self.words += 1
            why = self.arms.feed(hash(m.group(0)), self.words)
            if why is not None:
                self.reason = why
                return why
        return None

    def flush(self) -> str | None:
        """The block ended: the last word counts too."""
        if self._part and self.reason is None:
            part, self._part = self._part, ""
            return self.feed(part + " ")
        return None

    @property
    def share(self) -> float | None:
        return self.arms.share
