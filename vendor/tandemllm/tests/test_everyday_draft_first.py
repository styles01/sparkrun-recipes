"""FX15 (UX3 W04, 2026-10-08): the everyday packages draft first; they never send the model to ask
before it answers.

The live case (0.4.6-test, packages v27): "Help me write a short cover letter for a junior data
analyst job at a bike company called Velo Nord." picked cv-cover-letter (0.996). Its step 1 said
"ask for it in one question if the whole document depends on it", everyday-core said "ask that one
question and stop there", and the model reasoned "I could just write it with placeholders ... but
I'd be inventing ... Let me use ask_user": Open WebUI's blocking form, the assistant message never
done, the user's next two messages queued in the client and never sent.

The rule pinned here, by wording (no output filter, no keyword routing): everyday-core says to take
the likely value, state it, put a [placeholder] where only the user knows the fact, and never ask
through a tool or a form; no everyday package tells the model to ask before it drafts. The quiz
turn ("ask one question only, then stop and wait") and the conversation practice question are the
task itself and stay.

Run: python tests/test_everyday_draft_first.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVERYDAY = os.path.join(os.path.dirname(HERE), "packages", "everyday")

# a sentence that sends the model to ask before it answers
ASK_FIRST = re.compile(
    r"ask (that|the|for it in) one question|ask that one question|and stop there|"
    r"ask for (it|the one|the age|the date)|ask if missing|ask the age if|"
    r"\(that is the one question\)|ask the user to (drop|pick)", re.I)
# the task itself is a question to the user: a quiz turn, a practice conversation
TASK_QUESTIONS = {"flashcards-quiz", "language-practice"}


def _texts():
    for n in sorted(os.listdir(EVERYDAY)):
        p = os.path.join(EVERYDAY, n, "SKILL.md")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                yield n, f.read()


def test_no_everyday_package_asks_before_it_drafts():
    bad = []
    for n, text in _texts():
        if n in TASK_QUESTIONS:
            continue
        for m in ASK_FIRST.finditer(text):
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]
            bad.append(f"{n}: {line.strip()[:160]}")
    assert not bad, "\n".join(bad)


def test_everyday_core_drafts_and_never_asks_through_a_form():
    text = dict(_texts())["everyday-core"]
    low = text.lower()
    assert "answer anyway" in low and "placeholder" in low
    assert "never ask through a tool, a form or a list of questions" in low
    assert "never stop a draft to ask first" in low


def test_the_cover_letter_is_written_with_placeholders():
    text = dict(_texts())["cv-cover-letter"]
    assert "never a reason to stop" in text and "write the whole document now" in text
    assert "the draft with placeholders comes first, every time" in text


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name:64s} ok")
            passed += 1
    print(f"{passed} passed")
