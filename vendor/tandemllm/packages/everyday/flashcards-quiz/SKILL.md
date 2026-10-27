---
name: flashcards-quiz
description: Turn a text, chapter, notes, a topic or a vocabulary list into flashcards, a quiz, practice questions with answers, or a question-and-answer test - at the user's level and count - or run a quiz turn by turn: ask one question, wait, grade the answer, explain, continue.
---
# Flashcards and quizzes

Use this when the user asks "make flashcards", "quiz me", "test me on", "practice questions for", "turn this into questions", or pastes material and wants to learn it.

Steps:
1. Find the source and the count: a pasted text, a topic, a vocabulary list; the number asked (default 10 cards or questions; 20 for a vocabulary list; up to 30 for a chapter). The user's level (school, university, beginner) if given.
2. For flashcards: one fact per card; the front is a question or a term, the back is a short answer (1 to 15 words). Cover the text in order of importance: definitions, numbers, dates, causes and effects, steps of a process, exceptions. No card that can be answered from its own wording.
3. For a quiz: mix the types: 60 % short-answer, 30 % multiple choice with 4 options and one right (wrong options plausible, no "all of the above"), 10 % "explain in one sentence". Put the answers at the end, numbered, each with a one-line reason.
4. For "quiz me" turn by turn: ask one question only, then stop and wait. On the user's answer: say right or wrong, give the correct answer, one line of explanation, the score so far, and the next question. After the last: the score, the three things to review.
5. For vocabulary: front the word in the language being learned, back the translation plus one short example sentence; group by topic; add the gender or article for German, French, Spanish, Arabic nouns.
6. Give the cards in a plain format the user can paste into a flashcard app: one line per card, "front ; back". Say that in one line.

Defaults:
- Questions come from the source only; if the user gave a topic, use common textbook facts and say "from general knowledge, check against your course".
- Difficulty: start easy, end hard; mark hard ones with a star.
- Answers short and exact; a number has its unit; a date has the year.
- Never put the answer in the question. Never make two cards for the same fact.

Output template (cards):
1. <front> ; <back>
2. ...
Paste these into your card app as "front ; back", one per line.

Output template (quiz):
Questions:
1. ...
2. (a) ... (b) ... (c) ... (d) ...
Answers:
1. <answer> - <why>
...

Example:
User pastes a page about the water cycle and asks for 8 flashcards.
Good cards: "What drives evaporation from oceans and lakes? ; the sun's heat" / "Water vapour rising and cooling forms what? ; clouds (condensation)" / "Name the four stages in order ; evaporation, condensation, precipitation, collection" / "What share of Earth's water is fresh? ; about 3 %" and four more, each a single fact from the page.

Do:
- One fact per card; exact short answers.
- Cover numbers, names and steps, not only definitions.
Don't:
- Don't ask vague questions ("what is important about X").
- Don't ask several questions at once in the turn-by-turn mode.
- Don't invent facts not in the source without saying so.

Before you answer, check: the count matches; one fact per card; answers short and exact; quiz types mixed with answers and reasons; the paste format line; in the user's language (vocabulary in both).
