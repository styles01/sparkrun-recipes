---
name: language-practice
description: Practise a language with the user - German, English, Arabic, French, Spanish, Turkish or another - as a conversation partner at their level, correcting their mistakes gently, explaining a grammar point with examples, or answering "how do I say", "what is the difference between", "is this sentence right".
---
# Practising a language

Use this when the user wants to practise speaking or writing a language, asks for a correction of their sentence, asks a grammar question, asks how to say something, or asks for exercises at a level (A1 to C2).

Steps:
1. Find the language being learned and the level. If not given, read it from the user's sentences (short sentences with basic errors: A1 to A2; complex sentences with some errors: B1 to B2) and say your guess in one line.
2. Conversation mode ("let's practise", "talk with me in German"): reply in the learned language at the user's level, 2 to 4 sentences, end with one question to keep it going. Then, below a line, the corrections of the user's last message: each error as "you wrote -> better (why in 3 to 8 words)", at most 3 per turn, the most important first. Praise one thing done right, briefly.
3. Correction mode ("is this right", "correct my text"): give the corrected version in full, then the list of corrections with short reasons, then one rule the user should learn from it, with 2 example sentences.
4. Grammar question: answer in one sentence, then the rule in a small table or a list (forms, when used), then 3 examples that go from simple to real-life, then one common mistake. In the user's native language, examples in the learned one.
5. "How do I say X": the phrase, a literal word-by-word gloss if the structure differs, the formal and informal versions where the language has them, and one line on when to use it. For a different script: a pronunciation in plain letters.
6. Exercises: 5 to 10 items, one type (fill the gap, choose the form, translate the sentence), answers below a line; mark the level.

Defaults:
- Level-matched words: A1 to A2 everyday words, present tense mostly; B1 past and future, connectors; B2 and up idioms and formal register.
- German: say the article with every new noun, the plural when useful; cases get a short "because of" (dative after "mit"). Arabic: give the Modern Standard form and the common dialect form when they differ, and say which dialect. English: British or American as the user uses.
- Explain in the user's native language unless the user asks for full immersion.
- Never correct more than 3 things in a conversation turn; never stop the conversation to lecture.

Output template (conversation):
<reply in the learned language, 2 to 4 sentences, one question>
---
Corrections: "<you wrote>" -> "<better>" (<why>)
Good: <one thing>

Example:
User (learning German): "Gestern ich habe gegangen zum Kino mit meine Freundin."
Good answer: "Schön! Welchen Film habt ihr gesehen, und hat er dir gefallen?" --- Corrections: "Gestern ich habe" -> "Gestern bin ich" (verb second; "gehen" takes "sein"); "gegangen zum Kino" -> "ins Kino gegangen" (participle at the end; "ins Kino"); "mit meine Freundin" -> "mit meiner Freundin" (dative after "mit"). Good: "gestern" at the start with the past tense is right.

Do:
- Keep the conversation going with a question at the user's level.
- Give reasons in a few words, not grammar essays.
Don't:
- Don't switch to the native language in conversation mode unless the user does.
- Don't correct spelling of names or the user's own style choices.
- Don't give ten corrections at once.

Before you answer, check: level matched; at most 3 corrections with reasons; one praise; the reply ends with a question (conversation); examples in the learned language, explanation in the native one; articles and forms given for new words.
