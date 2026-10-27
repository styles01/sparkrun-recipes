---
name: proofread
description: Correct spelling, grammar, punctuation and wording in a text the user gives, in any language, without changing its meaning or style - returning the corrected text and a short list of what was fixed.
---
# Proofreading

Use this when the user asks to check, correct, fix or polish a text: an email, an essay, a form, a message in a foreign language, a translation they wrote themselves.

Steps:
1. Read the whole text once before changing anything. Keep the user's language, tone, person ("I", "we") and format.
2. Fix errors only: spelling, grammar (agreement, tense, case, articles, gender), punctuation, wrong words, word order, missing words, double words, wrong capitals.
3. Do not change style, content or length. A sentence that is correct but clumsy stays as it is unless the user asked for "polish" or "improve"; then rewrite clumsy sentences gently and keep the meaning.
4. Give the corrected text in full, ready to copy.
5. Below it, list the fixes: up to 10 lines, each "wrong -> right (why, in 3 to 6 words)". Group repeated errors of the same kind in one line ("comma before 'dass', 4 times"). If there were no errors, say so in one line.
6. If something may be an error but you are not sure (a name, a technical term, a regional spelling), leave it and mention it in one line.

Defaults:
- Spelling standard follows the text: British or American English as the text uses it; German with the current rules (ss/ß, capitals for nouns); Arabic with hamza and ta marbuta written correctly.
- Numbers, names, quotes and addresses are not changed.
- A text over 400 words gets the corrected text and the list, but the list keeps only the error kinds, with one example each.

Output template:
<corrected text>

Fixes (<number>):
- <wrong> -> <right>: <why>
...
Not changed: <doubtful item, if any>

Example:
User: "Check: 'Me and him goes to the shop yesterday and buyed to apples.'"
Good answer: "He and I went to the shop yesterday and bought two apples." / Fixes (4): "Me and him" -> "He and I": subject pronouns; "goes" -> "went": past tense; "buyed" -> "bought": irregular verb; "to" -> "two": wrong word.

Do:
- Keep every meaning exactly, even when you would have said it differently.
- Keep the user's formatting: line breaks, lists, bold.
Don't:
- Don't rewrite correct sentences.
- Don't add content, greetings or sign-offs that were not there.
- Don't lecture about grammar; the "why" is 3 to 6 words.

Before you answer, check: the corrected text has the same meaning and length; every fix in the list is in the text; nothing correct was changed; the language and spelling standard are the user's.
