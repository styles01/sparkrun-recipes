---
name: summarize-text
description: Summarise a long text the user pasted or uploaded - an article, report, PDF, email thread, meeting notes, transcript, contract or chat - into the key points, decisions, open questions and next actions, at the length the user wants, without adding anything.
---
# Summarising a text

Use this when the user gives a text and asks for a summary, the key points, the gist, "what does this say", the decisions from a meeting, or "the TL;DR".

Steps:
1. Read the whole text before writing. Note what kind it is: article, report, meeting notes, email thread, contract, study, story. The shape of the summary follows the kind (below).
2. Pick the length: the user's number if given; else "short" = 3 to 5 bullet points; "summary" = one paragraph of 4 to 6 sentences plus 3 to 6 bullet points; "detailed" = one short paragraph per section of the text. Never longer than a quarter of the original.
3. Write the one-sentence answer first: what the text is and its main point.
4. Then the points, in the order of importance (not the order of the text), each one line, each a fact or decision from the text with its number, date or name if there is one.
5. For meeting notes, email threads and chats add three blocks: Decisions, Open questions, Next actions (who, what, by when). Leave a block out if it is empty and say "none".
6. End with "Not in the text:" only when the user asked something the text does not answer. Nothing else.

Defaults:
- Use the user's language for the summary even if the text is in another language; say the text's language in the first line if it differs.
- Keep the text's own terms and names; do not rename things.
- Quote only when the exact words matter (a deadline, a price, a legal condition), and mark it as a quote.
- A text that is cut off or partly unreadable: summarise what is there and say what is missing in one line.
- Never add opinion, background or advice unless asked. Never fill gaps with what "probably" happened.

Output template:
<one sentence: what it is and the main point>
Key points:
- ...
Decisions: ...   (meetings, threads)
Open questions: ...
Next actions: <who> - <what> - <by when>

Example:
User pastes 2 pages of meeting notes and asks "summary please".
Good answer: "Notes of the 12 May product meeting: the launch moves from June to 15 September because the payment test failed." Key points: 4 lines with dates and names. Decisions: 2 lines. Open questions: 1 line. Next actions: 3 lines "Anna - new test plan - 20 May".

Do:
- Put numbers, dates and names in the points; they are what readers look for.
- Keep every point checkable against the text.
Don't:
- Don't start with "This text is about".
- Don't repeat the same point in the paragraph and the bullets.
- Don't summarise the user's own question.

Before you answer, check: the first sentence carries the main point; every point is in the text; the length is within the limit; decisions and actions have names and dates; nothing added.
