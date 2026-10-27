---
name: decision-helper
description: Help the user make a personal decision they are stuck on - take the job or not, move cities, buy or rent, which course, break up or stay, keep the car or sell it - by setting out their options, what matters to them, the risks, and a reasoned recommendation they can disagree with.
---
# Helping with a decision

Use this when the user says "should I", "I can't decide", "what would you do", "help me decide", or describes a choice with worry and asks for a view.

Steps:
1. Write the decision as one question with the real options, including "wait" or "do nothing" when that is possible. Usually 2 or 3 options.
2. Pull what matters to the user from their message: 3 to 5 things (money, time, family, health, growth, freedom, safety). Use their words. If they gave none, use the usual ones for this kind of decision and say so.
3. Score each option on each thing with a simple word scale: better, same, worse, and give the reason in a few words. A table if 3 or more things.
4. Name the biggest risk of each option and whether it can be undone (a job can be left; a sold flat is gone).
5. Give a recommendation in one or two sentences, with the one fact that would change it. Say it plainly; the user asked for a view, not a list.
6. Suggest one small step the user can do this week that makes the decision easier (a trial, a conversation, a number to find out).
7. For a decision about health, money above a few months' income, law or safety: add that a professional should check the facts (doctor, tax adviser, lawyer), without turning the answer into a referral.

Defaults:
- Keep the user's values; do not add yours. Do not moralise.
- If the user is in distress (a breakup, grief, fear), start with one line of acknowledgement, then the structure, calmly.
- Never say "only you can decide" as the whole answer; say it once at the end if the choice is about values, after you gave a view.
- Missing facts: state the assumption and recommend under it; if another value would flip the recommendation, say which in one line ("If the commute is over an hour, the other option wins").

Output template:
The decision: <one question, the options>
What matters to you: <list>
| | Option A | Option B |
|-|-|-|
| <thing> | better - <why> | worse - <why> |
Biggest risk: A ... (undoable: yes/no). B ...
My view: <recommendation and what would change it>.
This week: <one small step>.

Example:
User: "Job offer: 20 % more money, but 1 h commute each way instead of 15 min. Two small kids. Take it?"
Good shape: the decision with options take / stay / negotiate remote days; what matters: money, time with the kids, energy, career; the table; the risk (the commute is 7.5 h a week; money is undoable, time is not); the view ("take it only if two remote days are possible; ask for them before you answer"); this week: ask the new employer about remote days and the real start time.

Do:
- Use the user's numbers; turn vague things into numbers where you can (hours per week, euros per year).
- Say what you would do, and why, in their terms.
Don't:
- Don't invent facts about the user's life.
- Don't hide behind "pros and cons" with no view.
- Don't give five options when the user has two.

Before you answer, check: one question with options; the user's own criteria; each option scored with reasons; risks with undoable yes/no; a clear view; one small step.
