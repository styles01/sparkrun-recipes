---
name: compare-options
description: Compare two or more things for the user - products, phones, cars, apps, subscriptions, cities, job offers, insurance plans, ways of doing something - on the points that matter for their case, as a table with a clear recommendation and what it depends on.
---
# Comparing options

Use this when the user asks "A or B", "which is better", "differences between", "pros and cons of", or wants a comparison of named options.

Steps:
1. List the options as the user named them. If the user named only one and asks "is it good", add the 1 or 2 usual alternatives and say you added them.
2. Pick the 4 to 6 points that matter for this user (price, running cost, size, time, effort, risk, what it is for). Take the points from the user's situation if they gave one ("for a family of four", "I travel a lot"); otherwise use the usual ones for that kind of thing and say so in one line.
3. Fill a table: one row per point, one column per option. Each cell: a fact or a short judgement (3 to 8 words). Mark the best cell per row with a plus sign when there is a clear winner.
4. Below the table, write the recommendation in one sentence: "Choose A if ..., B if ...". Name the one condition that flips it.
5. State what you are not sure about (current prices, this year's models, local rules) in one line and where to check.
6. Stop there unless the user asked for more.

Defaults:
- 2 to 4 options. More than 4: group them, or compare the 4 most likely and name the ones left out.
- Do not invent prices, specifications, ratings or dates. Use ranges or "usually" when you do not know the exact value, and say that it must be checked.
- Keep facts and judgement apart: "16 GB" is a fact, "enough for most" is a judgement.
- When the options are equally good, say so and give the tie-breaker (the one the user already has, the cheaper one, the one easier to leave).
- No marketing words; no "it depends" without saying on what.

Output template:
| | Option A | Option B |
|-|-|-|
| Point 1 | ... | ... |
| ... | | |
Recommendation: <one sentence with the condition>.
Not sure about: <what and where to check>.

Example:
User: "Electric kettle or stovetop kettle? I live alone, small kitchen."
Good answer: table with rows price (stovetop usually cheaper), speed (electric faster for one cup), counter space (stovetop takes none), safety (electric switches off), noise and durability; "Choose electric if you boil water several times a day and have a free socket; stovetop if counter space is the problem and you have a gas stove." Not sure about: current prices in your shop.

Do:
- Put the user's own criteria first in the table.
- Give the answer, not a list of considerations only.
Don't:
- Don't fill the table with the same word ("good") in every cell.
- Don't invent numbers.
- Don't add options the user did not ask for beyond the usual one or two.

Before you answer, check: options as named; 4 to 6 rows; cells short; one recommendation with a condition; uncertainty named; in the user's language.
