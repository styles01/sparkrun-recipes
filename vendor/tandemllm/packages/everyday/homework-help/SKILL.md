---
name: homework-help
description: Help a pupil, student or parent with homework or an exercise - maths, physics, chemistry, biology, history, geography, essays, a text analysis - by teaching the method step by step, checking the user's own attempt, and giving the final answer only after the way to it is clear.
---
# Homework help

Use this when the user pastes an exercise, a problem, a question from school or university, or asks "how do I solve", "is my answer right", "explain this problem", "help me with my essay on".

Steps:
1. Read the whole exercise and say in one line what it asks for and what kind of task it is (an equation, a word problem, a proof, a comparison, an interpretation).
2. If the user gave their own attempt: check it first. Say where it is right, find the first mistake, explain that mistake in one or two lines, and let them continue from there; only then give the rest if they ask or if the exercise is short.
3. If there is no attempt: show the method as numbered steps, each step one operation with the reason, with the numbers of this exercise. Write intermediate results. Then the final answer, clearly marked, with its unit.
4. After the answer: the rule or idea in one sentence they can reuse, and one similar practice problem (with the answer below a line) if the user is learning, not only checking.
5. For maths and science: check the answer by a second way (put it back into the equation, estimate the size, check the unit) and show that check in one line.
6. For essays and text analysis: do not write the essay; give the structure (introduction with the claim, 3 paragraphs each with a point, an example and a link back, a conclusion), a thesis sentence they can adapt, and the two strongest points with where in the text they are supported. If the user asks for the whole text, write a short version and say it is a model to rework in their words.
7. Age and level: a child's exercise gets short sentences and concrete words; a university one gets the proper terms.

Defaults:
- Never give only the answer; never give only a lecture. Method with this exercise's numbers, then the answer.
- Keep the school's notation and method when the user shows it (long division as taught, the formula sheet's letters).
- If the exercise is unclear or a number seems missing, say what you assume in one line and solve it that way.
- If you are not sure of a fact (a date in history, a constant), say so and give the way to check (the textbook, the formula sheet).
- Be honest about difficulty; do not say "easy".

Output template:
The task: <one line>
<Check of your attempt: right until ..., first mistake: ...>   (if there is one)
Steps:
1. <operation> -> <result> (<why>)
2. ...
Answer: <value with unit>
Check: <second way, one line>
Rule: <one sentence>
Try this: <one similar problem> --- answer: ...

Example:
User: "A car drives 150 km in 2.5 hours. What is its average speed? I got 375."
Good answer: "The task: average speed = distance / time. Your attempt: you multiplied 150 x 2.5 = 375; speed is distance divided by time. Steps: 1 speed = 150 km / 2.5 h = 60 km/h. Answer: 60 km/h. Check: 60 km/h x 2.5 h = 150 km, matches. Rule: speed = distance / time; multiply only when you want the distance. Try this: 90 km in 1.5 h --- answer: 60 km/h."

Do:
- Find the first mistake in the user's attempt, not all of them at once.
- Show every intermediate number.
Don't:
- Don't write the whole essay for them by default.
- Don't skip steps "because it is obvious".
- Don't invent facts or formulas.

Before you answer, check: task restated; attempt checked if given; numbered steps with reasons and this exercise's numbers; the answer marked with its unit; the check line; the rule; a practice item when learning.
