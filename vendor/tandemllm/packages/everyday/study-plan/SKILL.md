---
name: study-plan
description: Make a study plan for an exam, a course, a certificate, a driving test, a language level or a skill the user wants to learn - from the date, the material, the hours available and the current level - with a week-by-week plan, methods that work (recall, spacing, practice tests) and a check each week.
---
# Making a study plan

Use this when the user asks "how do I prepare for", "study plan for my exam in 6 weeks", "learn Python in 3 months", "plan to reach B2", "I have 10 chapters and 2 weeks", or "I keep starting and stopping".

Steps:
1. Fix the frame: the goal (the exam or level, the date), the material (chapters, topics, a syllabus if given), the hours per week the user really has, the current level, and the past problem (no time, forgetting, panic). State your assumptions in one line.
2. Do the arithmetic: weeks to the date, hours total, hours per topic; reserve the last 15 to 20 % of the time for review and practice tests, and one buffer week in 6 or a buffer day in 7.
3. Order the material: foundations first, then the heaviest-weighted topics, then the rest; weak topics get 1.5 times the hours.
4. Write the plan week by week: topics, hours, and what "done" means that week (a page of notes from memory, 20 practice questions at 70 %, a mock test in time).
5. Fix the method per session: 25 to 50 minutes of work, 5 to 10 minutes of break; learn by recalling (close the book, write what you know, then check), not by rereading; every session starts with 5 minutes of yesterday's material; every week ends with a 30-minute self-test; old topics come back after 1 day, 1 week, 1 month.
6. Give the weekly check as three questions (did I do the hours, did the self-test pass, what moves to next week) and the rule for falling behind (cut depth, not topics; drop the lowest-weight topic last).
7. For exam day: the last day is light review and sleep, not new material; the week before: two full mock runs under exam conditions.

Defaults:
- Hours per week: ask or use 6 for someone working full-time, 15 for a student in term, 25 in a study week.
- A chapter of a textbook: 2 to 3 hours for a first pass plus 1 hour of review later. A language level step (A2 to B1): about 150 to 200 hours. A driving theory test: 10 to 15 hours over 2 weeks.
- A topic the user names as hard goes early in the week and in the morning.
- Say honestly when the time is not enough and what to drop or aim for instead (a pass instead of a top mark, the next exam date).

Output template:
Goal: <exam, date>. Weeks: <n>. Hours: <per week>, <total>. Assumed: ...
Week 1 (<dates>): <topics>, <hours>; done when: ...
Week 2: ...
Last week: review, 2 mock tests, light last day.
Each session: 5 min recall of yesterday, 25 to 50 min work, break; end with 3 questions from memory.
Weekly check: ...
If behind: ...

Example:
User: "Exam in 4 weeks, 8 chapters, I work and have evenings, chapters 5 and 7 are hard."
Good shape: 4 weeks x 8 h = 32 h; 25 h for chapters, 7 h for review and 2 mocks; chapters 1 to 3 in week 1 (6 h) with 5 started early at 1.5x, 4 to 6 in week 2, 7 to 8 in week 3 with 5 revisited, week 4 review and mocks; done-when per week; sessions Mon, Wed, Fri evenings 1.5 h and Saturday 3 h; the check and the if-behind rule.

Do:
- Show the hours arithmetic.
- Make each week's "done when" checkable.
Don't:
- Don't plan rereading; plan recall and practice.
- Don't plan more hours than the user has.
- Don't skip the review reserve.

Before you answer, check: frame and assumptions; arithmetic; weeks with topics, hours and done-when; the session method; the weekly check and the if-behind rule; exam-week rule; user's language.
