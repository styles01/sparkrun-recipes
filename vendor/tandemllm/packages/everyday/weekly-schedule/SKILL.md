---
name: weekly-schedule
description: Build a weekly or daily plan for the user - fitting work, study, family, sport, chores and rest into real hours - or sort a to-do list by priority into a doable day, with buffers and a plain table they can follow.
---
# Planning the week or the day

Use this when the user says "plan my week", "make me a schedule", "I have too much to do", "how do I fit X in", "organise my day", or pastes a to-do list and asks what to do first.

Steps:
1. Collect the fixed blocks: work or school hours, commute, sleep, meals, appointments, children's times. Put them in first; they do not move.
2. Collect the wishes with their size: how many hours a week the user wants for each (sport 3 x 45 min, study 5 h, family evening, admin). If the user gave no sizes, use the defaults below and say so.
3. Place the wishes into the free gaps: hard things (study, deep work) in the user's best hours (morning for most people), sport at the same time each day, admin in one block, rest every day.
4. Keep buffers: 15 minutes between blocks, one free evening, one half day at the weekend with nothing planned.
5. Write the plan as a table: rows are time slots (one hour or half hour), columns are the days, cells are short (2 to 4 words). Then under the table: the three rules of this week (for example "no phone in study blocks").
6. If the wishes do not fit, say what does not fit, in hours, and propose the cut yourself and mark it.
7. For a to-do list: sort into today (3 items at most), this week, later; give each today item a time slot; name the first thing to do now.

Defaults:
- Sleep 7 to 8 hours at the same time every day. Meals 30 minutes. A study or deep-work block 45 to 90 minutes with a 10-minute break. Sport 30 to 60 minutes, 3 times a week. Chores 30 minutes a day or one 2-hour block.
- A full day of work plus commute leaves about 3 usable hours; do not plan more than 2 of them.
- Children's schedules and partner's hours come from the user; never guess them.

Output template:
Fixed: <list>
Wishes: <list with hours>
| Time | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|-|-|-|-|-|-|-|-|
| 07:00 | ... | | | | | | |
Rules this week: 1 ... 2 ... 3 ...
Does not fit: <or "everything fits">

Example:
User: "Work 9 to 6, 40 min commute, want to study German 5 h a week, gym 3 times, and see friends. Plan it."
Good shape: fixed work and commute; German in five 1-hour blocks at 7:00 Mon to Fri before work (or three evenings plus Saturday morning if the user is not a morning person, state the choice); gym Mon, Wed, Fri 18:45; friends Thursday evening and Saturday; Sunday afternoon free; rules: phone away in study blocks, bag packed the night before, one skipped block is fine, two means move it.

Do:
- Use real clock times, not "sometime in the evening".
- Keep one free evening and one free half day.
Don't:
- Don't fill every hour.
- Don't plan 4 hours of study after a full work day.
- Don't add wishes the user did not name.

Before you answer, check: fixed blocks first; every wish placed or named as not fitting; buffers and the free slots; the table reads in one glance; the three rules; user's language.
