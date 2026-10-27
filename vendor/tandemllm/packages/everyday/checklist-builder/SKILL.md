---
name: checklist-builder
description: Make a complete, ordered checklist for a big life task - moving house, moving to another country, starting a new job, a new baby, a wedding, selling a car, packing for a trip, closing a flat, preparing for surgery or a long absence - with deadlines, documents, who to inform and what is easy to forget.
---
# A checklist for a big task

Use this when the user asks "what do I need to do for", "a checklist for", "don't let me forget anything when I", "packing list", or names a big change and asks where to start.

Steps:
1. Name the task and the date it happens (no date given: count in weeks before the day, or assume one and say it). Count backwards from that date.
2. Split the list into time blocks: as soon as possible, 4 weeks before, 2 weeks before, the last week, the day, the first week after. For packing: a week before, the night before, the morning, in the hand luggage.
3. Inside each block, list the items as short actions ("cancel the internet contract, notice period 1 month", "order 5 copies of the birth certificate"), each on its own line, with a checkbox. Put the items with the longest notice period or the longest waiting time first.
4. Add the groups that are easy to forget, as their own short lists: documents to have, people and offices to inform (address changes, insurers, bank, employer, school, doctor, subscriptions, the post forwarding), things to measure or photograph (the meter readings, the flat's condition), money (deposits, fees, refunds).
5. Mark the items that depend on the user's country or contract with "check" (the notice period, the registration deadline after moving, the school enrolment), and say where to check (the contract, the city's website).
6. End with the three items most people forget for this task and the first thing to do today.

Defaults:
- 20 to 40 items for a move, 30 to 60 for a move abroad, 15 to 30 for a new baby or a wedding, 20 to 35 for a packing list, 10 to 20 for a new job.
- Numbers for lists: 3 boxes per room per week of packing, 1 bag of clothes per 7 days of travel, 2 copies of every document plus a photo of each on the phone.
- Do not invent fees, deadlines or office names; "usually 2 weeks" with "check" is right; a wrong date is not.
- Keep the user's situation: children, pets, a car, a garden, a disability each add their own 3 to 5 items.

Output template:
<Task> on <date>. Today is day -<n>.
As soon as possible:
- [ ] ...
4 weeks before:
- [ ] ...
2 weeks before: ...
Last week: ...
The day: ...
First week after: ...
Documents: ... People and offices to inform: ... Money: ...
Most forgotten: 1 ... 2 ... 3 ...
Today: <one item>

Example:
User: "Moving to a new flat in the same city on 1 August, two adults, a cat."
Good shape: as soon as possible: give notice on the old flat (check the notice period), book a van or movers for 31 July, start a box count; 4 weeks: declutter, order boxes, tell the internet provider the new address (notice period), find a vet near the new flat; 2 weeks: pack room by room, label boxes with room and content, confirm the handover appointment for the old flat, post forwarding; last week: pack the kitchen last, a first-night box (kettle, bedding, cat food, litter tray), take meter readings with photos; the day: cat in a closed room until the van is gone, then the carrier, the keys; after: register the new address (check the deadline), update the bank, insurers, employer; most forgotten: the cat's chip registration address, the old flat's final meter reading, the deposit return in writing.

Do:
- Order by notice period and waiting time.
- Give each item as one action with a checkbox.
Don't:
- Don't write paragraphs; lists only.
- Don't invent legal deadlines or fees.
- Don't leave out the after-the-day block.

Before you answer, check: date and countdown; blocks in time order; items as actions with checkboxes; documents, people, money groups; "check" marks on country-specific items; the forgotten three and the today item; user's language.
