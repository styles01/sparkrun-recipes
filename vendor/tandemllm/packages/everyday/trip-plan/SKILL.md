---
name: trip-plan
description: Plan a trip, holiday, city break, road trip or day out - a day-by-day itinerary with areas, sights, food, travel times between stops and a rough budget - from the user's dates, place, travellers and interests, with what to book early and what to check before going.
---
# Planning a trip

Use this when the user asks for an itinerary, "what to do in", "plan my week in", "3 days in Rome with kids", a road trip route, or a packing-and-plan for a holiday.

Steps:
1. Fix the frame: place, dates or number of days, who travels (adults, children with ages, mobility), budget level (low, middle, high), interests the user named, where they sleep (if known). State the ones you had to assume in one line.
2. Group the place into areas so each day stays in one or two areas; no zig-zag across the city.
3. Write the days. Each day: a morning block, lunch, an afternoon block, evening; 2 to 4 stops a day, each stop with 3 to 10 words of why, the time it usually takes, and how to get to the next stop (walk 15 min, metro 20 min, car 1 h). Name a neighbourhood or dish for food rather than a specific restaurant unless you are sure it exists.
4. Build in rest: one slow afternoon per 3 days; with small children, a nap or playground stop every day; after a long flight, an easy first day.
5. Add the practical block: what to book before going (museum tickets with time slots, trains, a popular restaurant), the usual costs (entry, a meal, a day of transport, as ranges), what to check (opening days, holidays, the weather month, visa and documents, local payment habits).
6. End with a one-line alternative for a rainy day and one for a tired day.

Defaults:
- Days: arrival day and departure day count as half days.
- Walking: under 2 km between stops for families and older travellers, under 4 km for others.
- Budget levels per day per person, excluding the hotel: low under 50, middle 50 to 150, high above 150 (euros or the local equivalent); say that prices change and must be checked.
- Do not invent opening hours, prices, train times or restaurant names. Say "usually closed on Mondays" only if you know it; otherwise "check the opening days".
- Safety: for countries with travel warnings, a health rule (vaccination, tap water) or a season risk (heat, monsoon, hurricane), say it in one line and name the official source (the foreign office of the user's country).

Output template:
Trip: <place, dates, travellers, budget level>. Assumed: <...>
Day 1 (<area>): morning ... (1.5 h) - walk 10 min - ... lunch: <area or dish> - afternoon ... - evening ...
Day 2 ...
Book before: ...
Costs (per person, per day): ...
Check: ...
Rainy day: ... Tired day: ...

Example:
User: "3 days in Lisbon in May, two adults, like food and views, middle budget."
Good shape: Day 1 Alfama and the castle (morning), lunch in Alfama, afternoon the viewpoints and tram 28 area, evening Fado street; Day 2 Belém (monastery, tower, pastry), afternoon LX Factory, sunset at a miradouro; Day 3 Baixa and Chiado, a market lunch, a free afternoon or Sintra as a swap; book: the monastery time slot and a Sintra train if chosen; costs as ranges; check: May holidays, hills (good shoes), cash for small cafés.

Do:
- Keep each day in one or two areas; write the travel between stops.
- Put the must-see of the trip on day 1 or 2, not the last day.
Don't:
- Don't list 8 stops a day.
- Don't invent prices, times or names.
- Don't write a history essay about the place.

Before you answer, check: the frame and assumptions stated; days grouped by area with times and transfers; rest built in; book-before, costs, check lines present; nothing invented; user's language.
