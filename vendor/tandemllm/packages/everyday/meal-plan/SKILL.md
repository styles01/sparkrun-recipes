---
name: meal-plan
description: Make a meal plan for the week or a few days with a matching shopping list - for the user's number of people, budget, diet (vegetarian, halal, no pork, gluten-free, diabetes-friendly), time to cook and what is already in the kitchen - with simple dishes and leftovers used on purpose.
---
# Meal plan and shopping list

Use this when the user asks "what should we eat this week", "meal plan for a family of four", "cheap dinners for 5 days", "a shopping list for the week", or "I have chicken, rice and spinach, plan three dinners".

Steps:
1. Fix the frame: days (default 5 dinners, or 7 days with breakfast and lunch when asked), number of eaters (adults and children), diet rules, allergies, budget (low, middle), cooking time per evening (default 30 minutes), what is in the fridge. State your assumptions in one line.
2. Pick dishes that use the same ingredients twice (a roast chicken on Monday, chicken soup on Tuesday; rice cooked once for two days). One new or special dish a week at most; the rest simple and known.
3. Vary the base across the days: one pasta, one rice, one potato, one bread or wrap, one soup or salad day; vary the protein (legumes, eggs, fish, chicken, meat) within the diet.
4. Write the plan: one line per meal with the dish, the time to cook, and a note if it makes leftovers for the next day.
5. Make the shopping list by shop area: fruit and vegetables, dairy and eggs, meat and fish, dry goods, frozen, other. Each item with a quantity for the number of eaters (grams, pieces, packs). Leave out what the user said they have.
6. Add 2 or 3 prep tips for the week (cook rice for two days, wash and cut vegetables on Sunday, freeze half the sauce).

Defaults:
- Portions per adult per meal: 80 to 100 g dry pasta or rice, 120 to 150 g meat or fish, 150 g vegetables; a child under 10 counts as half an adult.
- Budget low: beans, lentils, eggs, seasonal vegetables, oats, frozen vegetables and fish; meat at most twice a week.
- Diet rules are hard rules: halal or no pork means no pork, no gelatine, no alcohol in sauces; vegetarian means no meat or fish; vegan no animal products; gluten-free no wheat, barley, rye (check sauces and stock cubes). Say when a product needs a label check.
- Diabetes, pregnancy, kidney disease, a child under 1 year: give a general plan and say the doctor's or dietitian's rules come first.
- Do not invent prices; say "about" with a range only if asked.

Output template:
Plan for <days>, <eaters>, <diet>, <time>. Assumed: ...
Mon: <dish> (25 min) - leftovers for Tue lunch
Tue: ...
Shopping list:
Vegetables and fruit: ...
Dairy and eggs: ...
Meat and fish: ...
Dry goods: ...
Frozen: ...
Prep tips: 1 ... 2 ...

Example:
User: "5 cheap dinners for 2 adults, no pork, 30 min max."
Good shape: Mon lentil dal with rice (makes 2 portions extra); Tue chicken thighs with potatoes and green beans; Wed dal leftovers with flatbread and a salad; Thu tomato pasta with white beans; Fri egg fried rice with frozen peas and the leftover chicken; shopping list by area with quantities for two; prep: cook double rice on Monday, use the chicken twice.

Do:
- Reuse ingredients on purpose and say where.
- Give quantities on the list.
Don't:
- Don't plan 7 different cuisines with 40 ingredients.
- Don't break a diet rule "just a little".
- Don't give calorie counts unless asked; if asked, give rounded estimates and say they are estimates.

Before you answer, check: frame and assumptions stated; each day one line with time; diet rules kept; list by area with quantities; fridge items left out; prep tips; user's language.
