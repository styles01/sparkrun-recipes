---
name: recipe-cooking
description: Give a recipe or cooking help - what to cook from the ingredients the user has, a dish by name, a substitute for a missing ingredient, scaling a recipe for more or fewer people, fixing a dish that went wrong, cooking times and temperatures - with exact amounts and steps a beginner can follow.
---
# Recipes and cooking

Use this when the user asks "what can I make with", "recipe for", "how do I cook", "can I replace X with Y", "my sauce is too salty", "how long does ... take in the oven", or wants a dish adapted (vegetarian, halal, gluten-free, for 8 people).

Steps:
1. Fix the frame: the dish or the ingredients at hand, the number of people (default 2), diet rules, the time available, the equipment (oven, pan, pot, microwave). State assumptions in one line.
2. For "what can I make with": propose 3 dishes in one line each (name, time, what else is needed from a basic pantry: oil, salt, onion, garlic, a lemon, flour). Then give the full recipe for the first, or the one the user picks.
3. Recipe format: title with portions and time; ingredient list with exact amounts in grams, millilitres, pieces and spoons (and cups if the user uses them), in the order they are used; numbered steps, each one action with the time, the heat level and what to look for ("until the onion is soft and clear, 5 minutes"). Short steps: 6 to 12.
4. Add the three checks a beginner needs: how to know it is done (colour, texture, internal temperature for meat: chicken 74 °C, pork 70 °C, whole beef roast 55 to 60 °C), the one mistake that ruins this dish, and what to serve with it.
5. Substitutes: give the amount and the difference ("1 egg -> 3 tablespoons of yoghurt in cakes; the cake is denser"). Say when there is no good substitute.
6. Fixing: too salty -> add a peeled potato or more liquid and unsalted bulk; too spicy -> dairy, sugar or acid; too thin -> reduce or a starch slurry; too thick -> liquid, a spoon at a time; burnt -> move the unburnt part to a new pot, do not scrape.
7. Scaling: multiply the ingredients; keep salt and spices at 80 % and taste; the cooking time rises only a little for a larger pot and stays the same per piece in the oven.

Defaults:
- Oven temperatures in °C with fan or not stated; pan heat as low, medium, high. Measures metric; add the user's system when they use it.
- Hard rules: halal or no pork means no pork, lard, gelatine or alcohol; vegetarian no meat or fish; vegan no animal products; allergies as named. Say when a product (stock cube, sauce) needs a label check.
- Food safety in one line where it matters: chicken, minced meat, eggs, rice kept warm, reheating (steaming hot right through), cooling within 2 hours.
- Do not invent "traditional" claims or health claims.

Output template:
<Dish> - <n> portions - <time>
Ingredients:
- <amount> <item>
Steps:
1. ...
Done when: ... Don't: ... Serve with: ...

Example:
User: "I have eggs, spinach, feta and some bread. Dinner for 2, 20 minutes."
Good answer: three one-line options (spinach and feta omelette; shakshuka-style eggs if there are tomatoes; a spinach-feta toast with a fried egg), then the omelette in full: 4 eggs, 150 g spinach, 80 g feta, 1 tbsp oil, salt, pepper; 6 steps with heat and timing; done when the top is just set; don't: high heat, which makes it rubbery; serve with the toasted bread.

Do:
- Exact amounts and times; what to look for at each step.
- Keep the dish doable with the user's equipment.
Don't:
- Don't list 20 ingredients for a weeknight dish.
- Don't skip the diet rules.
- Don't give a story before the recipe.

Before you answer, check: portions and time in the title; amounts for every ingredient; numbered steps with heat, time and cue; done-when and the don't; diet rules kept; food safety line where needed; user's language.
