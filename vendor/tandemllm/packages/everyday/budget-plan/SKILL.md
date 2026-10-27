---
name: budget-plan
description: Set up a personal or household budget, a saving plan for a goal (a trip, a car, an emergency fund), a plan to pay off debts, or check whether the user can afford something - from their income and costs, with the numbers worked out and shown, and the limits of what a chat can say about money.
---
# Money: budget, saving, affording

Use this when the user asks "help me budget", "can I afford", "how do I save for", "how to pay off my debts", "where does my money go", "how much should I spend on rent", or gives income and expenses and asks what to do.

Steps:
1. Take the numbers the user gave: monthly net income, fixed costs (rent, utilities, insurance, transport, loans, subscriptions), variable costs (food, clothes, going out), savings, debts with interest rates. Missing numbers: use a typical value or a placeholder, say which in one line, and give the whole plan as an example until it is filled; name the one number that would change it most.
2. Do the arithmetic and show it: income minus fixed minus variable = what is left. Round to whole units. Show the share of income for the big items in percent.
3. Compare with simple guides and say they are guides, not rules: rent up to about a third of net income; fixed costs together up to half; saving 10 to 20 % if possible; an emergency fund of 3 months of costs before other goals.
4. For a goal: amount divided by months to the date = the monthly saving; if it does not fit, show the two ways: a later date or a smaller amount, each with the number.
5. For debts: list them by interest rate, highest first; pay minimums on all, extra on the highest rate; show when the first one is gone at the user's extra amount. Name the alternative ("smallest first") in one line.
6. Give 3 to 5 concrete cuts or moves from the user's own list, each with the monthly amount it frees. No generic "cook at home" without a number from their list.
7. End with the one line that says what to do this month.

Defaults and limits:
- Use the user's currency and figures; never invent an income, a rent or a rate.
- No investment, tax, pension or insurance product advice: say what the kinds of options are and that a licensed adviser or the official body (the consumer advice centre, the tax office's site) must confirm; this matters for amounts above a few months' income.
- If the money does not cover the basics, or debts grow every month: say so plainly, name the free debt advice service type in the user's country (non-profit debt counselling), and the two things to do first (stop new credit, talk to the creditors before missing a payment).
- Do not shame; numbers, not judgement.

Output template:
Income: ... Fixed: ... (x %) Variable: ... (x %) Left: ...
Compared with the usual guides: ...
Goal / debts: <the arithmetic, month by month if short>
Moves: 1 <what> frees <amount>/month ...
This month: <one action>
Note: <the limit line: who must confirm what>

Example:
User: "Net 2,400, rent 950, other fixed 420, food and stuff about 600. I want 3,000 for a trip in 10 months. Realistic?"
Good answer: "2,400 - 950 - 420 - 600 = 430 left a month. Rent is 40 % of your income, above the usual third, so the rest has to be tight. The trip needs 3,000 / 10 = 300 a month, which fits in the 430 but leaves 130 for everything else and no emergency fund. Two ways: 300 a month and the trip as planned; or 200 a month and the trip in 15 months while 100 goes to a small emergency fund. Moves: check the 420 fixed for a subscription you do not use (often 20 to 40); food at 600 for one person has room, 500 is realistic (frees 100). This month: open a separate account and set a standing order of 300 on payday."

Do:
- Show every calculation.
- Use the user's own categories and numbers.
Don't:
- Don't recommend a specific product, fund, bank or loan.
- Don't invent interest rates or prices.
- Don't give a lecture about money habits.

Before you answer, check: arithmetic shown and right; shares in percent; guides called guides; goal or debt plan with months; moves with amounts; the one action; the limit line present.
