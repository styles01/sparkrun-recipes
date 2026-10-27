---
name: explain-simply
description: Explain a topic, word, idea, process, device, news item or document the user does not understand - as to a smart beginner, with one everyday comparison, the few terms that matter, and an example - at a length that fits the question.
---
# Explaining something

Use this when the user asks "what is", "how does ... work", "explain", "I don't understand", "like I'm five", "what does this mean", about anything from inflation to a heat pump to a word in a letter.

Steps:
1. Answer the question in the first sentence, in plain words, as if to a friend who is smart but new to the topic.
2. Give one comparison from everyday life that carries the main idea (a heat pump "moves heat like a fridge in reverse"). One, not three.
3. Explain how it works in 3 to 6 numbered steps or short sentences, in the order things happen.
4. Name the 2 or 3 terms the user will meet again, each with a 5-to-10-word meaning in brackets.
5. Give one concrete example with numbers if the topic has numbers (a price, a percentage, a time).
6. End with the one thing most people get wrong about it, or the one question to ask next, in one line. Nothing else.

Length defaults:
- A word or a short "what is": 3 to 5 lines. A "how does it work": 10 to 15 lines. "Explain this document": the first sentence says what the document is, then the parts in the document's order.
- "Like I'm five" or "for a child": sentences under 12 words, no numbers over 100, a comparison with toys, food or the playground, 6 to 10 lines.
- "In detail" or "I'm an engineer / a doctor": skip the comparison, keep the terms, add the limits and the exceptions.

Defaults:
- The user's language; if the term is from another language, give it once in the original with the translation.
- If the topic is disputed or uncertain, say so in one line and give the common view and the main other view.
- If you are not sure of a fact (a date, a number, a law), say "I am not sure" or give a range, and name where it can be checked (the official site, the manual, the doctor).
- Do not explain what the user did not ask about; one topic per answer.

Output template:
<the answer in one sentence>
Think of it like <comparison>.
How it works:
1. ...
Terms: <term> (<meaning>), ...
Example: ...
Common mistake / next question: ...

Example:
User: "What is an ETF?"
Good answer: "An ETF is a fund you can buy on the stock exchange that holds many shares at once, so one purchase spreads your money over hundreds of companies. Think of it like a basket of fruit instead of a single apple. How it works: 1 a company builds the basket to copy an index, 2 you buy a share of the basket through a broker, 3 its price moves with the companies inside, 4 you pay a small yearly fee, often under 0.5 %. Terms: index (a list of companies), broker (the app or bank you buy through). Example: 100 euros a month in a world ETF buys a slice of about 1,500 companies. Common mistake: an ETF can still lose value; it spreads risk, it does not remove it."

Do:
- Use short sentences and concrete nouns.
- Keep the comparison honest; say where it breaks if it would mislead.
Don't:
- Don't start with history or definitions from a dictionary.
- Don't pile on five comparisons or ten terms.
- Don't pretend certainty on numbers you do not know.

Before you answer, check: first sentence answers it; one comparison; steps in order; terms explained; an example with a number where possible; length fits; uncertainty named.
