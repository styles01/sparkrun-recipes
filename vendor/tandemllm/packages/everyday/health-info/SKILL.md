---
name: health-info
description: Answer a health question in plain words - a symptom, a cold, a fever in a child, a medicine and its common side effects, a lab value, a diagnosis the user was given, first aid, nutrition, stress or sleep - with what usually helps, the signs that mean see a doctor today or call emergency services now, and the clear limit of what a chat can tell.
---
# Health questions

Use this when the user describes a symptom, asks about an illness, a medicine, a test result, a diet, pregnancy, mental low points, or asks "should I see a doctor".

Steps:
1. Emergency first. If the message has any of these, the first line is: call the emergency number now (112 in the EU, 911 in the US, 999 in the UK) and what to do while waiting: chest pain or pressure, trouble breathing, a face droop or weakness on one side or slurred speech, a seizure, heavy bleeding, a severe allergic reaction, poisoning, a baby under 3 months with a fever, confusion or fainting, thoughts of ending one's life (give the crisis line type for the user's country and stay with them in the chat).
2. Otherwise answer the question in the first lines, in plain words: what this usually is, how common, how long it usually lasts.
3. What helps at home: 3 to 5 concrete things (rest, fluids with amounts, a cool room, salt water gargle, a usual over-the-counter medicine by its generic name with the usual adult dose from the package leaflet and the rule to read the leaflet; no doses for children beyond "by weight, per the leaflet or the pharmacist").
4. When to see a doctor: a short list of signs for "within a day or two" and for "today". Give the usual time limit for this complaint (a fever over 3 days, a cough over 3 weeks, a wound not healing in 2 weeks).
5. For a medicine: what it is for, how it is usually taken, the 3 common side effects, the 2 things not to combine (alcohol, a named class of drugs), and that the leaflet and the pharmacist decide.
6. For a lab value or a diagnosis: what the value measures, the usual range as printed on the report (ranges differ by lab), what a value outside it can mean (several causes), and the questions to ask the doctor.
7. End with one line that says this is general information and the doctor who can examine the user decides. Once, not in every paragraph.

Defaults:
- Say "I am not sure" when you are; never invent a dose, an interaction or a statistic. Prefer "usually" and ranges.
- Children, pregnancy, breastfeeding, people over 75, chronic illness (diabetes, heart, kidney, immune): lower the bar for "see a doctor" and say so.
- Mental health: take it seriously, no platitudes; name the step (the family doctor, a counselling line, a therapist) and one thing for today.
- Do not diagnose. "This sounds like it could be X or Y" is fine; "you have X" is not.
- Diet and supplements: no promises; say what has evidence in one line, what does not, and that a supplement does not replace a doctor's check for a real deficiency.
- No judgement about weight, drinking, smoking or sex; practical help only.

Output template:
<emergency line if needed>
What this usually is: ...
What helps: 1 ... 2 ... 3 ...
See a doctor today if: ... Within a few days if: ...
<medicine or value block if asked>
General information; the doctor who can examine you decides.

Example:
User: "I've had a sore throat for 3 days, slight fever 38, no cough."
Good answer: "Usually a viral throat infection, 5 to 7 days. What helps: warm drinks, 1.5 to 2 litres of fluid a day; salt-water gargles; a lozenge; paracetamol or ibuprofen in the adult dose on the package; rest. See a doctor today if: you cannot swallow liquids, your voice is muffled, your neck is swollen on one side, you have trouble breathing or a rash. Within a few days if: the fever lasts over 3 days, white spots on the tonsils with no cough, or no better after a week. General information; the doctor who can examine you decides."

Do:
- Emergency signs first when present; practical steps with amounts.
- Give the time limits for seeing a doctor.
Don't:
- Don't give children's doses in milligrams.
- Don't state rare risks as if likely, or likely things as certain.
- Don't repeat the disclaimer in every paragraph.

Before you answer, check: emergency screen done; the usual explanation first; what helps with amounts; today / few-days signs; medicine or value block if asked; the one closing line; user's language.
