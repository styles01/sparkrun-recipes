---
name: translate
description: Translate a text, message, document or phrase between languages (for example German, English, Arabic, Turkish, French, Spanish) keeping the meaning, tone and layout, with a note on words that have no exact match - or explain what a foreign text says.
---
# Translating

Use this when the user asks to translate something, wants to know what a text in another language says, or asks how to say a phrase in another language.

Steps:
1. Find the target language. If the user did not say it, translate into the language the user writes in. If the source is in that language already, translate into English (or the other language the conversation shows) and say in one line that you chose it.
2. Read the whole source first. Translate meaning, not word by word. Keep the tone: formal stays formal, a joke stays a joke, a rude message stays rude (do not soften it).
3. Keep the layout: paragraphs, lists, headings, line breaks, bold. Keep names, brands, addresses, numbers, dates, email addresses and links unchanged; convert date formats and units only if the user asks.
4. Give the translation in full, ready to copy. Nothing before it.
5. After it, up to 5 short notes only when needed: a word with no exact match and what you chose; a formal/informal choice ("Sie" or "du", "vous" or "tu") you had to make, and the other option; a part of the source that is unclear or has a typo and how you read it.
6. For a short phrase ("how do I say ... in German?"): give the phrase, a pronunciation hint in plain letters if the script differs, and one line on when to use it.

Defaults:
- Formality: for German, French, Spanish, Arabic and Turkish use the formal "you" to strangers, offices and customers, the informal one to friends and family; say which you used when the source does not show it.
- Arabic: Modern Standard Arabic for letters, forms and news; if the source is in a dialect and the user wants a dialect, say which one you used. Write right-to-left text as plain text, one paragraph per source paragraph.
- German: current spelling, nouns capitalised, the "ß" where it belongs.
- Do not add, explain or summarise inside the translation. Explanations go in the notes.
- A text too long to translate in one answer (over about 1,500 words): translate the first part in full and say where you stopped and that the rest follows on request.

Output template:
<translation>

Notes:
- <word or choice>: <what you did and the other option>

Example:
User: "Translate to German, to my doctor: 'I've had a sore throat for 5 days and now a fever of 38.5. Can I get an appointment tomorrow morning?'"
Good answer: "Sehr geehrte Frau Doktor / Sehr geehrter Herr Doktor, ich habe seit 5 Tagen Halsschmerzen und jetzt 38,5 Fieber. Könnte ich morgen Vormittag einen Termin bekommen?" / Notes: formal "Sie" to a doctor; "38.5" written "38,5" as in German.

Do:
- Keep the same number of paragraphs and the same order.
- Keep idioms as idioms: find the matching saying, not the literal words.
Don't:
- Don't improve, shorten or correct the content while translating.
- Don't mix two languages in one sentence of the output.
- Don't leave a word untranslated without a note.

Before you answer, check: nothing is missing or added; names, numbers and links are unchanged; the tone and formality match; the layout matches; the notes are only about real choices.
