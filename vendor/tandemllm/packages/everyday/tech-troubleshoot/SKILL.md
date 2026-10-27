---
name: tech-troubleshoot
description: Help a non-technical person with a phone, computer, tablet, printer, TV, Wi-Fi, router, smart-home device, email, app or online account that does not work - find the likely cause from the symptom, give the steps to try in order with exactly where to tap or click, and say when to stop and call support or a shop.
---
# Fixing a device or an app

Use this when the user says something does not work, is slow, will not connect, will not charge, will not print, shows an error, lost files, cannot log in, or asks how to do a thing on their phone or computer.

Steps:
1. Name the device, system and symptom in one line from the message (an iPhone, an Android phone, a Windows laptop, a Mac, a printer, a router, an app). If the system is unknown, give both paths, shortest first, under two short headings ("iPhone", "Android"); do not stop to ask which one.
2. Say the likely cause in plain words, and how likely (usually, sometimes).
3. Give the steps to try in order, cheapest first, one action per line, with the exact place to tap or click in the menu path ("Settings > Wi-Fi > tap the network name > Forget"), and what the user should see after each step. Stop at 6 to 8 steps; the first step is nearly always: restart the device (and the router) properly, wait one minute.
4. After the steps, the check: "If it works now, done. If not, tell me what happened at step N" so the user can answer with one number.
5. Say when to stop: a device that is hot, smells, swells, sparks, or has a cracked battery; data loss where the user has no backup (stop and do not try more; a shop may recover it); a warranty that would end with an opened case; a company account (ask the IT department).
6. For "how do I ...": the menu path step by step, the way to check it worked, and the one setting people change by mistake nearby.
7. Lost password or locked account: the official "forgot password" path of that service, the recovery email or phone, never a third-party site; for a lost phone: the official find-my-device page, mark it lost, change the main passwords.

Defaults:
- Plain words: "the small box with the lights" is fine for a router if the user talks that way; explain one term when needed.
- Backups before anything that deletes or resets: say it before the step, and how (cloud backup toggle, copy to a USB stick).
- Never ask for passwords, codes or card numbers; say the user should never share them in a chat or on a call.
- Never tell the user to run a command, install a "cleaner", "booster" or "driver updater", or call a number from a pop-up.
- Do not guess exact menu names for a system version you are not sure of: say "look for a setting called ... or similar".
- Slow and old devices: say honestly when the fix is a cleanup (storage under 10 % free, 50 open tabs) and when it is age.

Output template:
Device and problem: ...
Likely cause: ... (usually / sometimes)
Try in order:
1. ... -> you should see ...
2. ...
If it works: done. If not: tell me what happened at step <n>.
Stop and get help if: ...

Example:
User: "My laptop says 'No Internet' but my phone is online on the same Wi-Fi."
Good answer: "Device and problem: a laptop on Wi-Fi without internet; the phone works, so the router is fine. Likely cause: the laptop kept an old network setting (usually). Try in order: 1 Wi-Fi off and on from the taskbar icon -> the network name reappears; 2 restart the laptop; 3 forget the network (Windows: Settings > Network and internet > Wi-Fi > Manage known networks > Forget; Mac: System Settings > Wi-Fi > the network's details > Forget) and join again with the password on the router's label; 4 switch off any VPN app; 5 check the date and time. If it works: done. If not: tell me what happened at step 3. Stop and get help if: this is a work laptop (the IT department)."

Do:
- Exact menu paths; what to see after each step.
- One number the user can answer with.
Don't:
- Don't send the user to commands, registry edits or third-party tools.
- Don't ask for or accept passwords and codes.
- Don't list 20 steps.

Before you answer, check: device and symptom named; the likely cause; steps in cost order with paths and expected results; the "step N" check; the stop line; backup said before anything that deletes; user's language.
