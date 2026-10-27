---
name: datetime-timezones
description: Handling dates, times and time zones in Python and JavaScript - storing UTC, aware datetimes, parsing and formatting ISO-8601, converting for display, durations, DST pitfalls and date arithmetic.
---
# Dates, times and time zones

Rules:
- Store and compute in UTC; convert to a zone only for display or for a user's "9:00 local" rule. Keep the zone name (`Europe/Berlin`), not an offset.
- Aware datetimes only: Python `datetime.now(timezone.utc)`, `ZoneInfo("Europe/Berlin")`; never naive `datetime.now()` or `utcnow()`.
- ISO-8601 on the wire: `2026-10-03T08:15:00Z` or with offset. Parse with `datetime.fromisoformat`; format with `.isoformat()`.
- Durations with `timedelta`; "one month later" is `relativedelta`, not 30 days.

What goes wrong:
- Comparing a naive and an aware datetime: `TypeError`, or silent wrong results when both are naive from different zones.
- A daily job at 02:30 local that runs twice or never on a DST change; schedule in UTC or handle the gap.
- JavaScript `new Date("2026-10-03")` is UTC midnight but `new Date("2026-10-03T00:00")` is local; display shifts by a day.

Check: print one instant as UTC, as Europe/Berlin and as a Unix timestamp; round-trip parse and format; test one date across a DST change (last Sunday of March or October).

Done when:
- [ ] The three representations of one instant printed side by side and consistent.
- [ ] A round-trip (format, parse, equal) shown for one value with an offset.
- [ ] A DST-crossing case computed and shown (the hour that goes missing or repeats).

## Reference

Python:
```python
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
BER = ZoneInfo("Europe/Berlin")
now = datetime.now(timezone.utc)
local = now.astimezone(BER)
print(now.isoformat(), local.isoformat(), int(now.timestamp()))
s = "2026-10-03T08:15:00+02:00"; d = datetime.fromisoformat(s); assert d.isoformat() == s
wall = datetime(2026, 3, 29, 2, 30, tzinfo=BER)     # the missing hour on the DST day
print(wall.astimezone(timezone.utc))                 # check what the library did with it
nine_local = datetime.combine(date(2026, 10, 3), datetime.min.time().replace(hour=9), tzinfo=BER).astimezone(timezone.utc)
from dateutil.relativedelta import relativedelta
print(date(2026, 1, 31) + relativedelta(months=1))   # 2026-02-28
```
SQL: Postgres `timestamptz` with `now()`; SQLite store `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')` text and compare as strings.
JavaScript:
```js
const t = new Date();                                   // an instant; getTime() in ms
t.toISOString();                                        // UTC ISO string
new Intl.DateTimeFormat('de-DE', { timeZone: 'Europe/Berlin', dateStyle: 'medium', timeStyle: 'short' }).format(t);
const d = new Date('2026-10-03T08:15:00+02:00');        // parse with an offset
Math.round(d.getTime() / 1000);                         // seconds for a Python side
```
Libraries when arithmetic grows: Python `dateutil`, `pendulum`; JS `date-fns-tz` or `Temporal` where available.
Freeze time in tests: `from freezegun import freeze_time; with freeze_time("2026-10-03T08:00:00Z"): ...`.

More rules:
- Dates without time (birthdays, deadlines) are `date`, not midnight datetimes; do not attach a zone.
- Tests with fixed instants; freeze time (`freezegun`, `monkeypatch`) instead of `sleep`.

More pitfalls:
- Unix timestamps in milliseconds (JS) vs seconds (Python) mixed up by a factor 1000.
- Database columns without a zone (`timestamp` vs `timestamptz`); stored local time read back as UTC.
