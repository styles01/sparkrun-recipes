---
name: sql-schema-migrations
description: Designing a relational schema in SQLite or PostgreSQL and changing it with migrations - tables, keys, constraints, timestamps, versioned migration files, applying and rolling back safely.
---
# Schema and migrations

Rules:
- Every table: an `id` primary key, `created_at`, `updated_at`. Foreign keys with `ON DELETE` decided on purpose (CASCADE for owned rows, RESTRICT for references).
- Constraints in the database, not only in code: `NOT NULL`, `UNIQUE`, `CHECK`. The database is the last line.
- Migrations are numbered files applied in order and recorded in a `schema_migrations` table. Never edit an applied migration; add a new one.
- Each migration is small and reversible where possible (an `up` and a `down`). Data migrations separate from schema migrations.

What goes wrong:
- Foreign keys silently off in SQLite; orphans appear.
- A `NOT NULL` column added to a table with rows and no default; the migration fails halfway.
- Timestamps stored in local time without a zone; store UTC (Postgres `timestamptz`, SQLite ISO-8601 text ending in `Z`).

Check: apply all migrations on an empty database and on a copy with data; `\d table` or `.schema table` shows the expected columns and constraints; insert a row that violates each constraint and read the error.

Done when:
- [ ] Migration run output: every file applied, exit 0, on an empty database and on a copy with rows.
- [ ] Schema dump of the changed tables in the result.
- [ ] One rejected insert per new constraint shown with its error text.

## Reference

Migration file `migrations/0003_items_done.sql`:
```sql
-- up
ALTER TABLE items ADD COLUMN done INTEGER NOT NULL DEFAULT 0 CHECK (done IN (0, 1));
CREATE INDEX IF NOT EXISTS idx_items_user_done ON items(user_id, done);
```
Minimal runner (Python, SQLite):
```python
import sqlite3, pathlib
def migrate(path="app.db", folder="migrations"):
    con = sqlite3.connect(path); con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")
    done = {r[0] for r in con.execute("SELECT name FROM schema_migrations")}
    for f in sorted(pathlib.Path(folder).glob("*.sql")):
        if f.name in done: continue
        with con: con.executescript(f.read_text()); con.execute("INSERT INTO schema_migrations(name) VALUES (?)", (f.name,))
        print("applied", f.name)
```
Postgres table:
```sql
CREATE TABLE items (
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  text TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 200),
  done BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```
Inspect: SQLite `sqlite3 app.db '.schema items'`; Postgres `psql "$DATABASE_URL" -c '\d items'`.
Tools when the project has them: Alembic (`alembic revision --autogenerate -m msg`, read the file, `alembic upgrade head`), Prisma, golang-migrate, dbmate.

More rules:
- SQLite: `PRAGMA foreign_keys = ON` per connection; `ALTER TABLE` is limited, so a column change is create-new, copy, drop, rename inside one transaction.
- Run migrations on a copy of the real data before production; back up first.

More pitfalls:
- Money as float; use integer cents or `NUMERIC(12,2)`.
- The ORM auto-generates a migration that drops a table because a model was renamed; read the generated SQL.
