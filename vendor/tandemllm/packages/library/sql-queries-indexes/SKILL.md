---
name: sql-queries-indexes
description: Writing correct and fast SQL queries - joins, aggregates, parameters against injection, pagination, N+1 avoidance, and choosing indexes with EXPLAIN in SQLite and PostgreSQL.
---
# SQL queries and indexes

Rules:
- Parameters always (`?` or `$1`), never string formatting with user values.
- Select the columns you need. `SELECT *` only in exploration.
- One query with a JOIN or `WHERE id IN (...)` instead of one query per row (N+1). Load the list, then the children in one call.
- Pagination by keyset (`WHERE id > $last ORDER BY id LIMIT 50`), not by large `OFFSET`.
- Index what the `WHERE` and `JOIN` filter on, in that order: equality columns first, then the range or sort column. One index per hot query shape; not every column.

What goes wrong:
- A `LEFT JOIN` with a condition in `WHERE` on the right table turns it into an inner join; move the condition into `ON`.
- `GROUP BY` with a column not in the group and not aggregated; wrong or rejected.
- `COUNT(*)` over the whole table on every page load; cache or keep a counter.

Check: run the query on real data with timing (`\timing` or `time`), show the plan before and after, and compare the row count against a simple counting query.

Done when:
- [ ] The query's rows and time on real data in the result.
- [ ] The `EXPLAIN` plan shows an index scan (or a seq scan justified by table size under 10k rows).
- [ ] The row count cross-checked by a second, simpler query.

## Reference

Children in one query (no N+1):
```sql
SELECT u.id, u.name, i.id AS item_id, i.text
FROM users u
LEFT JOIN items i ON i.user_id = u.id AND i.done = false
WHERE u.id = ANY($1)
ORDER BY u.id, i.id;
```
Aggregate per user with a filter:
```sql
SELECT user_id, COUNT(*) FILTER (WHERE done) AS done, COUNT(*) AS total
FROM items GROUP BY user_id HAVING COUNT(*) > 0;
```
Keyset pagination: `SELECT id, text FROM items WHERE user_id = $1 AND id > $2 ORDER BY id LIMIT 50;` with index `(user_id, id)`.
Plan:
```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT ... ;          -- Postgres
EXPLAIN QUERY PLAN SELECT ... ;                  -- SQLite
CREATE INDEX CONCURRENTLY idx_items_user_id ON items(user_id, id);   -- Postgres, no table lock
```
Python parameters:
```python
cur.execute("SELECT id, text FROM items WHERE user_id = ? AND done = ?", (user_id, 0))    # sqlite3
cur.execute("SELECT id, text FROM items WHERE user_id = %s", (user_id,))                   # psycopg
```
Upsert: `INSERT INTO kv(k, v) VALUES ($1, $2) ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v;`
Timing: `psql -c '\timing' -c "SELECT ..."` or `time sqlite3 app.db "SELECT ..."`.

More rules:
- Prove a slow query with `EXPLAIN (ANALYZE)` (Postgres) or `EXPLAIN QUERY PLAN` (SQLite) before and after the index; keep the lines.

More pitfalls:
- A function on the indexed column (`lower(email)`, `date(created_at)`) skips the index; index the expression or store the normalized value.
- Timeouts in a transaction left open; commit or roll back on every path.
