---
name: pandas-cleaning
description: Loading and cleaning tabular data with pandas - reading CSV and Excel, types, missing values, duplicates, dates, string normalization, joins, and proving row counts at every step.
---
# pandas data cleaning

Rules:
- Read with explicit `dtype=` for ids and codes (strings, not numbers), `parse_dates=` for dates, `encoding=` when known. Look at `df.head()`, `df.dtypes`, `df.shape` first.
- Write the expected row count before every step and check after: a filter, a join and a dedup each must explain their delta.
- Missing values: decide per column (drop, fill with a constant, fill from another column) and say why. Count nulls before and after.
- Normalize strings once: `str.strip().str.lower()`, then categorize. Never compare raw strings.

What goes wrong:
- Leading zeros lost (zip codes, ids) because pandas read them as int.
- `SettingWithCopyWarning`: assign on a slice; use `.loc[mask, col] = value` or `.copy()`.
- A join multiplies rows because the key is not unique; `validate=` catches it.

Check: print shape and null counts after each step, `df.describe(include="all")` on the result, and compare three totals (rows, sum of a numeric column, distinct keys) against the raw file.

Done when:
- [ ] A table of row counts per step with the reason for every change in the result.
- [ ] Null counts per column before and after.
- [ ] Output written and read back; shape and one checksum equal; `clean()` idempotent on its own output.

## Reference

```python
import pandas as pd
def load(path):
    return pd.read_csv(path, dtype={"id": "string", "zip": "string"}, parse_dates=["created"], encoding="utf-8")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    log = []
    n = len(df); log.append(("raw", n))
    df = df.drop_duplicates(subset=["id"]); log.append(("dedup id", len(df)))
    df = df[df["amount"].notna()]; log.append(("drop null amount", len(df)))
    df = df.copy()
    df["email"] = df["email"].str.strip().str.lower()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["created"] = pd.to_datetime(df["created"], errors="coerce", utc=True)
    df.loc[df["country"].isna(), "country"] = "unknown"
    print(pd.DataFrame(log, columns=["step", "rows"]).to_string(index=False))
    return df
df = clean(load("in.csv"))
print(df.shape, df.isna().sum().to_dict(), df["amount"].sum())
df.to_parquet("out.parquet"); assert pd.read_parquet("out.parquet").shape == df.shape
```
Join check:
```python
m = orders.merge(customers, on="customer_id", how="left", validate="many_to_one", indicator=True)
print(m["_merge"].value_counts())
```
Useful: `df.info()`, `df[col].value_counts(dropna=False).head(20)`, `df.query("amount < 0")`, `pd.read_excel(path, sheet_name=0, dtype=str)`, `df.memory_usage(deep=True).sum()/1e6`.
State file for the judge: `json.dump({"tandem_state": 1, "rows_in": n, "rows_out": len(df), "nulls": df.isna().sum().to_dict()}, open("{tandem_scratch}/state.json", "w"))`.

More rules:
- Joins with `validate="one_to_one"` or `"many_to_one"` and `indicator=True`; inspect the `_merge` counts.
- Keep the pipeline a function `clean(df) -> df` that runs on 10 rows and on the full file.

More pitfalls:
- Dates parsed with day and month swapped; set `dayfirst=` or a `format=`.
- `inplace=True` and chained methods silently doing nothing.
