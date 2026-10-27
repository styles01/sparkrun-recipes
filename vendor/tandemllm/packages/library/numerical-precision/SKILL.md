---
name: numerical-precision
description: Avoiding floating-point mistakes in code that computes money, statistics, physics or geometry - when to use integers or Decimal, comparing with tolerances, summation order, catastrophic cancellation, float32 limits, and rounding for display.
---
# Numerical precision

Rules:
- Money and counts: integers (cents) or `Decimal` with a fixed quantize; never float. Sum in integers, convert for display.
- Compare floats with a tolerance: `math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)` or `np.allclose`; never `==` on computed values. Choose and state the tolerance.
- Know the limits: float64 holds about 15 significant digits; float32 about 7; integers above 2^53 lose precision in JavaScript and in float64.
- Sum many values with `math.fsum` or `np.sum` (pairwise), not a Python loop; sort small to large for a plain loop.

What goes wrong:
- `0.1 + 0.2 != 0.3`; a test with `==` flaps.
- A running total in float drifts after 10^6 additions; the ledger does not balance.
- Variance comes out negative from the one-pass formula.

Check: a known case with an exact answer (sum of 0.1 ten times, a variance of a fixed list, a money total), the error against the exact value printed, and the tolerance used stated.

Done when:
- [ ] One known-answer test per numeric routine with the absolute and relative error printed.
- [ ] Tolerances stated in the tests (`isclose` arguments or `allclose` values).
- [ ] Money paths use integers or Decimal (grep shows no float arithmetic on amounts).

## Reference

```python
import math
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
print(sum([0.1] * 10) == 1.0, math.fsum([0.1] * 10) == 1.0)       # False True
cents = sum(int(round(float(p) * 100)) for p in ["19.99", "0.01"]); print(cents / 100)
d = (Decimal("19.99") * Decimal("1.19")).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
assert math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)
```
Welford mean and variance (one pass, stable):
```python
def mean_var(xs):
    n = 0; mean = 0.0; m2 = 0.0
    for x in xs:
        n += 1; d = x - mean; mean += d / n; m2 += d * (x - mean)
    return mean, (m2 / (n - 1) if n > 1 else 0.0)
```
Log-sum-exp: `m = max(xs); math.log(sum(math.exp(x - m) for x in xs)) + m`.
Stable forms: `math.log1p(x)` for `log(1+x)`, `math.expm1(x)`, `math.hypot(a, b)` for `sqrt(a*a+b*b)`, `2*math.sin(x/2)**2` for `1-cos(x)`.
NumPy: `np.sum` is pairwise and fine; `dtype=np.float64` for accumulators even when data is float32; `np.allclose(a, b, rtol=1e-5, atol=1e-8)` defaults are loose, tighten them on purpose.
JavaScript: integers safe up to `Number.MAX_SAFE_INTEGER` (2^53-1); beyond that `BigInt`. Money in cents as integers; `Intl.NumberFormat('de-DE', { style: 'currency', currency: 'EUR' })` for display.
Geometry: epsilon tests for collinearity `abs(cross) < 1e-9 * scale`; never compare slopes.
Known-answer sources: `math.fsum` as the exact sum; `statistics.variance` from the standard library; `Decimal` with `getcontext().prec = 50` as a high-precision oracle.

More rules:
- Avoid subtracting nearly equal numbers (variance as E[x^2]-E[x]^2, `1 - cos(x)` for small x); use the stable formula (Welford, `2*sin(x/2)^2`, `log1p`, `expm1`, `hypot`).
- Round only at output: `round(x, 2)` for display, `Decimal.quantize` for money, and say which rounding mode (half-even is Python's default).

More pitfalls:
- Overflow to `inf` or underflow to 0 in `exp` of large values; use the log-sum-exp trick.
- Dividing by a tiny number from a near-cancellation; the result explodes.
