---
name: numpy-performance
description: Making numeric Python fast with NumPy - vectorizing loops, broadcasting, dtypes and memory, avoiding copies, and measuring with timeit before and after.
---
# NumPy performance

Rules:
- Measure first: `timeit` the slow function on a realistic size (the real N, not 10). Write the number down before changing anything.
- Replace Python loops over arrays with array expressions: slicing, broadcasting, `np.where`, `np.cumsum`, `np.einsum`, boolean masks.
- Choose the dtype: `float32` halves memory and often doubles speed when 7 digits are enough; `int64` for indices and counts. Say which and why.
- Allocate once outside loops; use `out=` and in-place operators (`a += b`) on hot paths.

What goes wrong:
- `np.append` in a loop: quadratic. Collect in a list, convert once, or preallocate.
- Broadcasting an (N,1) against (N,) creates (N,N) by accident; memory explodes. Print shapes.
- `float64` to `float32` conversion inside the loop every iteration.

Check: time the old and the new version on the same input, show both numbers and the speedup, and `np.allclose(old, new)` with the tolerance stated.

Done when:
- [ ] Before and after timings on the real size in the result (ms, and the ratio).
- [ ] `np.allclose(old, new, rtol=1e-6)` is True (or the differences explained).
- [ ] Peak memory or the array shapes and dtypes printed for the hot arrays.

## Reference

Timing:
```python
import timeit, numpy as np
x = np.random.default_rng(0).random(1_000_000)
t_old = timeit.timeit(lambda: f_old(x), number=5) / 5
t_new = timeit.timeit(lambda: f_new(x), number=5) / 5
print(f"old {t_old*1e3:.1f} ms  new {t_new*1e3:.1f} ms  x{t_old/t_new:.1f}  close={np.allclose(f_old(x), f_new(x))}")
```
Loop to vector:
```python
# before
out = np.empty(len(x))
for i in range(1, len(x)): out[i] = out[i-1] * 0.9 + x[i]
# after (moving sum, distances, conditionals)
m = np.convolve(x, np.ones(5) / 5, mode="valid")
d = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))   # (N, M) distances; watch memory N*M*8 bytes
y = np.where(x > 0.5, x, 0.0)
```
Preallocate and in-place: `acc = np.zeros(n, dtype=np.float32); np.add(acc, x, out=acc)`.
Memory: `a.nbytes / 1e6` MB; `np.shares_memory(a, b)` to check views; `np.ascontiguousarray(a)` before passing to C code.
numba for an unavoidable recurrence:
```python
from numba import njit
@njit(cache=True)
def ewma(x, a):
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1 - a) * out[i-1]
    return out
```
Peak memory: `/usr/bin/time -l python script.py` (macOS) or `-v` (Linux), read "maximum resident set size".

More rules:
- Know what copies: fancy indexing and `np.concatenate` copy; slices and `reshape` of contiguous arrays do not. Check `a.flags["C_CONTIGUOUS"]` before a hot loop.
- Beyond NumPy, in this order: `numba` for an unavoidable loop, then `scipy` routines, then a GPU library, each justified by a measured number.

More pitfalls:
- A "vectorized" version that gives different results from the loop; compare with `np.allclose` before trusting the speedup.
