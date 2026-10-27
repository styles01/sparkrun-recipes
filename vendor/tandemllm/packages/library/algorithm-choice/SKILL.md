---
name: algorithm-choice
description: Choosing a data structure or algorithm for a task - sizing the input, complexity targets, hash maps versus sorted structures, sets, heaps, graphs (BFS, DFS, Dijkstra), dynamic programming, and checking with a brute-force oracle.
---
# Algorithm and data structure choice

Rules:
- Size the input first: items (n), queries (q), how often it runs. n up to 10^4: anything works; 10^6: O(n log n) or better; 10^8: linear or a different approach.
- Pick by the operation: membership and lookup by key = hash map or set (O(1)); ordered iteration, range queries, nearest = sorted array with bisect or a tree; repeated min or max = heap; FIFO = deque; LIFO = list.
- Graphs: unweighted shortest path = BFS; reachability or components = DFS or union-find; weighted non-negative = Dijkstra with a heap; dependencies = topological sort.
- Overlapping subproblems with optimal substructure = dynamic programming; write the recurrence and the base case before code, memoize first, tabulate if memory allows.

What goes wrong:
- `x in list` inside a loop: O(n^2) hiding in one line. Use a set.
- Sorting on every query instead of once.
- Recursion depth on n = 10^5 (Python limit 1000); use an explicit stack or iteration.

Check: measured time at the real n, a 100-case comparison with the brute force (all equal), and the edges (empty, one, all equal, max size).

Done when:
- [ ] The chosen structure and its complexity named with the real n and q.
- [ ] Random comparison with a brute force: 100 cases, 0 mismatches (output shown).
- [ ] Timing at the real size in the result.

## Reference

Oracle test:
```python
import random
def brute(xs, k): return sum(1 for i in range(len(xs)) for j in range(i+1, len(xs)) if xs[i] + xs[j] == k)
def fast(xs, k):
    from collections import Counter
    c = Counter(); n = 0
    for x in xs: n += c[k - x]; c[x] += 1
    return n
for _ in range(100):
    xs = [random.randint(-5, 5) for _ in range(random.randint(0, 12))]; k = random.randint(-10, 10)
    assert brute(xs, k) == fast(xs, k), (xs, k)
print("oracle ok")
```
Pieces:
```python
import bisect, heapq
from collections import deque, Counter, defaultdict
i = bisect.bisect_left(sorted_xs, target)              # first index >= target
heapq.heappush(h, (dist, node)); d, u = heapq.heappop(h)
def bfs(adj, s):
    dist = {s: 0}; q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist: dist[v] = dist[u] + 1; q.append(v)
    return dist
def dijkstra(adj, s):           # adj[u] = [(v, w)]
    dist = {s: 0}; h = [(0, s)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]: continue
        for v, w in adj[u]:
            if d + w < dist.get(v, float("inf")): dist[v] = d + w; heapq.heappush(h, (d + w, v))
    return dist
```
DP shape: `@functools.lru_cache(None) def f(i, rem): if i == n: return 0 if rem == 0 else -inf; return max(f(i+1, rem), val[i] + f(i+1, rem-w[i]) if rem >= w[i] else -inf)`; raise the limit with `sys.setrecursionlimit(10**6)` or tabulate.
Sizes: Python does about 10^7 simple operations a second; a dict of 10^6 ints uses about 100 MB.

More rules:
- Use the standard library (`sorted`, `bisect`, `heapq`, `collections.Counter`, `functools.lru_cache`) before writing a structure.
- Verify against a brute force on random small inputs (100 cases) before trusting the fast version.

More pitfalls:
- Floating-point keys in a dict or set; round or use integers.
- An approach correct on the example but exponential on the real size; count the operations with the real n.
