"""The walk of a verified draft tree (the body of `Qwen38Engine.accept_tree`, engine/model.py), for every
model: the server's loop calls this, the adapters keep no copy."""

from __future__ import annotations


def accept_tree(tree, picks: list[int]) -> tuple[list[int], list[int]]:
    """Walk the tree along what the target actually wrote. Returns (path, new tokens).

    Greedy verification of a tree is a walk, not a comparison: at each node take the target's own
    argmax and descend into the child carrying that token, stopping where there is none. The tokens
    gained are the path's own plus the target's token at the node it stopped at, so a tree yields at
    least one token exactly as a chain does.
    """
    node, path = 0, [0]
    while True:
        want = picks[node]
        nxt = next((c for c in range(node + 1, len(tree.tokens))
                    if tree.parents[c] == node and tree.tokens[c] == want), None)
        if nxt is None:
            break
        path.append(nxt)
        node = nxt
    new = [tree.tokens[i] for i in path[1:]] + [picks[node]]
    return path, new
